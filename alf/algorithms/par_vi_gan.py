# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import logging
import functools
import gin
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from typing import Callable

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.config import TrainerConfig
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.algorithms.generator import Generator
from alf.networks import EncodingNetwork, ImageDecodingNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops, summary_utils
from alf.utils.summary_utils import record_time

AdversarialImageGeneratorLossInfo = namedtuple(
    "AdversarialImageGeneratorLossInfo", ["loss"])

@gin.configurable
class AdversarialImageGenerator(Algorithm):
    """AdversarialImageGenerator 

    Generator algorithm that is used for learning to sample from 
    a distribution of natural images. Similar to GAN training, the generator
    samples from a parameterized distribution, and transforms samples
    from this distribution into samples from the target image distribution

    Unlike GANs, this ImageGenerator is optimized with the minmax amortized
    svgd algorithm to fit the target density without using a 2-sample 
    discriminator function. Instead we use a critic function that depends only
    on the score function of the generator. 
    It is inspired oby the following works:

    Hu, Tianyang, et al. "Stein neural sampler."
    arXiv preprint arXiv:1810.03545 (2018).

    Grathwohl, Will, et al. "Learning the Stein Discrepancy for Training and
    Evaluating Energy-Based Models without Sampling" ICML, 2020. 
    
    """

    def __init__(self,
                 input_tensor_spec,
                 generator_conv_layer_params,
                 discriminator_conv_layer_params,
                 activation=torch.relu_,
                 final_activation_generator=math_ops.identity,
                 final_activation_discriminator=math_ops.identity,
                 noise_dim=64,
                 optimizer=None,
                 logging_network=False,
                 logging_training=False,
                 logging_evaluate=False,
                 config: TrainerConfig = None,
                 name="ImageGenerator"):
        """
        Args:

            Args for the generator
            ====================================================================
            noise_dim (int): dimension of noise
            hidden_layers (tuple): size of hidden layers.
            use_fc_bn (bool): whether use batnch normalization for fc layers.
            particles (int): number of sampling particles
            entropy_regularization (float): weight of entropy regularization

            Args for training and testing
            ====================================================================
            loss_type (str): loglikelihood type for the generated functions,
                types are [``classification``, ``regression``]
            par_vi (str): types of particle-based methods for variational inference,
                currently, only ``minmax`` is supported
            optimizer (torch.optim.Optimizer): The optimizer for training.
            logging_network (bool): whether logging the archetectures of networks.
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            config (TrainerConfig): configuration for training
            name (str):
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)

        gen_output_dim = np.prod(input_tensor_spec.shape)
        noise_spec = TensorSpec(shape=(noise_dim, ))
        

        gen_output_tensor_spec = input_tensor_spec

        transconv_layer_params = (
            (noise_dim * 4, 4, 2, 1),
            (noise_dim * 2, 4, 2, 1),
            (noise_dim,  4, 2, 1),
            (1,          4, 2, 1))
        start_decoding_size = 2
        start_decoding_channels = 8 * noise_dim

        net = ImageDecodingNetwork(
            input_size=noise_dim,
            transconv_layer_params=generator_conv_layer_params,
            start_decoding_size=start_decoding_size,
            start_decoding_channels=start_decoding_channels,
            activation=activation,
            output_activation=last_activation_generator,
            name="Generator")
        
        mnist_spec = TensorSpec(shape=(1, 32, 32, ))
        conv_layer_params = (
            (noise_dim,     4, 2, 1),
            (noise_dim * 2, 4, 2, 1),
            (noise_dim * 4, 4, 2, 1),
            (noise_dim * 8, 4, 2, 1))

        discriminator = EncodingNetwork(
            input_tensor_spec=input_tensor_spec,
            conv_layer_params=discriminator_conv_layer_params,
            activation=activation
            last_layer_size=1,
            last_activation=last_activation_discriminator,
            name='Discriminator')

        if par_vi == 'minmax':
            critic = EncodingNetwork(
                TensorSpec(shape=(gen_output_dim, )),
                conv_layer_params=None,
                fc_layer_params=(1024, 1024, ),
                activation=torch.relu_,
                last_layer_size=gen_output_dim,
                last_activation=math_ops.identity,
                name="Critic")
        else:
            critic = None
        
        if logging_network:
            logging.info("Generator network")
            logging.info("-" * 68)
            logging.info(net)
            
            logging.info("Discriminator network")
            logging.info("-" * 68)
            logging.info(discriminator)
            
            logging.info("Critic network")
            logging.info("-" * 68)
            logging.info(critic)

        self._generator = Generator(
            gen_output_dim,
            noise_dim=noise_dim,
            net=net,
            entropy_regularization=entropy_regularization,
            par_vi=par_vi,
            critic=critic,
            optimizer=None,
            name=name)
        
        self.discriminator = discriminator

        self._disc_iters = 5
        self._critic_iters = 5

        self._input_shape = input_tensor_spec.shape
        self._batch_size = batch_size
        self._entropy_regularization = entropy_regularization
        self._train_loader = None
        self._test_loader = None
        self._use_fc_bn = use_fc_bn
        self._loss_type = loss_type
        self._logging_training = logging_training
        self._logging_evaluate = logging_evaluate
        self._config = config

    def set_data_loader(self, train_loader, test_loader=None):
        """Set data loadder for training and testing."""
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._entropy_regularization = 1 / len(train_loader)

    def sample_outputs(self, noise=None, batch_size=None, training=True):
        "Sample images from an ensemble of generators"
        if batch_size is None:
            batch_size = self._batch_size
        generator_step = self._generator.predict_step(
            noise=noise, batch_size=batch_size, training=training)
        return generator_step.output

    def train_iter(self, state=None):
        """Perform one epoch (iteration) of training."""

        assert self._train_loader is not None, "Must set data_loader first."
        alf.summary.increment_global_counter()
        with record_time("time/train"):
            loss = 0.
            for batch_idx, (data, _) in enumerate(self._train_loader):
                data = data.to(alf.get_default_device())
                if batch_idx % (self._critic_iters + 1):
                    model = 'critic'
                else:
                    model = 'generator'
                alg_step = self.train_step(data,
                                           model=model,
                                           state=state)
                loss_info, samples = self.update_with_gradient(alg_step.info)
                self._generator.after_update(alg_step.info)
                loss += loss_info.loss
                print (batch_idx)

                if batch_idx % 50 == 0:
                    samples = self.sample_outputs().detach()
                    self._save_samples(
                            samples,
                            step=alf.summary.get_global_counter(),
                            batch=batch_idx)
        return batch_idx + 1

    def train_step(self,
                   inputs,
                   entropy_regularization=None,
                   model=None,
                   state=None):
        """Perform one batch of training computation.

        Args:
            inputs (nested Tensor): input training data. 
            model (str): 
            state: not used

        Returns:
            AlgorithmStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
        if model == 'critic':
            for param in self.discriminator.parameters():
                param.requires_grad = True
            disc_data = self.discriminator(inputs)[0]
            with torch.no_grad():
                samples = self.sample_outputs()
            samples.requires_grad_(True)
            disc_samples = self.discriminator(samples)[0]
            grad_penalty = self._gradient_penalty(inputs, samples)
            loss = (-1 * disc_data.mean()) + disc_samples.mean() + grad_penalty

        else:
            for param in self.discriminator.parameters():
                param.requires_grad = False
            samples = self.sample_outputs()
            disc_samples = self.discriminator(samples)[0].mean()
            loss = -1 * disc_samples
        
        return AlgStep(
            output=samples,
            state=(),
            info=LossInfo(
                loss=loss,
                extra=AdversarialImageGeneratorLossInfo(loss=loss)))

        """
        outputs = self.sample_outputs()
        if entropy_regularization is None:
            entropy_regularization = self._entropy_regularization

        return self._generator.train_step(
            inputs=None,
            loss_func=functools.partial(neglogprob, inputs,
                                        self._loss_type),
            entropy_regularization=entropy_regularization,
            outputs=outputs,
            model=model,
            state=())
        """
    def evaluate(self, particles=None):
        """Evaluate on a randomly drawn network. """
        assert self._test_loader is not None, "Must set test_loader first."
        logging.info("==> Begin testing")
        with record_time("time/test"):
            disc_data_loss = 0.
            disc_samples_loss = 0.
            w1_distance = 0
            for i, (data, _) in enumerate(self._test_loader):
                data = data.to(alf.get_default_device())
                disc_data = self.discriminator(data)[0]
                disc_data_cost = -1 * disc_data.mean()
                samples = self.sample_outputs()
                disc_samples_cost = self.discriminator(samples)[0].mean()
                disc_data_loss += disc_data_cost.item()
                disc_samples_loss += disc_samples_cost.item()
                w1_distance += (disc_data_cost - disc_samples_cost).item()

        if self._logging_evaluate:
            logging.info("Test Disc Data Loss: {}".format(disc_data_loss))
            logging.info("Test Disc Samples Loss: {}".format(disc_samples_loss))
            logging.info("Test W1 Distance: {}".format(w1_distance))
        alf.summary.scalar(name='eval/test_disc_data_loss', data=disc_data_loss)
        alf.summary.scalar(name='eval/test_disc_samples_loss', data=disc_samples_loss)
        alf.summary.scalar(name='eval/test_w1_dist', data=w1_distance)
    
    def _gradient_penalty(self, data, samples, lamb=10.0):
        """ computes a gradient penalty (wgan-gp) """
        data = data.view(data.shape[0], -1)
        samples = samples.view(samples.shape[0], -1)
        alpha = torch.randn(data.shape[0], 1, requires_grad=True)
        alpha = alpha.to(alf.get_default_device())
        alpha = alpha.expand(data.size())
        interpolates = alpha * data + (1 - alpha) * samples
        disc_inputs = interpolates.view(-1, *self._input_shape)
        disc_interpolates = self.discriminator(disc_inputs)[0]
        grad_outputs = torch.ones(disc_interpolates.size())
        grad_outputs = grad_outputs.to(alf.get_default_device())
        grads = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_penalty = ((grads.norm(2, dim=1) - 1)**2).mean() * lamb
        return grad_penalty


    def _spectral_norm(self, module):
        if 'weight' in module._parameters:
            torch.nn.utils.spectral_norm(module)

    def _save_samples(self, samples, step, batch):
        assert samples.dim() == 4, "samples are of the wrong shape "\
            "expected dim=4, but got dim={}".format(samples.dim())
        
        path = 'generated_samples_iter_{}_batch_{}.png'.format(step, batch)
        print (samples.shape)
        rows = int(samples.shape[0]**.5)
        torchvision.utils.save_image(samples, path, nrow=rows)

    def summarize_train(self, samples, loss_info, cum_loss=None):
        """Generate summaries for training & loss info after each gradient update.
        The default implementation of this function only summarizes params
        (with grads) and the loss. An algorithm can override this for additional
        summaries. See ``RLAlgorithm.summarize_train()`` for an example.

        Args:
            experience (nested Tensor): samples used for the most recent
                ``update_with_gradient()``. By default it's not summarized.
            train_info (nested Tensor): ``AlgStep.info`` returned by either
                ``rollout_step()`` (on-policy training) or ``train_step()``
                (off-policy training). By default it's not summarized.
            loss_info (LossInfo): loss
            params (list[Parameter]): list of parameters with gradients
        """
        if self._config.summarize_grads_and_vars:
            summary_utils.summarize_variables(samples)
            summary_utils.summarize_gradients(samples)
        if self._config.debug_summaries:
            summary_utils.summarize_loss(loss_info)
        if cum_loss is not None:
            alf.summary.scalar(name='train_epoch/adversarial_likelihood', data=cum_loss)
