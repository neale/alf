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
                 last_activation_generator=math_ops.identity,
                 last_activation_discriminator=math_ops.identity,
                 noise_dim=64,
                 discriminator_training_iters=5.,
                 generator_optimizer=None,
                 discriminator_optimizer=None,
                 grad_lambda=0.,
                 discriminator_weight_clip=0.01,
                 generator_bn=False,
                 discriminator_bn=False,
                 last_layer_param=None,
                 last_activation=None,
                 optimizer=None,
                 logging_network=False,
                 logging_training=False,
                 logging_evaluate=False,
                 config: TrainerConfig = None,
                 name="AdversarialImageGenerator"):
        """
        Args:

            Args for the generator
            ====================================================================
            noise_dim (int): dimension of noise

            Args for training and testing
            ====================================================================
            optimizer (torch.optim.Optimizer): The optimizer for training.
            last_layer_param (None): unused
            last_activation (None): unused
            logging_network (bool): whether logging the archetectures of networks.
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            config (TrainerConfig): configuration for training
            name (str):
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)

        noise_spec = TensorSpec(shape=(noise_dim, ))

        start_decoding_size = 2
        start_decoding_channels = 8 * 64

        net = ImageDecodingNetwork(
            input_size=noise_dim,
            transconv_layer_params=generator_conv_layer_params,
            start_decoding_size=start_decoding_size,
            start_decoding_channels=start_decoding_channels,
            activation=activation,
            use_conv_bn=generator_bn,
            output_activation=last_activation_generator,
            name="Generator")
        
        discriminator = EncodingNetwork(
            input_tensor_spec=input_tensor_spec,
            conv_layer_params=discriminator_conv_layer_params,
            activation=activation,
            use_conv_bn=discriminator_bn,
            last_layer_size=1,
            last_activation=last_activation_discriminator,
            name='Discriminator')

        if logging_network:
            logging.info("Generator network")
            logging.info("-" * 68)
            logging.info(net)
            
            logging.info("Discriminator network")
            logging.info("-" * 68)
            logging.info(discriminator)

        self._generator = Generator(
            np.prod(input_tensor_spec),
            noise_dim=noise_dim,
            net=net,
            entropy_regularization=0.,
            par_vi=None,
            optimizer=None,
            name=name)
        
        self.discriminator = discriminator

        self._disc_iters = discriminator_training_iters
        self._grad_lambda = grad_lambda
        self._disc_weight_clip = discriminator_weight_clip

        self._input_shape = input_tensor_spec.shape
        self._train_loader = None
        self._test_loader = None
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
                if batch_idx % (self._disc_iters + 1):
                    model = 'discriminator'
                else:
                    model = 'generator'
                alg_step = self.train_step(data,
                                           model=model,
                                           state=state)
                loss_info, samples = self.update_with_gradient(alg_step.info)
                self._generator.after_update(alg_step.info)
                loss += loss_info.loss



        samples = self.sample_outputs(batch_size=data.shape[0])
        samples = samples.detach()
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
        if model == 'discriminator':
            for param in self.discriminator.parameters():
                param.requires_grad = True
                if self._disc_weight_clip > 0:
                    param.data.clamp_(
                        -self._disc_weight_clip,
                    self._disc_weight_clip)
            
            disc_data = self.discriminator(inputs)[0]
            with torch.no_grad():
                samples = self.sample_outputs(batch_size=inputs.shape[0])
            samples.requires_grad_(True)
            disc_samples = self.discriminator(samples)[0]
            if self._grad_lambda != 0.:
                grad_penalty = self._gradient_penalty(
                        inputs,
                        samples,
                        self._grad_lambda)
            else:
                grad_penalty = 0.
                
            loss = (-1 * disc_data.mean()) + (disc_samples.mean()) + grad_penalty

        else:
            self._generator._net.zero_grad()
            for param in self.discriminator.parameters():
                param.requires_grad = False
            samples = self.sample_outputs(batch_size=inputs.shape[0])
            disc_samples = self.discriminator(samples)[0].mean()
            loss = -1 * disc_samples
        
        return AlgStep(
            output=samples,
            state=(),
            info=LossInfo(
                loss=loss,
                extra=AdversarialImageGeneratorLossInfo(loss=loss)))

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
                samples = self.sample_outputs(batch_size=data.shape[0])
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
        
        path = 'wgan_cp_samples_iter_{}_batch_{}.png'.format(step, batch)
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
