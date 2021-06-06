# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import os
import functools
import gin
import sys
import math
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from typing import Callable

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.config import TrainerConfig
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.algorithms.generator2 import Generator
from alf.networks import EncodingNetwork, ImageDecodingNetwork, ReluMLP
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops, summary_utils
from alf.utils.summary_utils import record_time

GenerativeAdversarialLossInfo = namedtuple("GANLossInfo", ["loss"])


@alf.configurable
class GenerativeAdversarialAlgorithm(Algorithm):
    """GAN 
    """

    def __init__(self,
                 output_dim,
                 input_tensor_spec=None,
                 conv_layer_params=None,
                 critic_conv_layer_params=None,
                 scaler=None,
                 net=None,
                 critic=None,
                 activation=torch.relu_,
                 last_layer_param=None,
                 last_activation=math_ops.identity,
                 last_activation_critic=math_ops.identity,
                 use_bn=False,
                 use_critic_bn=False,
                 grad_lambda=0.,
                 metric='jsd',
                 critic_weight_clip=0.01,
                 noise_dim=64,
                 entropy_regularization=1.,
                 critic_iter_num=5,
                 functional_gradient=None,
                 block_pinverse=False,
                 jac_autograd=True,
                 force_fullrank=True,
                 fullrank_diag_weight=1.0,
                 pinverse_solve_iters=1,
                 pinverse_hidden_size=100,
                 par_vi=None,
                 expectation_logp=True,
                 use_kernel_logp=True,
                 critic_optimizer=None,
                 pinverse_optimizer=None,
                 optimizer=None,
                 logging_network=False,
                 logging_training=False,
                 logging_evaluate=False,
                 config: TrainerConfig = None,
                 name="GAN"):
        """
        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format 
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            fc_layer_params (tuple[tuple]): a tuple of tuples where each tuple
                takes a format ``(FC layer sizes. use_bias)``, where 
                ``use_bias`` is optional.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            last_layer_param (tuple): an optional tuple of the format
                ``(size, use_bias)``, where ``use_bias`` is optional,
                it appends an additional layer at the very end. 
                Note that if ``last_activation`` is specified, 
                ``last_layer_param`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_param``. Note that if
                ``last_layer_param`` is not None, ``last_activation`` has to be
                specified explicitly.

            noise_dim (int): dimension of noise
            hidden_layers (tuple): size of hidden layers.
            use_fc_bn (bool): whether use batnch normalization for fc layers.

            critic_optimizer (torch.optim.Optimizer): the optimizer for training critic.
            critic_hidden_layers (tuple): sizes of critic hidden layeres. 
            critic_iter_num (int)
            critic_l2_weight (float)

            functional_gradient (bool)
            force_fullrank (bool)
            fullrank_diag_weight (float)
            pinverse_solve_iters (int)
            pinverse_hidden_size (int)

            par_vi (str): types of particle-based methods for variational inference,
                types are [``svgd3``],

                * svgd3: empirical expectation of SVGD is evaluated by 
                    resampled particles of the same batch size. It has better
                    convergence but involves resampling, so less efficient
                    computaionally comparing with svgd2.
            critic_optimizer (torch.optim.Optimizer)
            pinverse_optimizer (torch.optim.Optimizer)
            optimizer (torch.optim.Optimizer): The optimizer for training generator.
            logging_network (bool): whether logging the archetectures of networks.
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            config (TrainerConfig): configuration for training
            name (str):
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)

        noise_spec = TensorSpec(shape=(noise_dim, ))
        start_decoding_size = 2
        start_decoding_channels = 8 * 128

        gen_output_dim = output_dim

        if functional_gradient:
            if net is None:
                net = ReluMLP(
                    noise_spec,
                    hidden_layers=hidden_layers,
                    output_size=gen_output_dim,
                    name='Generator')
        else:
            if net is None:
                net = ImageDecodingNetwork(
                    input_size=noise_dim,
                    transconv_layer_params=conv_layer_params,
                    start_decoding_size=start_decoding_size,
                    start_decoding_channels=start_decoding_channels,
                    activation=activation,
                    use_conv_bn=use_bn,
                    output_activation=last_activation,
                    name='Generator')

        if critic is None:
            critic = EncodingNetwork(
                input_tensor_spec=input_tensor_spec,
                conv_layer_params=critic_conv_layer_params,
                activation=activation,
                use_conv_bn=use_critic_bn,
                last_layer_size=1,
                last_activation=last_activation_critic,
                name='Critic')

        if logging_network:
            logging.info("Generator network")
            logging.info("-" * 68)
            logging.info(net)

            logging.info("Critic network")
            logging.info("-" * 68)
            logging.info(critic)

        if par_vi == 'svgd':
            par_vi = 'svgd3'

        self._generator = Generator(
            gen_output_dim,
            noise_dim=noise_dim,
            net=net,
            entropy_regularization=entropy_regularization,
            par_vi=par_vi,
            functional_gradient=functional_gradient,
            block_pinverse=block_pinverse,
            force_fullrank=force_fullrank,
            jac_autograd=jac_autograd,
            expectation_logp=expectation_logp,
            use_kernel_logp=use_kernel_logp,
            fullrank_diag_weight=fullrank_diag_weight,
            pinverse_solve_iters=pinverse_solve_iters,
            pinverse_hidden_size=pinverse_hidden_size,
            optimizer=None,
            name=name)

        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.optimizer = optimizer
        assert metric in ['jsd', 'w1', 'kl-w1'], "metric to minimize must " \
            "be ``jsd``, ``w1``, or ``kl-w1``"
        self._scaler = scaler
        self._metric = metric
        self._critic_iter_num = critic_iter_num
        self._grad_lambda = grad_lambda
        self._critic_weight_clip = critic_weight_clip
        self._par_vi = par_vi
        self._input_shape = input_tensor_spec.shape
        self._entropy_regularization = entropy_regularization
        self._train_loader = None
        self._test_loader = None
        self._functional_gradient = functional_gradient
        self._logging_training = logging_training
        self._logging_evaluate = logging_evaluate
        self._config = config

    def _trainable_attributes_to_ignore(self):
        return ["critic"]

    def set_data_loader(self,
                        train_loader,
                        test_loader=None,
                        data_transform=None,
                        entropy_regularization=None):
        """Set data loadder for training and testing.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
            test_loader (torch.utils.data.DataLoader): testing data loader
            data_transform (Callable): function to transform data before 
                train step. Function must take (data, target) tensors and
                return a tuple of (transformed data, transformed target)
                tensors. 
        """
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._data_transform = data_transform
        if entropy_regularization is not None:
            self._entropy_regularization = entropy_regularization

    def sample_outputs(self, noise=None, batch_size=None, training=True):
        "Sample images"
        if batch_size is None:
            batch_size = self._batch_size
        generator_step = self._generator.predict_step(
            noise=noise, batch_size=batch_size, training=training)
        return generator_step.output

    def train_iter(self, save_samples=False, state=None):
        """Perform one epoch (iteration) of training."""
        import time
        assert self._train_loader is not None, "Must set data_loader first."
        alf.summary.increment_global_counter()
        with record_time("time/train"):
            loss = 0.
            cum_d_loss = 0.
            cum_g_loss = 0.
            pinverse_loss = 0.
            for batch_idx, (data, _) in enumerate(self._train_loader):
                data = data.to(alf.get_default_device())
                if batch_idx % (self._critic_iter_num + 1):
                    alg_step = self.train_step(
                        data, model='critic', state=state)
                    d_loss = alg_step.info.loss
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    d_loss.backward()
                    self.critic_optimizer.step()
                    cum_d_loss += d_loss.item()
                else:
                    alg_step = self.train_step(
                        data, model='generator', state=state)
                    g_loss = alg_step.info.loss
                    loss_info, samples = self.update_with_gradient(
                        alg_step.info, scaler=self._scaler)
                    if self._functional_gradient:
                        pinverse_loss += loss_info.extra.pinverse.item()
                    cum_g_loss += g_loss.mean().item()

        if self._logging_training:
            logging.info("Cum G Loss: {}".format(cum_g_loss))
            logging.info("Cum D Loss: {}".format(cum_d_loss))
            if self._functional_gradient:
                logging.info("Avg Pinverse Loss: {}".format(
                    pinverse_loss / batch_idx))
        if save_samples:
            samples = self.sample_outputs(batch_size=data.shape[0])
            if isinstance(samples, tuple):
                _, samples = samples
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
        if model == 'critic':
            for param in self.critic.parameters():
                param.requires_grad = True
                if self._critic_weight_clip > 0:
                    param.data.clamp_(-self._critic_weight_clip,
                                      self._critic_weight_clip)
            self.critic.zero_grad()  #set_to_none=True)
            critic_data = self.critic(inputs)[0]
            with torch.no_grad():
                samples = self.sample_outputs(batch_size=inputs.shape[0])
            if isinstance(samples, tuple):
                _, samples = samples
            samples.requires_grad_(True)
            critic_samples = self.critic(samples)[0]
            if self._grad_lambda != 0.:
                grad_penalty = self._gradient_penalty(inputs, samples)
            else:
                grad_penalty = 0.
            if self._metric in ['w1', 'kl-w1']:
                if self._metric == 'w1':
                    loss_data = -1 * critic_data.mean()
                    loss_samples = critic_samples.mean()
                else:
                    loss_data, loss_samples = self._kl_critic_loss(
                        critic_data, critic_samples)
                loss = loss_data + loss_samples + grad_penalty
            elif self._metric == 'jsd':
                real_labels = torch.ones(inputs.shape[0]).float()
                loss_data = F.binary_cross_entropy_with_logits(
                    critic_data, real_labels)
                fake_labels = torch.zeros(samples.shape[0]).float()
                loss_samples = F.binary_cross_entropy_with_logits(
                    critic_samples, fake_labels)
                loss = loss_samples + loss_data + grad_penalty

            step = AlgStep(
                output=samples,
                state=(),
                info=LossInfo(
                    loss=loss, extra=GenerativeAdversarialLossInfo(loss=loss)))

        else:
            for param in self.critic.parameters():
                param.requires_grad = False
            self._generator._net.zero_grad(set_to_none=True)
            step = self._generator.train_step(
                inputs=None,
                loss_func=self._critic_loss,
                batch_size=inputs.shape[0],
                entropy_regularization=self._entropy_regularization,
                state=state)
        return step

    def _critic_loss(self, inputs):
        outputs = self.critic(inputs)[0]
        if self._metric == 'w1':
            loss = -1 * outputs
        elif self._metric == 'jsd':
            labels = torch.ones(inputs.shape[0]).float()
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
        elif self._metric == 'kl-w1':
            loss = self._kl_gen_loss(outputs)
        return loss

    def evaluate(self):
        """Evaluate on a randomly drawn network. """
        assert self._test_loader is not None, "Must set test_loader first."
        logging.info("==> Begin testing")
        with record_time("time/test"):
            critic_data_loss = 0.
            critic_samples_loss = 0.
            w1_distance = 0
            for i, (data, _) in enumerate(self._test_loader):
                data = data.to(alf.get_default_device())
                critic_data = self.critic(data)[0]
                critic_data_cost = -1 * critic_data.mean()
                samples = self.sample_outputs(batch_size=data.shape[0])
                critic_samples_cost = self.critic(samples)[0].mean()
                critic_data_loss += critic_data_cost.item()
                critic_samples_loss += critic_samples_cost.item()
                if self._metric == 'w1':
                    w1_distance += (
                        critic_data_cost - critic_samples_cost).item()

        if self._logging_evaluate:
            logging.info("Test Disc Data Loss: {}".format(critic_data_loss))
            logging.info(
                "Test Disc Samples Loss: {}".format(critic_samples_loss))
            logging.info("Test W1 Distance: {}".format(w1_distance))
        alf.summary.scalar(
            name='eval/test_critic_data_loss', data=critic_data_loss)
        alf.summary.scalar(
            name='eval/test_critic_samples_loss', data=critic_samples_loss)
        if self._metric == 'w1':
            alf.summary.scalar(name='eval/test_w1_dist', data=w1_distance)

    def _kl_gen_loss(self, samples):
        samples_norm = torch.exp(samples).mean()
        samples_r = torch.exp(samples) / samples_norm
        samples = samples * samples_r
        loss = -1 * samples.mean()
        return loss

    def _kl_critic_loss(self, data, samples):
        loss_data = F.relu(1. - data).mean()
        samples_norm = torch.exp(samples).mean()
        samples_r = torch.exp(samples) / samples_norm
        samples = samples * samples_r
        loss_samples = F.relu(1. + samples).mean()
        return loss_data, loss_samples

    def _gradient_penalty(self, data, samples):
        """ computes a gradient penalty (wgan-gp) """
        data = data.view(data.shape[0], -1)
        samples = samples.view(samples.shape[0], -1)
        alpha = torch.randn(data.shape[0], 1, requires_grad=True)
        alpha = alpha.to(alf.get_default_device())
        alpha = alpha.expand(data.size())
        interpolates = alpha * data + (1 - alpha) * samples
        critic_inputs = interpolates.view(-1, *self._input_shape)
        critic_interpolates = self.critic(critic_inputs)[0]
        grad_outputs = torch.ones(critic_interpolates.size())
        grad_outputs = grad_outputs.to(alf.get_default_device())
        grads = torch.autograd.grad(
            outputs=self._scaler.scale(critic_interpolates),
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_penalty = (
            (grads.norm(2, dim=1) - 1)**2).mean() * self._grad_lambda
        return grad_penalty

    def _spectral_norm(self, module):
        if 'weight' in module._parameters:
            torch.nn.utils.spectral_norm(module)

    def _save_samples(self, samples, step, batch):
        assert samples.dim() == 4, "samples are of the wrong shape "\
            "expected dim=4, but got dim={}".format(samples.dim())

        path = '/nfs/hpc/share/ratzlafn/alf-plots/gan/mnist/'
        os.makedirs(path, exist_ok=True)
        fn = path + 'samples_iter_{}_batch_{}.png'.format(step, batch)
        rows = int(samples.shape[0]**.5)
        torchvision.utils.save_image(samples, fn, nrow=rows)

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
            alf.summary.scalar(
                name='train_epoch/adversarial_likelihood', data=cum_loss)
