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
"""A generic generator."""

import gin
import numpy as np
import torch

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.mi_estimator import MIEstimator
from alf.data_structures import AlgStep, LossInfo, namedtuple
import alf.nest as nest
from alf.networks import Network, EncodingNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops
from alf.utils.averager import AdaptiveAverager


@gin.configurable
class CriticAlgorithm(Algorithm):
    """
    Wrap a critic network as an Algorithm for flexible gradient updates
    called by the Generator when par_vi is 'minmax'.
    """

    def __init__(self,
                 input_tensor_spec,
                 output_dim=None,
                 hidden_layers=(3, 3),
                 activation=torch.relu_,
                 net: Network = None,
                 use_bn=True,
                 optimizer=None,
                 name="CriticAlgorithm"):
        """Create a CriticAlgorithm.
        Args:
            input_tensor_spec (TensorSpec): spec of inputs. 
            output_dim (int): dimension of output, default value is input_dim.
            hidden_layers (tuple): size of hidden layers.
            activation (nn.functional): activation used for all critic layers.
            net (Network): network for predicting outputs from inputs.
                If None, a default one with hidden_layers will be created
            use_bn (bool): whether use batch norm for each critic layers.
            optimizer (torch.optim.Optimizer): (optional) optimizer for training.
            name (str): name of this CriticAlgorithm.
        """
        if optimizer is None:
            optimizer = alf.optimizers.Adam(lr=1e-3)
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)

        self._input_dim = input_tensor_spec.shape[0]
        self._output_dim = output_dim
        if output_dim is None:
            self._output_dim = input_tensor_spec.shape[0]
        if net is None:
            net = EncodingNetwork(
                input_tensor_spec=input_tensor_spec,
                fc_layer_params=hidden_layers,
                use_fc_bn=use_bn,
                activation=activation,
                last_layer_size=self._output_dim,
                last_activation=math_ops.identity,
                last_use_fc_bn=use_bn,
                name='Critic')
        self._net = net

    def reset_net_parameters(self):
        for fc in self._net._fc_layers:
            fc.reset_parameters()

    def predict_step(self, inputs, state=None):
        """Predict for one step of inputs.
        Args:
            inputs (torch.Tensor): inputs for prediction.
            state: not used.
        Returns:
            AlgStep:
            - output (torch.Tensor): predictions or (predictions, diag_jacobian)
                if requires_jac_diag is True.
            - state: not used.
        """
        outputs = self._net(inputs)[0]
        return AlgStep(output=outputs, state=(), info=())


@gin.configurable
class ParVIAlgorithm(Algorithm):
    """ParVIAlgorithm

    ParVIAlgorithm maintains a set of particles that keep chasing some target
    distribution. Two particle-based variational inference (par_vi) methods 
    are implemented:

        1. Stein Variational Gradient Descent (SVGD):

        Feng et al "Learning to Draw Samples with Amortized Stein Variational
        Gradient Descent" https://arxiv.org/pdf/1707.06626.pdf

        2. Wasserstein Particle-based VI with Smooth Functions (GFSF):

        Liu, Chang, et al. "Understanding and accelerating particle-based 
        variational inference." International Conference on Machine Learning. 2019.
    """

    def __init__(self,
                 particle_dim,
                 train_state_spec=(),
                 num_particles=10,
                 entropy_regularization=1.,
                 par_vi="gfsf",
                 critic_input_dim=None,
                 critic_iter_num=2,
                 critic_l2_weight=10.,
                 critic_use_bn=True,
                 critic_hidden_layers=(100,100),
                 optimizer=None,
                 critic_optimizer=None,
                 debug_summaries=False,
                 name="ParVIAlgorithm"):
        r"""Create a ParVIAlgorithm.

        Args:
            particle_dim (int): dimension of the particles.
            train_state_spec (nested TensorSpec): for the network state of
                ``train_step()``.
            num_particles (int): number of particles.
            entropy_regularization (float): weight of the repulsive term in par_vi. 
            par_vi (string): par_vi methods, options are [``svgd``, ``gfsf``],
                * svgd: empirical expectation of SVGD is evaluated by reusing
                    the same batch of particles.   
                * gfsf: wasserstein gradient flow with smoothed functions. It 
                    involves a kernel matrix inversion, so computationally more
                    expensive, but in some cases the convergence seems faster 
                    than svgd approaches.
            optimizer (torch.optim.Optimizer): (optional) optimizer for training
            name (str): name of this generator
        """
        super().__init__(
            train_state_spec=train_state_spec,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)
        self._particle_dim = particle_dim
        self._num_particles = num_particles
        self._entropy_regularization = entropy_regularization
        self._particles = None
        self._par_vi = par_vi
        if par_vi == 'gfsf':
            self._grad_func = self._gfsf_grad
        elif par_vi == 'svgd':
            self._grad_func = self._svgd_grad
        elif par_vi == 'minmax':
            self._grad_func = self._minmax_grad
            if critic_input_dim is None:
                critic_input_dim = particle_dim
            self._critic_iter_num = critic_iter_num
            self._critic_l2_weight = critic_l2_weight
            if critic_optimizer is None:
                critic_optimizer = alf.optimizers.Adam(lr=1e-3)
            self._critic = CriticAlgorithm(
                TensorSpec(shape=(critic_input_dim, )),
                hidden_layers=critic_hidden_layers,
                use_bn=critic_use_bn,
                optimizer=critic_optimizer)
        elif par_vi == None:
            self._grad_func = self._ml_grad
        else:
            raise ValueError("Unsupported par_vi method: %s" % par_vi)

        self._kernel_width_averager = AdaptiveAverager(
            tensor_spec=TensorSpec(shape=()))

        self._particles = torch.nn.Parameter(
            torch.randn(num_particles, particle_dim, requires_grad=True))

    @property
    def num_particles(self):
        return self._num_particles

    @property
    def particles(self):
        return self._particles

    def predict_step(self, state=None):
        """Generate outputs given inputs.

        Args:
            state: not used

        Returns:
            AlgorithmStep: outputs with shape (num_particles, output_dim)
        """
        return AlgStep(output=self.particles, state=(), info=())

    def train_step(self,
                   loss_func,
                   transform_func=None,
                   entropy_regularization=None,
                   loss_mask=None,
                   state=None):
        """
        Args:
            outputs (Tensor): generator's output (possibly from previous runs) used
                for this train_step.
            loss_func (Callable): loss_func(loss_inputs) returns a Tensor or 
                namedtuple of tensors with field `loss`, which is a Tensor of
                shape [num_particles] a loss term for optimizing the generator.
            transform_func (Callable): tranform functoin on particles. Used in 
                function value based par_vi, where each particle represents 
                parameters of a neural network function. It is call by
                transform_func(particles) which returns the following,
                - outputs: outputs of network parameterized by particles evaluated
                    on predifined training batch.
                - extra_outputs: outputs of network parameterized by particles
                    evaluated on additional sampled data.
            entropy_regularization (float): weight of the repulsive term in par_vi. 
                If None, use self._entropy_regularization.
            loss_mask (Tensor): mask indicating which samples are valid for loss
                propagation.
            state: not used

        Returns:
            AlgorithmStep:
                outputs: Tensor with shape (num_particles, dim)
                info: LossInfo
        """
        if entropy_regularization is None:
            entropy_regularization = self._entropy_regularization
        loss, loss_propagated = self._grad_func(
            self.particles, loss_func, entropy_regularization, transform_func)
        if loss_mask is not None:
            loss_propagated = loss_propagated * loss_mask

        return AlgStep(
            output=self.particles,
            state=(),
            info=LossInfo(loss=loss_propagated, extra=loss))

    def _kernel_width(self, dist):
        """Update kernel_width averager and get latest kernel_width. """
        if dist.ndim > 1:
            dist = torch.sum(dist, dim=-1)
            assert dist.ndim == 1, "dist must have dimension 1 or 2."
        width, _ = torch.median(dist, dim=0)
        width = width / np.log(len(dist))
        self._kernel_width_averager.update(width)

        return self._kernel_width_averager.get()

   
    def _rbf_func(self, x, y):
        r"""
        Compute the rbf kernel and its gradient w.r.t. first entry 
        :math:`K(x, y), \nabla_x K(x, y)`, used by svgd_grad.

        Args:
            x (Tensor): set of N particles, shape (Nx x W), where W is the 
                dimenseion of each particle
            y (Tensor): set of N particles, shape (Ny x W), where W is the 
                dimenseion of each particle

        Returns:
            :math:`K(x, y)` (Tensor): the RBF kernel of shape (Nx x Ny)
            :math:`\nabla_x K(x, y)` (Tensor): the derivative of RBF kernel of shape (Nx x Ny x D)
            
        """
        Nx, Dx = x.shape
        Ny, Dy = y.shape
        assert Dx == Dy
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [Nx, Ny, W]
        dist_sq = torch.sum(diff**2, -1)  # [Nx, Ny]
        #h = self._kernel_width(dist_sq.view(-1))
        h, _ = torch.median(dist_sq.view(-1), dim=0)
        if h == 0.:
            h = torch.ones_like(h)
        else:
            h = h / max(np.log(Nx), 1.)

        kappa = torch.exp(-dist_sq / h)  # [Nx, Nx]
        kappa_grad = torch.einsum('ij,ijk->ijk', kappa,
                                  -2 * diff / h)  # [Nx, Ny, W]
        return kappa, kappa_grad

    def _score_func(self, x, alpha=1e-5):
        r"""
        Compute the stein estimator of the score function 
        :math:`\nabla\log q = -(K + \alpha I)^{-1}\nabla K`,
        used by gfsf_grad. 

        Args:
            x (Tensor): set of N particles, shape (N x D), where D is the 
                dimenseion of each particle
            alpha (float): weight of regularization for inverse kernel
                this parameter turns out to be crucial for convergence.

        Returns:
            :math:`\nabla\log q` (Tensor): the score function of shape (N x D)
            
        """
        N, D = x.shape
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N, N, D]
        dist_sq = torch.sum(diff**2, -1)  # [N, N]
        h, _ = torch.median(dist_sq.view(-1), dim=0)
        h = h / np.log(N)

        kappa = torch.exp(-dist_sq / h)  # [N, N]
        kappa_inv = torch.inverse(kappa + alpha * torch.eye(N))  # [N, N]
        kappa_grad = torch.einsum('ij,ijk->jk', kappa, -2 * diff / h)  # [N, D]

        return kappa_inv @ kappa_grad
    
    def _ml_grad(self,
                 particles,
                 loss_func,
                 entropy_regularization=None,
                 transform_func=None):
        if transform_func is not None:
            particles, extra_particles, _ = transform_func(particles)
            aug_particles = torch.cat([particles, extra_particles], dim=-1)
        else:
            aug_particles = particles
        loss_inputs = particles
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        grad = torch.autograd.grad(neglogp.sum(), loss_inputs)[0]
        loss_propagated = torch.sum(grad.detach() * particles, dim=-1)

        return loss, loss_propagated


    def _svgd_grad(self,
                   particles,
                   loss_func,
                   entropy_regularization,
                   transform_func=None):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by splitting half of the sampled batch. 
        """
        if transform_func is not None:
            particles, extra_particles = transform_func(particles)
            aug_particles = torch.cat([particles, extra_particles], dim=-1)
        else:
            aug_particles = particles
        loss_inputs = particles
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(),
                                        loss_inputs)[0]  # [N, D]

        # [N, N], [N, N, D]
        kernel_weight, kernel_grad = self._rbf_func(aug_particles.detach(),
                                                    aug_particles.detach())
        kernel_logp = torch.matmul(kernel_weight.t(),
                                   loss_grad) / self.num_particles  # [N, D]

        loss_prop_kernel_logp = torch.sum(
            kernel_logp.detach() * particles, dim=-1)
        loss_prop_kernel_grad = torch.sum(
            -entropy_regularization * kernel_grad.mean(0).detach() *
            aug_particles,
            dim=-1)
        loss_propagated = loss_prop_kernel_logp + loss_prop_kernel_grad

        return loss, loss_propagated

    def _gfsf_grad(self,
                   particles,
                   loss_func,
                   entropy_regularization,
                   transform_func=None):
        """Compute particle gradients via GFSF (Stein estimator). """
        if transform_func is not None:
            particles, extra_particles = transform_func(particles)
            aug_particles = torch.cat([particles, extra_particles], dim=-1)
        else:
            aug_particles = particles
        score_inputs = aug_particles.detach()
        loss_inputs = particles
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), particles)[0]  # [N, D]
        logq_grad = self._score_func(score_inputs) * entropy_regularization

        loss_prop_neglogp = torch.sum(loss_grad.detach() * particles, dim=-1)
        loss_prop_logq = torch.sum(-logq_grad.detach() * aug_particles, dim=-1)
        loss_propagated = loss_prop_neglogp + loss_prop_logq

        return loss, loss_propagated

    def _jacobian_trace(self, fx, x):
        """Hutchinson's trace Jacobian estimator O(1) call to autograd,
            used by "\"minmax\" method"""
        assert fx.shape[-1] == x.shape[-1], (
            "Jacobian is not square, no trace defined.")
        eps = torch.randn_like(fx)
        jvp = torch.autograd.grad(
            fx, x, grad_outputs=eps, retain_graph=True, create_graph=True)[0]
        tr_jvp = torch.einsum('bi,bi->b', jvp, eps)
        return tr_jvp

    def _critic_train_step(self, inputs, loss_func, entropy_regularization=1.):
        """
        Compute the loss for critic training.
        """
        loss = loss_func(inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), inputs)[0]  # [N, D]
        outputs = self._critic.predict_step(inputs).output
        tr_gradf = self._jacobian_trace(outputs, inputs)  # [N]

        f_loss_grad = (loss_grad.detach() * outputs).sum(1)  # [N]
        loss_stein = f_loss_grad - entropy_regularization * tr_gradf  # [N]

        l2_penalty = (outputs * outputs).sum(1).mean() * self._critic_l2_weight
        critic_loss = loss_stein.mean() + l2_penalty

        return critic_loss


    def _minmax_grad(self,
                     particles,
                     loss_func,
                     entropy_regularization,
                     transform_func=None):
        """
        Compute particle gradients via minmax svgd (Fisher Neural Sampler). 
        """
        if transform_func is not None:
            aug_particles, extra_particles = transform_func(particles)
            #aug_particles = torch.cat([particles, extra_particles], dim=-1)
        else:
            aug_particles = particles

        for i in range(self._critic_iter_num):
            critic_inputs = aug_particles.detach().clone()
            critic_inputs.requires_grad = True

            critic_loss = self._critic_train_step(critic_inputs, loss_func,
                                                  entropy_regularization)
            self._critic.update_with_gradient(LossInfo(loss=critic_loss))
        
        loss_inputs = aug_particles
        # compute amortized svgd
        loss = loss_func(loss_inputs.detach())
        critic_outputs = self._critic.predict_step(aug_particles.detach()).output
        loss_propagated = torch.sum(-critic_outputs.detach() * aug_particles, dim=-1)

        return loss, loss_propagated


