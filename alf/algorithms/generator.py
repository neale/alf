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
"""A generic generator."""

import gin
import numpy as np
import torch

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.mi_estimator import MIEstimator
from alf.data_structures import AlgStep, LossInfo, namedtuple
import alf.nest as nest
from alf.networks import Network, EncodingNetwork, PinverseNetwork
from alf.networks.relu_mlp import ReluMLP
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops
from alf.utils.averager import AdaptiveAverager

import torch.nn.utils.spectral_norm as sn

from scipy.sparse.linalg import bicg, bicgstab

GeneratorLossInfo = namedtuple("GeneratorLossInfo",
                               ["generator", "mi_estimator", "pinverse"])


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
                 use_relu_mlp=False,
                 use_bn=True,
                 force_fullrank=False,
                 fullrank_diag_weight=1.0,
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
            use_relu_mlp (bool): whether use ReluMLP as default net constrctor.
                Diagonals of Jacobian can be explicitly computed for ReluMLP.
            use_bn (bool): whether use batch norm for each critic layers.
            force_fullrank (bool): whether force the network function to be
                of fullrank, if True, achieved by adding a direct link from
                input to output.
            fullrank_diag_weight (float): the weight parameter multiplied to
                the direct input-output link when force_fullrank.
            optimizer (torch.optim.Optimizer): (optional) optimizer for training.
            name (str): name of this CriticAlgorithm.
        """
        if optimizer is None:
            optimizer = alf.optimizers.Adam(lr=1e-3)
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)

        self._use_relu_mlp = use_relu_mlp
        self._force_fullrank = force_fullrank
        self._fullrank_diag_weight = fullrank_diag_weight
        self._input_dim = input_tensor_spec.shape[0]
        self._output_dim = output_dim
        if output_dim is None:
            self._output_dim = input_tensor_spec.shape[0]
        if net is None:
            if use_relu_mlp:
                net = ReluMLP(
                    input_tensor_spec=input_tensor_spec,
                    output_size=self._output_dim,
                    hidden_layers=hidden_layers,
                    activation=activation)
            else:
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

    def predict_step(self,
                     inputs,
                     vector=None,
                     state=None,
                     requires_jac_diag=False,
                     requires_jac=False):
        """Predict for one step of inputs.
        Args:
            inputs (torch.Tensor): inputs for prediction.
            vector (torch.Tensor): used for vector-jacobian product, default 
                value is None, i.e., no vjp computation.
            state: not used.
            requires_jac_trace (bool): whether outputs diagonals of Jacobian.
        Returns:
            AlgStep:
            - output (torch.Tensor): predictions or (predictions, diag_jacobian)
                if requires_jac_diag is True.
            - state: not used.
        """
        if self._use_relu_mlp:
            if vector is not None:
                input_dim = inputs.shape[-1]
                if self._force_fullrank:
                    assert input_dim == self._output_dim
                    vjp, outputs = self._net.compute_vjp(
                        inputs[:, :self._input_dim], vector)
                    outputs += self._fullrank_diag_weight * inputs
                    vjp = torch.cat(
                        (vjp,
                         torch.zeros(vjp.shape[0],
                                     self._output_dim - self._input_dim)),
                        dim=-1)
                    vjp += self._fullrank_diag_weight * inputs  # [N2*N, D]
                else:
                    assert input_dim == self._input_dim
                    vjp, outputs = self._net.compute_vjp(inputs, vector)
                outputs = (outputs, vjp)
            else:
                outputs = self._net(
                    inputs,
                    requires_jac_diag=requires_jac_diag,
                    requires_jac=requires_jac)[0]
        else:
            outputs = self._net(inputs)[0]

        return AlgStep(output=outputs, state=(), info=())


@gin.configurable
class PinverseAlgorithm(Algorithm):
    r"""PinverseNet Algorithm
        
    Simple MLP network used with the amortized functional gradient vi methods.
    It is used to predict :math:`x=J^{-1}*eps` given eps or :math:`xJ^{-1}e`
    for a standard Gaussian noise e if eps is not given.
    
    If using ``rkhs``, then the eps quantity represents the kernel grad
        :math:`\nabla_{z'}k(z', z)`
    if using ``minmax``, the eps is not given
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 eps_dim=None,
                 hidden_size=500,
                 net: Network = None,
                 optimizer=None,
                 name="PinverseAlgorithm"):
        r"""Create a PinverseAlgorithm.
        Args:
            input_dim (int): dimension of input noise vector z, if equals to 
                output_dim, use fullsize of z, otherwise, input the first 
                input_dim dim of z, and add a direct link of z to the output.
            output_dim (int): total output dimension of the pinverse net, will differ
                between ``svgd3`` and ``minmax`` methods
            eps_dim (int): dimension of the kernel_grad term when Pinverse is used
                for ``rkhs``, None when used for ``minmax``.
            hidden_size (tuple(int,)): base hidden width for pinverse net
            net (Network): network for predicting outputs from inputs.
                If None, a default ReluMLP with hidden_layers will be created
            optimizer (torch.optim.Optimizer): (optional) optimizer for training.
            name (str): name of this CriticAlgorithm.
        """
        if optimizer is None:
            optimizer = alf.optimizers.Adam(lr=1e-3)
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)

        self._output_dim = output_dim
        if net is None:
            z_spec = TensorSpec(shape=(input_dim, ))
            if eps_dim is None:
                input_tensor_spec = z_spec
            else:
                eps_spec = TensorSpec(shape=(eps_dim, ))
                input_tensor_spec = (z_spec, eps_spec)
            net = PinverseNetwork(
                input_tensor_spec,
                output_dim,
                hidden_size,
                name='PinverseNetwork')
        self._net = net

    def predict_step(self, inputs, state=None):
        """Predict for one step of inputs.

        Args:
            inputs (tuple of Tensors): inputs (z, eps) for prediction.
            state: not used.
            
        Returns:
            AlgStep:
            - output (torch.Tensor): predictions
                if requires_jac_diag is True.
            - state: not used.
        """
        outputs = self._net(inputs)[0]

        return AlgStep(output=outputs, state=(), info=())


@gin.configurable
class Generator(Algorithm):
    r"""Generator

    Generator generates outputs given `inputs` (can be None) by transforming
    a random noise and input using `net`:

        outputs = net([noise, input]) if input is not None
                  else net(noise)

    The generator is trained to minimize the following objective:

        :math:`E(loss\_func(net([noise, input]))) - entropy\_regulariztion \cdot H(P)`

    where P is the (conditional) distribution of outputs given the inputs
    implied by `net` and H(P) is the (conditional) entropy of P.

    If the loss is the (unnormalized) negative log probability of some
    distribution Q and the ``entropy_regularization`` is 1, this objective is
    equivalent to minimizing :math:`KL(P||Q)`.

    It uses two different ways to optimize `net` depending on
    ``entropy_regularization``:

    * ``entropy_regularization`` = 0: the minimization is achieved by simply
      minimizing loss_func(net([noise, inputs]))

    * entropy_regularization > 0: the minimization is achieved using amortized
      particle-based variational inference (ParVI), in particular, three ParVI
      methods are implemented:

        1. amortized Stein Variational Gradient Descent (SVGD):

        Feng et al "Learning to Draw Samples with Amortized Stein Variational
        Gradient Descent" https://arxiv.org/pdf/1707.06626.pdf

        2. amortized Wasserstein ParVI with Smooth Functions (GFSF):

        Liu, Chang, et al. "Understanding and accelerating particle-based 
        variational inference." International Conference on Machine Learning. 2019.

        3. amortized Fisher Neural Sampler with Hutchinson's estimator (MINMAX):

        Hu et at. "Stein Neural Sampler." https://arxiv.org/abs/1810.03545, 2018.

    It also supports an additional optional objective of maximizing the mutual
    information between [noise, inputs] and outputs by using mi_estimator to
    prevent mode collapse. This might be useful for ``entropy_regulariztion`` = 0
    as suggested in section 5.1 of the following paper:

    Hjelm et al `Learning Deep Representations by Mutual Information Estimation
    and Maximization <https://arxiv.org/pdf/1808.06670.pdf>`_
    """

    def __init__(self,
                 output_dim,
                 noise_dim=32,
                 input_tensor_spec=None,
                 hidden_layers=(256, ),
                 net: Network = None,
                 net_moving_average_rate=None,
                 entropy_regularization=0.,
                 mi_weight=None,
                 mi_estimator_cls=MIEstimator,
                 par_vi=None,
                 functional_gradient=None,
                 force_fullrank=True,
                 fullrank_diag_weight=1.,
                 pinverse_solve_iters=1,
                 use_jac_regularization=True,
                 critic_input_dim=None,
                 critic_hidden_layers=(100, 100),
                 critic_l2_weight=10.,
                 critic_iter_num=2,
                 critic_relu_mlp=False,
                 critic_use_bn=True,
                 minmax_resample=True,
                 critic_optimizer=None,
                 pinverse_optimizer=None,
                 optimizer=None,
                 name="Generator"):
        r"""Create a Generator.

        Args:
            output_dim (int): dimension of output
            noise_dim (int): dimension of noise
            input_tensor_spec (nested TensorSpec): spec of inputs. If there is
                no inputs, this should be None.
            hidden_layers (tuple): sizes of hidden layers.
            net (Network): network for generating outputs from [noise, inputs]
                or noise (if inputs is None). If None, a default one with
                hidden_layers will be created
            net_moving_average_rate (float): If provided, use a moving average
                version of net to do prediction. This has been shown to be
                effective for GAN training (arXiv:1907.02544, arXiv:1812.04948).
            entropy_regularization (float): weight of entropy regularization.
            mi_weight (float): weight of mutual information loss.
            mi_estimator_cls (type): the class of mutual information estimator
                for maximizing the mutual information between [noise, inputs]
                and [outputs, inputs].
            par_vi (string): ParVI methods, options are
                [``svgd``, ``svgd2``, ``svgd3``, ``gfsf``],
                * svgd: empirical expectation of SVGD is evaluated by a single 
                    resampled particle. The main benefit of this choice is it 
                    supports conditional case, while all other options do not.
                * svgd2: empirical expectation of SVGD is evaluated by splitting
                    half of the sampled batch. It is a trade-off between 
                    computational efficiency and convergence speed.
                * svgd3: empirical expectation of SVGD is evaluated by 
                    resampled particles of the same batch size. It has better
                    convergence but involves resampling, so less efficient
                    computaionally comparing with svgd2.
                * gfsf: wasserstein gradient flow with smoothed functions. It 
                    involves a kernel matrix inversion, so computationally most
                    expensive, but in some case the convergence seems faster 
                    than svgd approaches.
                * minmax: Fisher Neural Sampler, optimal descent direction of
                    the Stein discrepancy is solved by an inner optimization
                    procedure in the space of L2 neural networks.
            functional_gradient (string): amortized functional gradient vi
                methods, options are [``rkhs``, ``minmax``]
                * rkhs: the function space where gradient is computed is the 
                    RKHS of Gaussian kernel.
                * minmax: the function space where gradient is computed is L2.
            force_fullrank (bool): whether force the generator function to be
                of fullrank, if True, achieved by adding a direct link from
                input to output.
            fullrank_diag_weight (float): the weight parameter multiplied to
                the direct input-output link when force_fullrank.
            optimizer (torch.optim.Optimizer): (optional) optimizer for training
            name (str): name of this generator
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)
        assert noise_dim <= output_dim, (
            "noise_dim is higher than the output_dim of the generator!")
        self._output_dim = output_dim
        self._noise_dim = noise_dim
        self._entropy_regularization = entropy_regularization
        self._par_vi = par_vi
        self._functional_gradient = functional_gradient

        if entropy_regularization == 0:
            self._grad_func = self._ml_grad
        else:
            if par_vi == 'gfsf':
                self._grad_func = self._gfsf_grad
            elif par_vi == 'svgd':
                self._grad_func = self._svgd_grad
            elif par_vi == 'svgd2':
                self._grad_func = self._svgd_grad2
            elif par_vi == 'svgd3':
                self._grad_func = self._svgd_grad3
            elif par_vi == 'minmax':
                if critic_input_dim is None:
                    critic_input_dim = output_dim
                self._grad_func = self._minmax_grad
                self._critic_iter_num = critic_iter_num
                self._critic_l2_weight = critic_l2_weight
                self._critic_relu_mlp = critic_relu_mlp
                self._minmax_resample = minmax_resample
                if critic_optimizer is None:
                    critic_optimizer = alf.optimizers.Adam(lr=1e-3)
                self._critic = CriticAlgorithm(
                    TensorSpec(shape=(critic_input_dim, )),
                    hidden_layers=critic_hidden_layers,
                    use_relu_mlp=critic_relu_mlp,
                    use_bn=critic_use_bn,
                    optimizer=critic_optimizer)
            else:
                raise ValueError("Unsupported par_vi method: %s" % par_vi)

        if functional_gradient is not None:
            if functional_gradient == 'rkhs':
                self._grad_func = self._rkhs_func_grad
                if force_fullrank:
                    self._eps_dim = output_dim
                    hidden_size = 100
                else:
                    self._eps_dim = noise_dim
                    hidden_size = 10
            elif functional_gradient == 'minmax':
                assert force_fullrank is True, (
                    "force fullrank when using minmax functional gradient!")
                self._grad_func = self._minmax_func_grad
                self._eps_dim = None
                hidden_size = 100
                self._critic_iter_num = critic_iter_num
                self._critic_l2_weight = critic_l2_weight
                self._minmax_resample = minmax_resample
                if critic_optimizer is None:
                    critic_optimizer = alf.optimizers.Adam(lr=1e-3)
                self._critic = CriticAlgorithm(
                    TensorSpec(shape=(noise_dim, )),
                    output_dim=output_dim,
                    hidden_layers=(hidden_size, hidden_size),
                    use_relu_mlp=True,
                    use_bn=critic_use_bn,
                    force_fullrank=force_fullrank,
                    optimizer=critic_optimizer)
            else:
                raise ValueError(
                    'functional gradient only supports ``rkhs`` and ``minmax``'
                )

            if noise_dim == output_dim:
                force_fullrank = False
            self._force_fullrank = force_fullrank
            self._fullrank_diag_weight = fullrank_diag_weight
            self._pinverse_solve_iters = pinverse_solve_iters
            self._use_jac_regularization = use_jac_regularization

            if pinverse_optimizer is None:
                pinverse_optimizer = alf.optimizers.Adam(
                    lr=1e-4, weight_decay=1e-5)

            self._pinverse = PinverseAlgorithm(
                noise_dim,
                output_dim,
                eps_dim=self._eps_dim,
                hidden_size=hidden_size,
                optimizer=pinverse_optimizer)

        self._kernel_width_averager = AdaptiveAverager(
            tensor_spec=TensorSpec(shape=()))

        noise_spec = TensorSpec(shape=(noise_dim, ))

        if net is None:
            net_input_spec = noise_spec
            if input_tensor_spec is not None:
                net_input_spec = [net_input_spec, input_tensor_spec]
            net = EncodingNetwork(
                input_tensor_spec=net_input_spec,
                fc_layer_params=hidden_layers,
                last_layer_size=output_dim,
                last_activation=math_ops.identity,
                name="Generator")

        self._mi_estimator = None
        self._input_tensor_spec = input_tensor_spec
        if mi_weight is not None:
            x_spec = noise_spec
            y_spec = TensorSpec((output_dim, ))
            if input_tensor_spec is not None:
                x_spec = [x_spec, input_tensor_spec]
            self._mi_estimator = mi_estimator_cls(
                x_spec, y_spec, sampler='shift')
            self._mi_weight = mi_weight
        self._net = net

        self._predict_net = None
        self._net_moving_average_rate = net_moving_average_rate
        if net_moving_average_rate:
            self._predict_net = net.copy(name="Generator_average")
            self._predict_net_updater = common.get_target_updater(
                self._net, self._predict_net, tau=net_moving_average_rate)

    def _trainable_attributes_to_ignore(self):
        return ["_predict_net"]

    @property
    def noise_dim(self):
        return self._noise_dim

    def _predict(self,
                 inputs=None,
                 noise=None,
                 batch_size=None,
                 training=True,
                 use_jac_regularization=None):

        if inputs is None:
            assert self._input_tensor_spec is None
            if noise is None:
                assert batch_size is not None
                noise = torch.randn(batch_size, self._noise_dim)
            gen_inputs = noise
        else:
            nest.assert_same_structure(inputs, self._input_tensor_spec)
            batch_size = nest.get_nest_batch_size(inputs)
            if noise is None:
                noise = torch.randn(batch_size, self._noise_dim)
            else:
                assert noise.shape[0] == batch_size
                assert noise.shape[1] == self._noise_dim
            gen_inputs = [noise, inputs]
        if self._predict_net and not training:
            outputs = self._predict_net(gen_inputs)[0]
        else:
            if self._functional_gradient is not None:
                if use_jac_regularization is None:
                    use_jac_regularization = self._use_jac_regularization
                outputs = self._net(
                    gen_inputs, requires_jac=use_jac_regularization)[0]
                if use_jac_regularization and self._force_fullrank:
                    outputs, jac = outputs
                    extra_noise = torch.randn(
                        noise.shape[0], self._output_dim - self._noise_dim)
                    gen_inputs = torch.cat((gen_inputs, extra_noise), dim=-1)
                    outputs += self._fullrank_diag_weight * gen_inputs
                    zero_vec = torch.zeros(noise.shape[0], self._output_dim,
                                           self._output_dim - self._noise_dim)
                    jac = torch.cat((jac, zero_vec), dim=-1)
                    jac += self._fullrank_diag_weight * torch.eye(
                        self._output_dim)
                    outputs = (outputs, jac)
                elif self._force_fullrank:
                    extra_noise = torch.randn(
                        noise.shape[0], self._output_dim - self._noise_dim)
                    gen_inputs = torch.cat((gen_inputs, extra_noise), dim=-1)
                    outputs += self._fullrank_diag_weight * gen_inputs
            else:
                outputs = self._net(gen_inputs)[0]

        return outputs, gen_inputs

    def predict_step(self,
                     inputs=None,
                     noise=None,
                     batch_size=None,
                     training=False,
                     state=None):
        """Generate outputs given inputs.

        Args:
            inputs (nested Tensor): if None, the outputs is generated only from
                noise.
            noise (Tensor): input to the generator.
            batch_size (int): batch_size. Must be provided if inputs is None.
                Its is ignored if inputs is not None
            training (bool): whether train the generator.
            state: not used

        Returns:
            AlgorithmStep: outputs with shape (batch_size, output_dim)
        """
        outputs, _ = self._predict(
            inputs=inputs,
            noise=noise,
            batch_size=batch_size,
            training=training)
        return AlgStep(output=outputs, state=(), info=())

    def train_step(self,
                   inputs,
                   loss_func,
                   batch_size=None,
                   transform_func=None,
                   entropy_regularization=None,
                   state=None):
        """
        Args:
            inputs (nested Tensor): if None, the outputs is generated only from
                noise.
            outputs (Tensor): generator's output (possibly from previous runs) used
                for this train_step.
            loss_func (Callable): loss_func([outputs, inputs])
                (loss_func(outputs) if inputs is None) returns a Tensor or namedtuple
                of tensors with field `loss`, which is a Tensor of
                shape [batch_size] a loss term for optimizing the generator.
            batch_size (int): batch_size. Must be provided if inputs is None.
                Its is ignored if inputs is not None.
            transform_func (Callable): transform_func(outputs) transforms the 
                generator outputs to its corresponding function values evaluated
                on current training batch.
            entropy_regularization (float): weight of entropy regularization.
            state: not used

        Returns:
            AlgorithmStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
        outputs, gen_inputs = self._predict(inputs, batch_size=batch_size)
        if self._functional_gradient:
            outputs = (outputs, gen_inputs)
        if entropy_regularization is None:
            entropy_regularization = self._entropy_regularization
        loss, loss_propagated = self._grad_func(
            inputs, outputs, loss_func, entropy_regularization, transform_func)
        mi_loss = ()
        if self._mi_estimator is not None:
            mi_step = self._mi_estimator.train_step([gen_inputs, outputs])
            mi_loss = mi_step.info.loss
            loss_propagated = loss_propagated + self._mi_weight * mi_loss

        if self._use_pinverse:
            loss, pinverse_loss = loss
        else:
            pinverse_loss = None

        return AlgStep(
            output=outputs,
            state=(),
            info=LossInfo(
                loss=loss_propagated,
                extra=GeneratorLossInfo(
                    generator=loss,
                    mi_estimator=mi_loss,
                    pinverse=pinverse_loss)))

    def _ml_grad(self,
                 inputs,
                 outputs,
                 loss_func,
                 entropy_regularization=None,
                 transform_func=None):
        if transform_func is not None:
            outputs = transform_func(outputs)
        loss_inputs = outputs if inputs is None else [outputs, inputs]
        loss = loss_func(loss_inputs)

        grad = torch.autograd.grad(loss.sum(), outputs)[0]
        loss_propagated = torch.sum(grad.detach() * outputs, dim=-1)

        return loss, loss_propagated

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
        """Compute RGF kernel, used by svgd_grad. """
        d = (x - y)**2
        d = torch.sum(d, -1)
        h = self._kernel_width(d)
        w = torch.exp(-d / h)

        return w

    def _rbf_func2(self, x, y):
        r"""
        Compute the rbf kernel and its gradient w.r.t. first entry 
        :math:`K(x, y), \nabla_x K(x, y)`, used by svgd_grad2 and svgd_grad3. 

        Args:
            x (Tensor): set of N particles, shape (Nx, ...), where Nx is the 
                number of particles.
            y (Tensor): set of N particles, shape (Ny, ...), where Ny is the 
                number of particles.

        Returns:
            :math:`K(x, y)` (Tensor): the RBF kernel of shape (Nx x Ny)
            :math:`\nabla_x K(x, y)` (Tensor): the derivative of RBF kernel of shape (Nx x Ny x D)

        """
        Nx = x.shape[0]
        Ny = y.shape[0]
        x = x.view(Nx, -1)
        y = y.view(Ny, -1)
        Dx = x.shape[1]
        Dy = y.shape[1]
        assert Dx == Dy
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [Nx, Ny, D]
        dist_sq = torch.sum(diff**2, -1)  # [Nx, Ny]
        h = self._kernel_width(dist_sq.view(-1))

        kappa = torch.exp(-dist_sq / h)  # [Nx, Nx]
        kappa_grad = kappa.unsqueeze(-1) * (-2 * diff / h)  # [Nx, Ny, D]
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

        return -kappa_inv @ kappa_grad

    def _svgd_grad(self,
                   inputs,
                   outputs,
                   loss_func,
                   entropy_regularization,
                   transform_func=None):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by a single resampled particle. 
        """
        outputs2, _ = self._predict(inputs, batch_size=outputs.shape[0])
        assert transform_func is None, (
            "function value based vi is not supported for svgd_grad")

        kernel_weight = self._rbf_func(outputs, outputs2)
        weight_sum = entropy_regularization * kernel_weight.sum()

        kernel_grad = torch.autograd.grad(weight_sum, outputs2)[0]

        loss_inputs = outputs2 if inputs is None else [outputs2, inputs]
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        weighted_loss = kernel_weight.detach() * neglogp

        loss_grad = torch.autograd.grad(weighted_loss.sum(), outputs2)[0]
        grad = loss_grad - kernel_grad
        loss_propagated = torch.sum(grad.detach() * outputs, dim=-1)

        return loss, loss_propagated

    def _svgd_grad2(self,
                    inputs,
                    outputs,
                    loss_func,
                    entropy_regularization,
                    transform_func=None):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by splitting half of the sampled batch. 
        """
        assert inputs is None, '"svgd2" does not support conditional generator'
        if transform_func is not None:
            outputs, extra_outputs, _ = transform_func(outputs)
            aug_outputs = torch.cat([outputs, extra_outputs], dim=1)
        else:
            aug_outputs = outputs
        num_particles = outputs.shape[0] // 2
        outputs_i, outputs_j = torch.split(outputs, num_particles, dim=0)
        aug_outputs_i, aug_outputs_j = torch.split(
            aug_outputs, num_particles, dim=0)

        loss_inputs = outputs_j
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(),
                                        loss_inputs)[0]  # [Nj, D]

        # [Nj, Ni], [Nj, Ni, D']
        kernel_weight, kernel_grad = self._rbf_func2(aug_outputs_j.detach(),
                                                     aug_outputs_i.detach())
        kernel_logp = torch.matmul(kernel_weight.t(),
                                   loss_grad) / num_particles  # [Ni, D]

        loss_prop_kernel_logp = torch.sum(
            kernel_logp.detach() * outputs_i, dim=-1)
        loss_prop_kernel_grad = torch.sum(
            -entropy_regularization * kernel_grad.mean(0).detach() *
            aug_outputs_i,
            dim=-1)

        loss_propagated = loss_prop_kernel_logp + loss_prop_kernel_grad

        return loss, loss_propagated

    def _svgd_grad3(self,
                    inputs,
                    outputs,
                    loss_func,
                    entropy_regularization,
                    transform_func=None):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by resampled particles of the same batch size. 
        """
        assert inputs is None, '"svgd3" does not support conditional generator'
        num_particles = outputs.shape[0]
        outputs2, _ = self._predict(inputs, batch_size=num_particles)
        if transform_func is not None:
            outputs, extra_outputs, samples = transform_func(outputs)
            outputs2, extra_outputs2, _ = transform_func((outputs2, samples))
            aug_outputs = torch.cat([outputs, extra_outputs], dim=-1)
            aug_outputs2 = torch.cat([outputs2, extra_outputs2], dim=-1)
        else:
            aug_outputs = outputs  # [N, D']
            aug_outputs2 = outputs2  # [N2, D']
        loss_inputs = outputs2
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss

        loss_grad = torch.autograd.grad(neglogp.sum(),
                                        loss_inputs)[0]  # [N2, D]
        # [N2, N], [N2, N, D]
        kernel_weight, kernel_grad = self._rbf_func2(aug_outputs2.detach(),
                                                     aug_outputs.detach())
        kernel_logp = torch.matmul(kernel_weight.t(),
                                   loss_grad) / num_particles  # [N, D]
        loss_prop_kernel_logp = torch.sum(
            kernel_logp.detach() * outputs, dim=-1)
        loss_prop_kernel_grad = torch.sum(
            -entropy_regularization * kernel_grad.mean(0).detach() *
            aug_outputs,
            dim=-1)
        loss_propagated = loss_prop_kernel_logp + loss_prop_kernel_grad

        return loss, loss_propagated

    def _pinverse_train_step(self, z, eps=None):
        r"""Compute the loss for pinverse training. 

        self._pinverse solves an inverse problem for the amortized 
        functional gradient vi methods ``rkhs`` and ``minmax``. 
        * rkhs: takes z' and :math:`\nabla_{z'}k(z', z)` as inputs, outputs
            :math:`(\partial f / \partial z')^{-1} * \nabla_{z'}k(z', z)`.
            The training loss is given by
            :math:`\|(\partical f / \partial z')^T y - \nabla_{z'}k(z',z)`,
            where y denotes the output of self._pinverse and f denotes the 
            generator function, the first term is computed by vjp of f and y.    
        * minmax: takes z as inputs, outputs
            :math:`(\partial f / \partial z)^{-1} * e`.
            The training loss is given by
            :math:`\|(\partical f / \partial z)^T y - e`,
            where y denotes the output of self._pinverse and f denotes the
            generator function, e is a random gaussian vector which we use
            the augmented noise of z. The first term is computed by jvp.

        Args: 
            z (torch.tensor): of size [N2, K] representing z' for ``rkhs``,
                or size [N, K] representing z for ``minmax``.
            eps (torch.tensor): only used for ``rkhs``, size [N2, N, D or K] 
                representing :math:`\nabla_{z'}k(z', z)`, where the last 
                dimension is D when self._force_fullrank is True, K otherwise. 
                In general, K is much less than D.

        Returns:
            pinverse_loss (float)

        """
        assert z.ndim == 2
        assert z.shape[-1] == self._noise_dim or z.shape[-1] == self._output_dim
        z_inputs = z[:, :self._noise_dim]

        if self._functional_gradient == 'rkhs':
            assert eps.ndim == 3
            assert z.shape[0] == eps.shape[0]
            assert eps.shape[-1] == self._eps_dim
            z_inputs = torch.repeat_interleave(
                z_inputs, eps.shape[1], dim=0)  # [N2*N, K]
            eps_inputs = eps.reshape(eps.shape[0] * eps.shape[1],
                                     -1)  # [N2*N, D or K]
            y = self._pinverse.predict_step((z_inputs, eps_inputs)).output
            # [N2*N, D]

            jac_y, _ = self._net.compute_vjp(z_inputs, y)  # [N2*N, K]
            if self._force_fullrank:
                jac_y = torch.cat(
                    (jac_y,
                     torch.zeros(jac_y.shape[0],
                                 self._output_dim - self._noise_dim)),
                    dim=-1)
                jac_y += self._fullrank_diag_weight * y  # [N2*N, D]
            jac_y = jac_y.reshape(z.shape[0], eps.shape[1],
                                  -1)  # [N2, N, D or K]
            loss = torch.nn.functional.mse_loss(jac_y, eps)

        elif self._functional_gradient == 'minmax':
            assert eps is None
            assert z.shape[-1] == self._output_dim
            y = self._pinverse.predict_step(z_inputs).output  # [N2, D]
            jac_y, _ = self._net.compute_vjp(z_inputs, y)  # [N2, K]
            if self._force_fullrank:
                jac_y = torch.cat(
                    (jac_y,
                     torch.zeros(jac_y.shape[0],
                                 self._output_dim - self._noise_dim)),
                    dim=-1)
                jac_y += self._fullrank_diag_weight * y  # [N2, D]
            loss = torch.nn.functional.mse_loss(jac_y, z.detach())

        else:
            raise ValueError('pinverse only supports ``rkhs`` and ``minmax``')

        return loss

    def _rkhs_func_grad(self,
                        inputs,
                        outputs,
                        loss_func,
                        entropy_regularization,
                        transform_func=None):
        """
        Compute the amortized functional gradient of generator, functional gradient
        represented in an RKHS. Empirical expectation evaluated by a resampling
        from the z space of the same batch size. 

        Args:
            inputs: None
            outputs (tuple of Tensors): (outputs, gen_inputs) of size [N, D] and
                [N, K] respectively, where N being the sample size, D being the 
                output dim of ReluMLP and K being the input dim of the generator.
            loss_func (callable)
            entropy_regularization (float): tradeoff parameter
            transform_func (not used)
        """
        assert inputs is None, (
            '``rkhs`` does not support conditional generator')
        outputs, gen_inputs = outputs  # [N, D], [N, D, K]
        if self._use_jac_regularization:
            outputs, jac = outputs
        num_particles = outputs.shape[0]
        outputs2, gen_inputs2 = self._predict(
            batch_size=num_particles)  # [N2, D]

        # [N2, N], [N2, N, D]
        kernel_weight, kernel_grad = self._rbf_func2(gen_inputs2, gen_inputs)

        # train pinverse
        for _ in range(self._pinverse_solve_iters):
            pinverse_loss = self._pinverse_train_step(gen_inputs2.detach(),
                                                      kernel_grad.detach())
            self._pinverse.update_with_gradient(LossInfo(loss=pinverse_loss))

        # construct functional gradient via pinverse
        gen_inputs2_batch = torch.repeat_interleave(
            gen_inputs2, num_particles, dim=0).detach()
        kernel_grad_batch = kernel_grad.reshape(num_particles * num_particles,
                                                -1).detach()
        J_inv_kernel_grad = self._pinverse.predict_step(
            gen_inputs2_batch, kernel_grad_batch).output  # [N2*N, D]
        J_inv_kernel_grad = J_inv_kernel_grad.reshape(
            num_particles, num_particles, -1)  # [N2, N, D]

        loss = loss_func(outputs2)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), outputs2)[0]  # [N2, D]
        kernel_logp = torch.matmul(kernel_weight.t(),
                                   loss_grad) / num_particles  # [N, D]

        grad = kernel_logp - entropy_regularization * J_inv_kernel_grad.mean(0)
        loss_propagated = torch.sum(grad.detach() * outputs, dim=1)

        if self._use_jac_regularization:
            jac_reg = .1 * jac.norm(keepdim=True).mean()
            loss_propagated -= jac_reg

        return (loss, pinverse_loss), loss_propagated

    def _func_critic_train_step(self, inputs, gen_outputs, loss_func):
        r"""Compute the following loss for critic training.
        :math:`\nabla_x neglogp(f(z))\phi(z) + Tr(J_f^{-1}(z)J_{\phi}(z)) - L2(\phi)`
        where the trace term is estimated by the Hutchinson's estimator. 

        Args:
            inputs (torch.Tensor): the (augmented) input noise of size [N, D]
                if force_fullrank or [N, K] if not.
            gen_outputs (torch.Tensor): outputs of the generator corresponding
                to the inputs, of size [N, D].
            loss_func (callable)
        """
        assert inputs.shape[-1] == self._critic._output_dim

        loss = loss_func(gen_outputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(),
                                        gen_outputs)[0]  # [N, D]

        critic_inputs = inputs[:, :self._critic._input_dim]
        J_inv_z = self._pinverse.predict_step(critic_inputs)
        critic_outputs, vjp_critic = self._critic.predict_step(
            critic_inputs, J_inv_z)  # [N, D], [N, D]

        # we use the augmented noise (inputs) as the random noise for the
        # Hutchinson's estimator
        tr_critic_J = torch.einsum('bi,bi->b', vjp_critic, inputs)  # [N]

        critic_loss_grad = (loss_grad * critic_outputs).sum(-1)  # [N]

        l2_penalty = (critic_outputs * critic_outputs).sum(-1).mean()
        l2_penalty = l2_penalty * self._critic_l2_weight

        critic_loss = critic_loss_grad - tr_critic_J + l2_penalty

        return critic_loss

    def _minmax_func_grad(self,
                          inputs,
                          outputs,
                          loss_func,
                          entropy_regularization,
                          transform_func=None):
        """
        Compute the amortized functional gradient of generator, functional gradient
        represented in an L2, which involves a minmax optimization procedure similar
        as the Fisher Neural Sampler. 

        Args:
            inputs: None
            outputs (tuple of Tensors): (outputs, gen_inputs) of size [N, D] and
                [N, D or K] respectively, where N being the sample size, D being the 
                output dim of ReluMLP and K being the input dim of the generator.
                The last dimension of ``gen_inputs`` being D if force_fullrank, 
                being K otherwise.
            loss_func (callable)
            entropy_regularization (float): tradeoff parameter
            transform_func (not used)
        """
        assert inputs is None, (
            '``rkhs`` does not support conditional generator')
        outputs, gen_inputs = outputs  # [N, D], [N, D or K]
        if self._use_jac_regularization:
            outputs, jac = outputs
        num_particles = outputs.shape[0]

        # train pinverse
        for _ in range(self._pinverse_solve_iters):
            # p_inputs = torch.randn_like(gen_inputs)
            p_inputs = outputs.detach().clone()
            p_inputs.requires_grad = True
            pinverse_loss = self._pinverse_train_step(p_inputs)
            self._pinverse.update_with_gradient(LossInfo(loss=pinverse_loss))

        # optimize critic
        for i in range(self._critic_iter_num):
            if self._minmax_resample:
                f_outputs, z_inputs = self._predict(
                    batch_size=num_particles, use_jac_regularization=False)
            else:
                z_inputs = gen_inputs.detach().clone()
                z_inputs.requires_grad = True
                f_outputs = outputs.detach().clone()
                f_outputs.requires_grad = True
            critic_loss = self._func_critic_train_step(z_inputs, f_outputs,
                                                       loss_func)
            self._critic.update_with_gradient(LossInfo(loss=critic_loss))

        # construct functional gradient
        loss = loss_func(outputs.detach())
        critic_outputs = self._critic.predict_step(outputs.detach()).output
        loss_propagated = torch.sum(-critic_outputs.detach() * outputs, dim=-1)

        return loss, loss_propagated

    def _gfsf_grad(self,
                   inputs,
                   outputs,
                   loss_func,
                   entropy_regularization,
                   transform_func=None):
        """Compute particle gradients via GFSF (Stein estimator). """
        assert inputs is None, '"gfsf" does not support conditional generator'
        if transform_func is not None:
            outputs, extra_outputs, _ = transform_func(outputs)
            aug_outputs = torch.cat([outputs, extra_outputs], dim=-1)
        else:
            aug_outputs = outputs
        score_inputs = aug_outputs.detach()
        loss_inputs = outputs
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(),
                                        loss_inputs)[0]  # [N2, D]
        logq_grad = self._score_func(score_inputs) * entropy_regularization

        loss_prop_neglogp = torch.sum(loss_grad.detach() * outputs, dim=-1)
        loss_prop_logq = torch.sum(logq_grad.detach() * aug_outputs, dim=-1)
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

        if self._critic_relu_mlp:
            critic_step = self._critic.predict_step(
                inputs, requires_jac_diag=True)
            outputs, jac_diag = critic_step.output
            tr_gradf = jac_diag.sum(-1)  # [N]
        else:
            outputs = self._critic.predict_step(inputs).output
            tr_gradf = self._jacobian_trace(outputs, inputs)  # [N]

        f_loss_grad = (loss_grad.detach() * outputs).sum(1)  # [N]
        loss_stein = f_loss_grad - entropy_regularization * tr_gradf  # [N]

        l2_penalty = (outputs * outputs).sum(1).mean() * self._critic_l2_weight
        critic_loss = loss_stein.mean() + l2_penalty

        return critic_loss

    def _minmax_grad(self,
                     inputs,
                     outputs,
                     loss_func,
                     entropy_regularization,
                     transform_func=None):
        """
        Compute particle gradients via minmax svgd (Fisher Neural Sampler). 
        """
        assert inputs is None, '``minmax`` does not support conditional generator'

        # optimize the critic using resampled particles
        if transform_func is not None:
            outputs, _ = transform_func(outputs)
        num_particles = outputs.shape[0]

        for i in range(self._critic_iter_num):

            if self._minmax_resample:
                critic_inputs, _ = self._predict(
                    inputs, batch_size=num_particles)
                if transform_func is not None:
                    critic_inputs, _ = transform_func(critic_inputs)
            else:
                critic_inputs = outputs.detach().clone()
                critic_inputs.requires_grad = True

            critic_loss = self._critic_train_step(critic_inputs, loss_func,
                                                  entropy_regularization)
            self._critic.update_with_gradient(LossInfo(loss=critic_loss))

        # compute amortized svgd
        loss = loss_func(outputs.detach())
        critic_outputs = self._critic.predict_step(outputs.detach()).output
        loss_propagated = torch.sum(-critic_outputs.detach() * outputs, dim=-1)

        return loss, loss_propagated

    def after_update(self, training_info):
        if self._predict_net:
            self._predict_net_updater()
