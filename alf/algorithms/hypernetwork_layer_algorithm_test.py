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

from absl.testing import parameterized
import numpy as np
import torch
import torch.nn.functional as F

import alf
from alf.algorithms.config import TrainerConfig
from alf.data_structures import LossInfo
from alf.algorithms.hypernetwork_layer_algorithm import HyperNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops, datagen, common
import time


class HyperNetworkTest(parameterized.TestCase, alf.test.TestCase):
    def cov(self, data, rowvar=False):
        """Estimate a covariance matrix given data.

        Args:
            data (tensor): A 1-D or 2-D tensor containing multiple observations 
                of multiple dimentions. Each row of ``mat`` represents a
                dimension of the observation, and each column a single
                observation.
            rowvar (bool): If True, then each row represents a dimension, with
                observations in the columns. Othewise, each column represents
                a dimension while the rows contains observations.

        Returns:
            The covariance matrix
        """
        x = data.detach().clone()
        if x.dim() > 2:
            raise ValueError('data has more than 2 dimensions')
        if x.dim() < 2:
            x = x.view(1, -1)
        if not rowvar and x.size(0) != 1:
            x = x.t()
        fact = 1.0 / (x.size(1) - 1)
        x -= torch.mean(x, dim=1, keepdim=True)
        return fact * x.matmul(x.t()).squeeze()

    def assertArrayGreater(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertGreater(float(torch.min(x - y)), eps)

    """
    @parameterized.parameters(
                              ('svgd3', False, None), # A-SVGD
                              #('svgd3', True, None),  #A-SVGD-fv
                              #('gfsf', False, None),  #A-GFSF
                              #('gfsf', True, None),  #A-GFSF-fv
                              #('minmax', False, None), #A-minmax
                              ('svgd3', False, 'rkhs'), #G-SVGD
                              #('minmax', True, None),  #A-minmax-fv
                              #('minmax', False, 'minmax', 100),  #G-minmax
    )"""

    def test_bayesian_linear_regression(self,
                                        par_vi='svgd3',
                                        function_vi=False,
                                        functional_gradient='rkhs',
                                        train_batch_size=10,
                                        num_particles=100):
        """
        The hypernetwork is trained to generate the parameter vector for a linear
        regressor. The target linear regressor is :math:`y = X\beta + e`, where 
        :math:`e\sim N(0, I)` is random noise, :math:`X` is the input data matrix, 
        and :math:`y` is target ouputs. The posterior of :math:`\beta` has a 
        closed-form :math:`p(\beta|X,y)\sim N((X^TX)^{-1}X^Ty, X^TX)`.
        For a linear generator with weight W and bias b, and takes standard Gaussian 
        noise as input, the output follows a Gaussian :math:`N(b, WW^T)`, which should 
        match the posterior :math:`p(\beta|X,y)` for both ``svgd``, ``gfsf``, and
        ``minmax``.
        """

        print("Testing {} with {} particles".format(par_vi, num_particles))
        print("Function values: {}, \nFunctional gradient: {}".format(
            function_vi, functional_gradient))
        common.set_random_seed(None)
        input_size = 3
        input_spec = TensorSpec((input_size, ), torch.float32)
        output_dim = 1
        batch_size = 100
        hidden_size = output_dim * batch_size
        inputs = input_spec.randn(outer_dims=(batch_size, ))
        beta = torch.rand(input_size, output_dim) + 5.
        print("beta: {}".format(beta))
        noise = torch.randn(batch_size, output_dim)
        targets = inputs @ beta + noise
        true_cov = torch.inverse(inputs.t() @ inputs)
        true_mean = true_cov @ inputs.t() @ targets
        noise_dim = input_size
        lr = 1e-3
        if functional_gradient:
            hidden_layers = ()
            parameterization = 'network'
        else:
            hidden_layers = None
            parameterization = 'network'

        algorithm = HyperNetwork(
            input_spec,
            last_layer_param=(output_dim, False),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            hidden_layers=hidden_layers,
            loss_type='regression',
            par_vi=par_vi,
            num_particles=num_particles,
            functional_gradient=functional_gradient,
            function_vi=function_vi,
            function_bs=train_batch_size,
            pinverse_solve_iters=1,
            pinverse_hidden_size=10,
            fullrank_diag_weight=1,
            #exact_inverse=True,
            parameterization=parameterization,
            critic_hidden_layers=(hidden_size, hidden_size),
            critic_iter_num=5,
            critic_l2_weight=10,
            optimizer=alf.optimizers.Adam(lr=lr),
            critic_optimizer=alf.optimizers.Adam(lr=lr),
            pinverse_optimizer=alf.optimizers.Adam(lr=1e-4))

        print("ground truth mean: {}".format(true_mean))
        print("ground truth cov norm: {}".format(true_cov.norm()))
        print("ground truth cov: {}".format(true_cov))

        def _train(i, train_batch=None, entropy_regularization=None):
            if train_batch is None:
                perm = torch.randperm(batch_size)
                idx = perm[:train_batch_size]
                train_inputs = inputs[idx]
                train_targets = targets[idx]
            else:
                train_inputs, train_targets = train_batch
            if entropy_regularization is None:
                entropy_regularization = train_batch_size / batch_size

            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                entropy_regularization=entropy_regularization,
                num_particles=num_particles)

            if functional_gradient:
                pinverse_loss = alg_step.info.extra.pinverse
                if i % 500 == 0:
                    print(pinverse_loss)

            loss_info, params = algorithm.update_with_gradient(alg_step.info)

        def _test(i, sampled_predictive=False, s=100):

            print("-" * 68)
            if parameterization == 'layer':
                weight = algorithm._generator._net.layer_encoders[
                    0]._fc_layers[0].weight
                learned_mean = algorithm._generator._net.layer_encoders[
                    0]._fc_layers[0].bias
            else:
                weight = algorithm._generator._net._fc_layers[0].weight
                learned_mean = algorithm._generator._net._fc_layers[0].bias
            print(weight.shape)
            learned_cov = weight @ weight.t()
            print("norm of generator weight: {}".format(weight.norm()))
            print("norm of learned cov: {}".format(learned_cov.norm()))

            predicts = inputs @ learned_mean  # [batch]
            pred_err = torch.norm(predicts - targets.squeeze())

            mean_err = torch.norm(learned_mean - true_mean.squeeze())
            mean_err = mean_err / torch.norm(true_mean)

            cov_err = torch.norm(learned_cov - true_cov)
            cov_err = cov_err / torch.norm(true_cov)

            print("Train Iter: {}".format(i))
            print("\tpred err {}".format(pred_err))
            print("\tmean err {}".format(mean_err))
            print("\tcov err {}".format(cov_err))

            if sampled_predictive:
                params = algorithm.sample_parameters(num_particles=100)
                pred_step = algorithm.predict_step(inputs, params=params)
                if functional_gradient:
                    params = params[0]
                sampled_preds = pred_step.output.squeeze(
                )  # [batch, n_particles]
                spred_err = torch.norm((sampled_preds - targets).mean(1))
                print("train_iter {}: sampled pred err {}".format(
                    i, spred_err))

                computed_mean = params.mean(0)
                smean_err = torch.norm(computed_mean - true_mean.squeeze())
                smean_err = smean_err / torch.norm(true_mean)
                print("train_iter {}: sampled mean err {}".format(
                    i, smean_err))

                computed_cov = self.cov(params)
                scov_err = torch.norm(computed_cov - true_cov)
                scov_err = scov_err / torch.norm(true_cov)
                print("train_iter {}: sampled cov err {}".format(i, scov_err))

        train_iter = 50000
        for i in range(train_iter):
            _train(i)
            if i % 1000 == 0:
                _test(i)

        if parameterization == 'layer':
            weight = algorithm._generator._net.layer_encoders[0]._fc_layers[
                0].weight
            learned_mean = algorithm._generator._net.layer_encoders[
                0]._fc_layers[0].bias
        else:
            weight = algorithm._generator._net._fc_layers[0].weight
            learned_mean = algorithm._generator._net._fc_layers[0].bias
        mean_err = torch.norm(learned_mean - true_mean.squeeze())
        mean_err = mean_err / torch.norm(true_mean)
        learned_cov = weight @ weight.t()
        cov_err = torch.norm(learned_cov - true_cov)
        cov_err = cov_err / torch.norm(true_cov)
        print("-" * 68)
        print("train_iter {}: mean err {}".format(train_iter, mean_err))
        print("train_iter {}: cov err {}".format(train_iter, cov_err))
        self.assertLess(mean_err, 0.5)
        self.assertLess(cov_err, 0.5)


if __name__ == "__main__":
    alf.test.main()
