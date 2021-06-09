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
from alf.algorithms.hypernetwork_algorithm import HyperNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
import matplotlib.cm as cm
import seaborn as sns

import os


class HyperNetworkRegressionTest(parameterized.TestCase, alf.test.TestCase):
    def generate_regression_data(self, n_train, n_test):
        x_train1 = torch.linspace(-6, -2, n_train // 2).view(-1, 1)
        x_train2 = torch.linspace(2, 6, n_train // 2).view(-1, 1)
        x_train3 = torch.linspace(-2, 2, 4).view(-1, 1)
        x_train = torch.cat((x_train1, x_train2, x_train3), dim=0)
        y_train = -(1 + x_train) * torch.sin(1.2 * x_train)
        y_train = y_train + torch.ones_like(y_train).normal_(0, 0.04)

        x_test = torch.linspace(-6, 6, n_test).view(-1, 1)
        y_test = -(1 + x_test) * torch.sin(1.2 * x_test)
        y_test = y_test + torch.ones_like(y_test).normal_(0, 0.04)
        return (x_train, y_train), (x_test, y_test)

    def plot_bnn_regression(self, i, algorithm, data, tag):
        sns.set_style('darkgrid')
        basedir = 'plots/regression/funcgrad_par_vi/{}/'.format(tag)
        os.makedirs(basedir, exist_ok=True)
        gt_x = torch.linspace(-6, 6, 500).view(-1, 1).cpu()
        gt_y = -(1 + gt_x) * torch.sin(1.2 * gt_x)
        (x_train, y_train), (x_test, y_test) = data
        outputs = algorithm.predict_step(
            x_test, num_particles=100).output.cpu()
        mean = outputs.mean(1).squeeze()
        std = outputs.std(1).squeeze()
        x_test = x_test.cpu().numpy()
        x_train = x_train.cpu().numpy()

        plt.fill_between(
            x_test.squeeze(),
            mean.T + 2 * std.T,
            mean.T - 2 * std.T,
            alpha=0.5)
        plt.plot(gt_x, gt_y, color='red', label='ground truth')
        plt.plot(x_test, mean.T, label='posterior mean', alpha=0.9)
        plt.scatter(
            x_train,
            y_train.cpu().numpy(),
            color='r',
            marker='+',
            label='train pts',
            alpha=1.0,
            s=50)
        plt.legend(fontsize=14, loc='best')
        plt.ylim([-6, 8])
        plt.savefig('{}/iter_{}.png'.format(basedir, i))
        plt.close('all')

    def test_regression(self, par_vi='svgd3', functional_gradient='rkhs'):
        input_size = 1
        output_dim = 1
        noise_dim = 151
        num_particles = 100
        train_batch_size = 80
        batch_size = 80
        test_batch_size = 200
        lr = 5e-4

        input_spec = TensorSpec((input_size, ), torch.float64)
        train_samples, test_samples = self.generate_regression_data(
            batch_size, test_batch_size)
        inputs, targets = train_samples
        test_inputs, test_targets = test_samples
        print('Fitting BNN to regression data')
        algorithm = HyperNetwork(
            input_tensor_spec=input_spec,
            fc_layer_params=((50, True), ),
            last_layer_param=(output_dim, True),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            hidden_layers=(151, 151, 151),
            loss_type='regression',
            par_vi=par_vi,
            functional_gradient=functional_gradient,
            pinverse_solve_iters=5,
            force_fullrank=True,
            pinverse_hidden_size=256,
            fullrank_diag_weight=1.0,
            num_particles=num_particles,
            optimizer=alf.optimizers.Adam(lr=lr),
            pinverse_optimizer=alf.optimizers.Adam(lr=1e-4))

        def _train(entropy_regularization=None):
            train_inputs = inputs
            train_targets = targets
            if entropy_regularization is None:
                entropy_regularization = train_batch_size / batch_size
                entropy_regularization = 1e-10

            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                entropy_regularization=entropy_regularization,
                num_particles=num_particles,
                state=())
            loss_info, params = algorithm.update_with_gradient(alg_step.info)

        def _test(i):
            outputs, _ = algorithm._param_net(test_inputs)
            mse_err = (outputs.mean(1) - test_targets).pow(2).mean()
            print('step [{}]: Expected MSE: {}'.format(i, mse_err))

        for i in range(200000):
            _train()
            if i % 200 == 0:
                _test(i)
                with torch.no_grad():
                    tag = par_vi + '_pi5_ph256_3l_ent1e10_lr{}'.format(lr)
                    data = (train_samples, test_samples)
                    self.plot_bnn_regression(i, algorithm, data, tag)


if __name__ == "__main__":
    alf.test.main()
