# Copyright (c) 2021 Horizon Robotics. All Rights Reserved.
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
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
import matplotlib.cm as cm
import seaborn as sns
matplotlib.rcParams.update({'font.size': 18})
import os


class FuncParVIRegressionTest(parameterized.TestCase, alf.test.TestCase):
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
        basedir = 'plots/regression/functional_par_vi/{}/'.format(tag)
        os.makedirs(basedir, exist_ok=True)
        sns.set_style('darkgrid')
        gt_x = torch.linspace(-6, 6, 500).view(-1, 1).cpu()
        gt_y = -(1 + gt_x) * torch.sin(1.2 * gt_x)
        (x_train, y_train), (x_test, y_test) = data
        outputs = algorithm.predict_step(x_test).output.cpu()
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
        plt.legend(fontsize=18, loc='best')
        plt.ylim([-6, 8])
        plt.savefig(basedir + 'iter_{}.png'.format(i))
        plt.close('all')

    @parameterized.parameters(
        ('svgd', False),
        #('svgd', True),
        #('gfsf', False),
        #('gfsf', True),
        #(None, False),
    )
    def test_regression_func_par_vi(self,
                                    par_vi='svgd',
                                    function_vi=False,
                                    num_particles=100):
        n_train = 80
        n_test = 200
        input_size = 1
        output_dim = 1
        input_spec = TensorSpec((input_size, ), torch.float64)
        train_batch_size = n_train
        batch_size = n_train

        train_samples, test_samples = self.generate_regression_data(
            n_train, n_test)
        inputs, targets = train_samples
        test_inputs, test_targets = test_samples
        print('{} - {} particles'.format(par_vi, num_particles))
        print('Functional Particle VI: Fitting Regressors')
        print("Function VI: {}".format(function_vi))

        algorithm = FuncParVIAlgorithm(
            input_tensor_spec=input_spec,
            fc_layer_params=((50, True), ),
            last_layer_param=(output_dim, True),
            last_activation=math_ops.identity,
            num_particles=num_particles,
            loss_type='regression',
            par_vi=par_vi,
            function_vi=function_vi,
            function_bs=train_batch_size,
            optimizer=alf.optimizers.Adam(lr=1e-2))

        def _train(entropy_regularization=None):
            train_inputs = inputs
            train_targets = targets
            if entropy_regularization is None:
                entropy_regularization = train_batch_size / batch_size

            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                entropy_regularization=entropy_regularization)

            loss_info, params = algorithm.update_with_gradient(alg_step.info)

        def _test(i):
            outputs, _ = algorithm._param_net(test_inputs)
            mse_err = (outputs.mean(1) - test_targets).pow(2).mean()
            print('Expected MSE: {}'.format(mse_err))

        for i in range(10000):
            _train()
            if i % 200 == 0:
                _test(i)
                with torch.no_grad():
                    data = (train_samples, test_samples)
                    tag = par_vi
                    if function_vi:
                        tag += '_fvi'
                    self.plot_bnn_regression(i, algorithm, data, tag)


if __name__ == "__main__":
    alf.test.main()
