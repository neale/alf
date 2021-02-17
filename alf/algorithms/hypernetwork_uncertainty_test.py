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
from alf.algorithms.hypernetwork_layer_algorithm import HyperNetwork
from alf.algorithms.hypernetwork_networks import ParamConvNet, ParamNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
import matplotlib.cm as cm
import seaborn as sns

import os

class HyperNetworkSampleTest(parameterized.TestCase, alf.test.TestCase):
    """ 
    HyperNetwork Sample Test
        Two tests given in order of increasing difficulty. 
        1. A 4 class classification problem, where the classes are distributed
            as 4 symmetric Normal distributions with non overlapping support. 
            The hypernetwork is trained to sample classification functions that
            fit the data, for the purpose of observing the predictive
            distributions of sampled funcitons on data outside the training
            distribution. 
        2. A regression problem that I will detail here
        
        We do not directly compute the posterior for any of these distributions
        We instead determine closeness qualitatively by sampling predictors from
        our hypernetwork, and comparing the resulting prediction statistics to 
        samples drawn from an HMC-based neural network. 
    """
    
    def generate_class_data(self, n_samples=200, n_classes=4):
        
        if n_classes == 4: 
            means=[(2., 2.), (-2., 2.), (2., -2.), (-2., -2.)]
        else:
            means=[(2., 2.), (-2., -2.)]

        data = torch.zeros(n_samples, 2)
        labels = torch.zeros(n_samples)
        size = n_samples//len(means)
        for i, (x, y) in enumerate(means):
            dist = torch.distributions.Normal(torch.tensor([x, y]), .3)
            samples = dist.sample([size])
            data[size*i:size*(i+1)] = samples
            labels[size*i:size*(i+1)] = torch.ones(len(samples)) * i
        
        return data, labels.long()
    
    def plot_classification(self, i, algorithm, n_classes, tag=''):
        basedir = 'plots/{}'.format(tag)
        os.makedirs(basedir, exist_ok=True)
        x = torch.linspace(-12, 12, 100)
        y = torch.linspace(-12, 12, 100)
        gridx, gridy = torch.meshgrid(x, y)
        grid = torch.stack((gridx.reshape(-1), gridy.reshape(-1)), -1)
        
        outputs = algorithm.predict_step(grid, num_particles=100).output.cpu()
        outputs = F.softmax(outputs, -1).detach()

        std_outputs = outputs.std(1)
        conf_std = std_outputs.max(-1)[0] * 1.94
        mean_outputs = outputs.mean(1)  # [B, D]
        conf_mean = mean_outputs.mean(-1)
        labels = mean_outputs.argmax(-1)
        
        data, _ = self.generate_class_data(n_classes=n_classes, n_samples=400)
        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_std,
            cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black',
            alpha=0.1)
        cbar = plt.colorbar(p1)
        cbar.set_label("Confidance (std)")
        plt.savefig(basedir+'std-map_{}.png'.format(i))
        plt.close('all')
        
        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=labels,
            cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black',
            alpha=0.1)
        cbar = plt.colorbar(p1)
        cbar.set_label("Predicted Labels")
        plt.savefig(basedir+'label-map_{}.png'.format(i))
        print ('saved figure: ', basedir)
        plt.close('all')

    @parameterized.parameters(
        ('svgd3', False, None),
        ('svgd3', True, None),
        ('svgd3', False, 'rkhs'),
        ('gfsf', False, None),
        ('gfsf', True, None),
        ('minmax', False, None),
        ('minmax', True, None),
        #('minmax', False, 'minmax'),
    )
    def test_classification_hypernetwork(self,
                                         par_vi='minmax',
                                         function_vi=False,
                                         functional_gradient='rkhs',
                                         n_classes=4,
                                         num_particles=100):
        """
        Symmetric 4-class classification problem. The training data are drawn
        from standard normal distributions, each class is given by one of
        these distributions. The hypernetwork is trained to generate parameters
        that achieves low loss / high accuracy on this data.
        """

        print ('Hypernetwork: Fitting {} Classes'.format(n_classes))
        print ('params: {} - {} particles'.format(par_vi, num_particles))
        print ("Function_vi : {}, Functional Gradient".format(
            function_vi, functional_gradient))
        input_size = 2
        output_dim = n_classes
        input_spec = TensorSpec((input_size, ), torch.float64)
        train_batch_size = 100
        parameterization = 'network'
        
        train_nsamples = 100
        test_nsamples = 200
        batch_size = train_nsamples
        inputs, targets = self.generate_class_data(train_nsamples, n_classes)
        test_inputs, test_targets = self.generate_class_data(test_nsamples, n_classes)
        if n_classes == 4:
            param_dim = 184
        else:
            param_dim = 162
        noise_dim = param_dim
        hidden_size = 64
        chidden_size =  100
        pinverse_iters = 1
        pinverse_hidden_size = param_dim*3
        lr = 1e-4
        weight_decay = 0
        critic_iter_num = 5
        critic_l2_weight = 10.
        algorithm = HyperNetwork(
            input_tensor_spec=input_spec,
            fc_layer_params=((10, True), (10, True)),
            last_layer_param=(output_dim, True),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            hidden_layers=(hidden_size, hidden_size),
            num_particles=num_particles,
            loss_type='classification',
            par_vi=par_vi,
            function_vi=function_vi,
            function_bs=train_batch_size,
            functional_gradient=functional_gradient,
            pinverse_solve_iters=pinverse_iters,
            pinverse_hidden_size=pinverse_hidden_size,
            fullrank_diag_weight=1.0,
            force_fullrank=True,
            use_jac_regularization=False,
            parameterization=parameterization,
            critic_hidden_layers=(chidden_size, chidden_size),
            critic_iter_num=critic_iter_num,
            critic_l2_weight=critic_l2_weight,
            optimizer=alf.optimizers.Adam(lr=lr, weight_decay=weight_decay),
            critic_optimizer=alf.optimizers.Adam(lr=lr),
            pinverse_optimizer=alf.optimizers.Adam(lr=1e-4))
        
        def _train(i, entropy_regularization=None):
            perm = torch.randperm(train_nsamples)
            idx = perm[:train_batch_size]
            train_inputs = inputs[idx]
            train_targets = targets[idx]
            if entropy_regularization is None:
                entropy_regularization = (train_batch_size / batch_size)
            
            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                entropy_regularization=entropy_regularization,
                num_particles=num_particles,
                state=())

            if functional_gradient:
                pinverse_loss = alg_step.info.extra.pinverse
                if i % 500 == 0: print ('pl', pinverse_loss)
            
            loss_info, params = algorithm.update_with_gradient(alg_step.info)

        def _test(i):
            outputs, _ = algorithm._param_net(test_inputs)

            params = algorithm.sample_parameters(num_particles=num_particles)
            if functional_gradient:
                params = params[0]
            _params = params.detach().cpu().numpy()

            probs = F.softmax(outputs, dim=-1)
            preds = probs.mean(1).cpu().argmax(-1)
            mean_acc = preds.eq(test_targets.cpu().view_as(preds)).float()
            mean_acc = mean_acc.sum() / len(test_targets)
            
            sample_preds = probs.cpu().argmax(-1).reshape(-1, 1)
            targets_unrolled = test_targets.unsqueeze(1).repeat(
                1, num_particles).reshape(-1, 1)
            
            sample_acc = sample_preds.eq(targets_unrolled.cpu().view_as(sample_preds)).float()
            sample_acc = sample_acc.sum()/len(targets_unrolled)

            print ('-'*86)
            print ('iter ', i)
            print ('mean particle acc: ', mean_acc.item())
            print ('all particles acc: ', sample_acc.item())

            with torch.no_grad():
                tag = par_vi
                if function_vi:
                    tag += '_fvi/'
                if functional_gradient:
                    tag += '_fg-vi/'
                else:
                    tag += '/'
                if par_vi == 'minmax':
                    sub = '{}cls/{}z_2h{}_40lr_ad{}w{}_{}iter_{}h_ci{}_l2{}'.format(
                        n_classes, noise_dim, hidden_size, lr, weight_decay,
                        pinverse_iters, chidden_size, critic_iter_num,
                        critic_l2_weight)
                else:
                    sub = '{}cls/{}z_2h{}_40lr_ad{}w{}_{}iter_{}h_{}p_2'.format(
                        n_classes, noise_dim, hidden_size, lr, weight_decay,
                        pinverse_iters, pinverse_hidden_size, num_particles)
                tag += '{}/'.format(sub)
                self.plot_classification(i, algorithm, n_classes, tag)
        """
        train_iter = 500000
        for i in range(train_iter):
            _train(i)
            if i % 2000 == 0:
                _test(i)
        """

    def generate_regression_data(self, n_train, n_test):
        x_train1 = torch.linspace(-6, -2, n_train//2).view(-1, 1)
        x_train2 = torch.linspace(2, 6, n_train//2).view(-1, 1)
        x_train3 = torch.linspace(-2, 2, 4).view(-1, 1)
        x_train = torch.cat((x_train1, x_train2, x_train3), dim=0)
        y_train = -(1 + x_train) * torch.sin(1.2*x_train) 
        y_train = y_train + torch.ones_like(y_train).normal_(0, 0.04)

        x_test = torch.linspace(-6, 6, n_test).view(-1, 1)
        y_test = -(1 + x_test) * torch.sin(1.2*x_test) 
        y_test = y_test + torch.ones_like(y_test).normal_(0, 0.04)
        return (x_train, y_train), (x_test, y_test)
    
    def plot_bnn_regression(self, i, algorithm, data, tag):
        sns.set_style('darkgrid')
        basedir = 'plots/regression/funcgrad_par_vi/{}/'.format(tag)
        os.makedirs(basedir, exist_ok=True)
        gt_x = torch.linspace(-6, 6, 500).view(-1, 1).cpu()
        gt_y = -(1+gt_x) * torch.sin(1.2*gt_x) 
        (x_train, y_train), (x_test, y_test) = data
        outputs = algorithm.predict_step(x_test, num_particles=100).output.cpu()
        mean = outputs.mean(1).squeeze()
        std = outputs.std(1).squeeze()
        x_test = x_test.cpu().numpy()
        x_train = x_train.cpu().numpy()
        print (x_test.shape, outputs.shape, mean.shape, std.shape)

        plt.fill_between(x_test.squeeze(), mean.T+2*std.T, mean.T-2*std.T, alpha=0.5)
        plt.plot(gt_x, gt_y, color='red', label='ground truth')
        plt.plot(x_test, mean.T, label='posterior mean', alpha=0.9)
        plt.scatter(x_train, y_train.cpu().numpy(),color='r', marker='+',
            label='train pts', alpha=1.0, s=50)
        plt.legend(fontsize=14, loc='best')
        plt.ylim([-6, 8])
        plt.savefig('{}/iter_{}.png'.format(basedir, i))
        plt.close('all')
    
    def test_BayesianNNRegression(self):
        n_train = 80
        n_test = 200
        input_size = 1
        output_dim = 1
        noise_dim = 151
        num_particles = 100
        amortize = True
        function_vi = False
        functional_gradient = 'rkhs'
        parameterization = 'network'
        input_spec = TensorSpec((input_size, ), torch.float64)
        train_batch_size = n_train
        batch_size = n_train
        par_vi = 'svgd3'
        train_samples, test_samples = self.generate_regression_data(
            n_train, n_test)
        inputs, targets = train_samples
        test_inputs, test_targets = test_samples
        print ('Fitting BNN to regression data')
        algorithm = HyperNetwork(
            input_tensor_spec=input_spec,
            fc_layer_params=((50, True),),
            last_layer_param=(output_dim, True),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            hidden_layers=(151,151,151),
            loss_type='regression',
            par_vi='svgd3',
            functional_gradient='rkhs',
            pinverse_solve_iters=1,
            force_fullrank=True,
            function_vi=function_vi,
            function_bs=train_batch_size,
            pinverse_hidden_size=100,
            fullrank_diag_weight=1.0,
            num_particles=num_particles,
            parameterization=parameterization,
            optimizer=alf.optimizers.Adam(lr=1e-4),
            pinverse_optimizer=alf.optimizers.Adam(lr=1e-4),
            critic_hidden_layers=(32,32),
            critic_optimizer=alf.optimizers.Adam(lr=1e-3))
        
        def _train(entropy_regularization=None):
            train_inputs = inputs
            train_targets = targets
            if entropy_regularization is None:
                entropy_regularization = train_batch_size / batch_size
                entropy_regularization = 1e-5
            
            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                entropy_regularization=entropy_regularization,
                num_particles=num_particles,
                state=())
            if amortize or function_vi:
                loss_info, params = algorithm.update_with_gradient(alg_step.info)
            else:
                update_direction = alg_step.info.loss
                algorithm._particle_optimizer.zero_grad()
                algorithm._params.grad = update_direction
                algorithm._particle_optimizer.step()

        def _test(i):
            outputs, _ = algorithm._param_net(test_inputs)
            mse_err = (outputs.mean(1) - test_targets).pow(2).mean()
            print ('Expected MSE: {}'.format(mse_err))
        
        for i in range(200000):
            _train()
            if i % 1000 == 0:
                _test(i)
                with torch.no_grad():
                    tag = par_vi
                    if function_vi:
                        tag += '_fvi'
                    data = (train_samples, test_samples)
                    self.plot_bnn_regression(i, algorithm, data, tag)
        
        
if __name__ == "__main__":
    alf.test.main()
