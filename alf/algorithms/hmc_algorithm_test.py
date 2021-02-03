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

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import alf
from alf.algorithms.hmc_algorithm import HMC
from alf.tensor_specs import TensorSpec

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
import matplotlib.cm as cm
import seaborn as sns


class BNN(nn.Module):
    def __init__(self, n_hidden):
        super(BNN, self).__init__()
        self.n_hiden = n_hidden
        self.layers = []
        self.linear1 = nn.Linear(n_hidden[0], n_hidden[1], bias=True)
        self.linear2 = nn.Linear(n_hidden[1], n_hidden[2], bias=True)
        self.linear3 = nn.Linear(n_hidden[2], n_hidden[3], bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)
 

class BNN2(nn.Module):
    def __init__(self, n_hidden):
        super(BNN2, self).__init__()
        self.n_hiden = n_hidden
        self.layers = []
        self.linear1 = nn.Linear(n_hidden[0], n_hidden[1], bias=False)

    def forward(self, x):
        return self.linear1(x)
              
class BNN3(nn.Module):
    def __init__(self, n_hidden):
        super(BNN3, self).__init__()
        self.n_hiden = n_hidden
        self.layers = []
        self.linear1 = nn.Linear(n_hidden[0], n_hidden[1], bias=True)
        self.linear2 = nn.Linear(n_hidden[1], n_hidden[2], bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)
 
class HMCTest(alf.test.TestCase):

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
    
    def plot_bnn_regression(self, bnn_preds, data):
        sns.set_style('darkgrid')
        gt_x = torch.linspace(-6, 6, 500).view(-1, 1).cpu()
        gt_y = -(1+gt_x) * torch.sin(1.2*gt_x) 
        #gt_y += torch.ones_like(gt_x).normal_(0, 0.04).cpu()
        (x_train, y_train), (x_test, y_test) = data
        x_test = x_test.cpu().numpy()
        x_train = x_train.cpu().numpy()
        bnn_preds = bnn_preds.cpu()
        mean = bnn_preds.mean(0).squeeze()
        std = bnn_preds.std(0).squeeze()
        print (x_test.shape, bnn_preds.shape)

        plt.fill_between(x_test.squeeze(), mean.T+2*std.T, mean.T-2*std.T, alpha=0.5)
        plt.plot(gt_x, gt_y, color='red', label='ground truth')
        plt.plot(x_test, mean.T, label='posterior mean', alpha=0.9)
        plt.scatter(x_train, y_train.cpu().numpy(),color='r', marker='+',
            label='train pts', alpha=1.0, s=50)
        plt.legend(fontsize=14, loc='best')
        plt.ylim([-6, 8])
        plt.savefig('plots/hmc_bnn_ss0005.png')
        plt.close('all')

    def test_BayesianNNRegression(self):
        n_train = 80
        n_test = 200
        train_samples, test_samples = self.generate_regression_data(
            n_train, n_test)
        net = BNN3([1, 50, 1])
        params_init = torch.cat([
            p.flatten() for p in net.parameters()]).clone()
        tau_list = []
        tau = .1
        for p in net.parameters():
            tau_list.append(tau)
        tau_list = torch.tensor(tau_list)
        step_size = 0.0005
        num_samples = 1000
        burn_in_steps=0
        steps_per_sample = 25
        tau_out = 100.
        print ('HMC: Fitting BNN to regression data')
        algorithm = HMC(
            params=params_init,
            num_samples=num_samples,
            steps_per_sample=steps_per_sample,
            step_size=step_size,
            model=net,
            burn_in_steps=burn_in_steps,
            model_loss='regression',
            tau_list=tau_list,
            tau_out=tau_out)

        def _train():
            train_data, train_labels = train_samples
            params_hmc = algorithm.sample_model(train_data, train_labels)
            return params_hmc

        def _test(hmc_params):
            test_data, test_labels = test_samples
            preds, log_probs = algorithm.predict_model(test_data, test_labels,
                samples=hmc_params)
            print ('Expected test log probability: {}'.format(torch.stack(
                log_probs).mean()))
            print ('Expected MSE: {}'.format(
                ((preds.mean(0) - test_labels)**2).mean()))
            return preds
        
        params = []
        if False:
            for i in range(10):
                hmc_params = _train()
                params = torch.stack(hmc_params).detach().cpu().numpy()
                np.save('plots/hmc/trial_ss0005/hmc_regression_params_{}.npy'.format(i), _params)
                #_params = np.load('plots/hmc/trial10k/hmc_regression_params_{}.npy'.format(i))
                params.append(torch.from_numpy(_params))
        
        #params = _train()
        #params = torch.stack(params)
        #print (params.shape)
        #params = params[:, ::100, :]
        #hmc_params = params.reshape(-1, 151).cuda()[-4000:]
        #hmc_params = hmc_params[::25]
        #bnn_preds = _test(hmc_params)
        #self.plot_bnn_regression(bnn_preds, (train_samples, test_samples))
    
    def generate_class_data(self,
        n_samples=100,
        means=[(2., 2.), (-2., 2.), (2., -2.), (-2., -2.)]):
        #means=[(2., 2.), (-2., -2.)]):
        data = torch.zeros(n_samples, 2)
        labels = torch.zeros(n_samples)
        size = n_samples//len(means)
        for i, (x, y) in enumerate(means):
            dist = torch.distributions.Normal(torch.tensor([x, y]), .3)
            samples = dist.sample([size])
            data[size*i:size*(i+1)] = samples
            labels[size*i:size*(i+1)] = torch.ones(len(samples)) * i
       
        plt.scatter(data[:, 0].cpu(), data[:, 1].cpu())
        plt.savefig('data_space.png')
        plt.close('all')
        return data, labels.long()
    
    def plot_bnn_classification(self, i, algorithm, samples):
        x = torch.linspace(-12, 12, 100)
        y = torch.linspace(-12, 12, 100)
        gridx, gridy = torch.meshgrid(x, y)
        grid = torch.stack((gridx.reshape(-1), gridy.reshape(-1)), -1)
        outputs, _ = algorithm.predict_model(grid, y=None, samples=samples)
        outputs = F.softmax(outputs, dim=-1)  # [B, D]
        torch.save(outputs, 'plots/hmc/classification/outputs_{}.pt'.format(i))
        mean_outputs = outputs.mean(0).cpu()
        std_outputs = outputs.std(0).cpu()

        conf_std = std_outputs.max(-1)[0] * 1.96
        labels = mean_outputs.argmax(-1)
        data, _ = self.generate_class_data(n_samples=400) 
        
        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_std, cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("confidance (std)")
        plt.savefig('plots/hmc/classification/conf_map-std_{}.png'.format(i))
        plt.close('all')
        
        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=labels, cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("predicted labels")
        plt.savefig('plots/hmc/classification/conf_map-labels_{}.png'.format(i))
        plt.close('all')

    def test_BayesianNNClassification(self):
        n_train = 100
        n_test = 20
        inputs, targets = self.generate_class_data(n_train)
        net = BNN([2, 10, 10, 4])
        params_init = torch.cat([p.flatten() for p in net.parameters()]).clone()
        tau_list = []
        tau = 1.
        for p in net.parameters():
            tau_list.append(tau)
        tau_list = torch.tensor(tau_list)
        step_size = .005
        num_samples = 10000
        steps_per_sample = 25
        tau_out = 1.
        burn_in_steps= 9800
        print ('HMC: Fitting BNN to classification data')
        algorithm = HMC(
            params=params_init,
            num_samples=num_samples,
            steps_per_sample=steps_per_sample,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            model=net,
            model_loss='classification',
            tau_list=tau_list,
            tau_out=tau_out)

        def _train():
            params_hmc = algorithm.sample_model(inputs, targets)
            return params_hmc

        def _test(hmc_params):
            test_data, test_labels = self.generate_class_data(n_test)
            preds, log_probs = algorithm.predict_model(test_data, test_labels,
                samples=hmc_params)
            print ('Expected test log probability: {}'.format(torch.stack(
                log_probs).mean()))
            print ('Expected XE loss: {}'.format(
                F.cross_entropy(preds.mean(0), test_labels).mean()))
            return preds
        
        #for i in range(10, 20):
        #    hmc_params = _train()
        #    bnn_preds = _test(hmc_params)
        #    _params = torch.stack(hmc_params).detach().cpu()
            #_parmas = _params[::25]
        #    print ('plotting')
        #    with torch.no_grad():
        #        self.plot_bnn_classification(num_samples, algorithm, hmc_params)
        
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

    def test_bayesian_linear_regression(self):
        """
        HMC is used to sample the parameter vector for a linear
        regressor. The target linear regressor is :math:`y = X\beta + e`, where 
        :math:`e\sim N(0, I)` is random noise, :math:`X` is the input data matrix, 
        and :math:`y` is target ouputs. The posterior of :math:`\beta` has a 
        closed-form :math:`p(\beta|X,y)\sim N((X^TX)^{-1}X^Ty, X^TX)`.
        For a linear model with weight W and bias b, and standard Gaussian prior,
        the output follows a Gaussian :math:`N(b, WW^T)`, which should 
        match the posterior :math:`p(\beta|X,y)`.
        """
        input_size = 3
        input_spec = TensorSpec((input_size, ), torch.float32)
        output_dim = 1
        batch_size = 100
        inputs = input_spec.randn(outer_dims=(batch_size, ))
        beta = torch.rand(input_size, output_dim) + 5.
        print("beta: {}".format(beta))
        noise = torch.randn(batch_size, output_dim)
        targets = inputs @ beta + noise
        true_cov = torch.inverse(
            inputs.t() @ inputs) 
        true_mean = true_cov @ inputs.t() @ targets
        net = BNN2([input_size, output_dim])
        params_init = torch.cat([p.flatten() for p in net.parameters()]).clone()
        tau_list = []
        tau = 1.
        for p in net.parameters():
            tau_list.append(tau)
        tau_list = torch.tensor(tau_list)
        step_size = .0005
        num_samples = 20000
        steps_per_sample = 50
        tau_out = 1.
        burn_in_steps= 19800
        print ('HMC: Fitting BNN to classification data')
        algorithm = HMC(
            params=params_init,
            num_samples=num_samples,
            steps_per_sample=steps_per_sample,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            model=net,
            model_loss='regression',
            tau_list=tau_list,
            tau_out=tau_out)

        def _train():
            params_hmc = algorithm.sample_model(inputs, targets)
            return params_hmc

        def _test(params):
            print("-" * 68)
            preds, log_probs = algorithm.predict_model(inputs, targets,
                samples=params)
            params = torch.stack(params)
            computed_mean = params.mean(0)
            computed_cov = self.cov(params)
            print("-" * 68)
            spred_err = torch.norm((preds - targets).mean(1))
            print("sampled pred err: ", spred_err)

            smean_err = torch.norm(computed_mean - true_mean.squeeze())
            smean_err = smean_err / torch.norm(true_mean)
            print("sampled mean err: ", smean_err)

            computed_cov = self.cov(params)
            scov_err = torch.norm(computed_cov - true_cov)
            scov_err = scov_err / torch.norm(true_cov)
            print("sampled cov err: ", scov_err)
            
            self.assertLess(smean_err, .5)
            self.assertLess(scov_err, .5)

        params_hmc = _train()
        _test(params_hmc)

        print("ground truth mean: {}".format(true_mean))
        print("ground truth cov norm: {}".format(true_cov.norm()))

if __name__ == "__main__":
    alf.test.main()
