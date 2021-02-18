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
 

class HMCTest(alf.test.TestCase):

    def generate_class_data(self, n_samples=100, n_classes=4):
        if n_classes == 4:
            means = [(2., 2.), (-2., 2.), (2., -2.), (-2., -2.)]
        else:
            means = [(2., 2.), (-2., -2.)]
        data = torch.zeros(n_samples, 2)
        labels = torch.zeros(n_samples)
        size = n_samples//len(means)
        for i, (x, y) in enumerate(means):
            dist = torch.distributions.Normal(torch.tensor([x, y]), .3)
            samples = dist.sample([size])
            data[size*i:size*(i+1)] = samples
            labels[size*i:size*(i+1)] = torch.ones(len(samples)) * i
        return data, labels.long()
    

    def test_classification_hmc(self):
        n_train = 100
        n_test = 20
        n_classes = 2
        inputs, targets = self.generate_class_data(n_train, n_classes)
        net = BNN([2, 10, 10, n_classes])
        params_init = torch.cat([p.flatten() for p in net.parameters()]).clone()
        tau_list = []
        tau = 1.
        for p in net.parameters():
            tau_list.append(tau)
        tau_list = torch.tensor(tau_list)
        step_size = .005
        num_samples = 200
        steps_per_sample = 25
        tau_out = 1.
        burn_in_steps= 100
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
            test_data, test_labels = self.generate_class_data(n_test, n_classes)
            preds, log_probs = algorithm.predict_model(test_data, test_labels,
                samples=hmc_params)
            test_log_prob = torch.stack(log_probs).mean()
            test_loss = F.cross_entropy(preds.mean(0), test_labels).mean()
            mean_preds = preds.mean(0).argmax(-1).cpu()
            test_acc = mean_preds.eq(test_labels.cpu().view_as(mean_preds))
            test_acc = test_acc.sum().item() / len(test_labels)
            print ('Test log prob: {}'.format(test_log_prob))
            print ('Test loss: {}'.format(test_loss))
            print ('Test acc: {}'.format(test_acc))
            self.assertLess(test_loss, 0.1)
            self.assertGreater(test_acc, 0.95)
        
        hmc_params = _train()
        bnn_preds = _test(hmc_params)


if __name__ == "__main__":
    alf.test.main()
