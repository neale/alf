# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

import math

from absl import logging
from absl.testing import parameterized
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import alf
from alf.algorithms.generator import Generator
from alf.algorithms.generative_adversarial_algorithm import GenerativeAdversarialAlgorithm
from alf.networks import Network
from alf.layers import FC
from alf.tensor_specs import TensorSpec
from alf.utils.datagen import Test8GaussiansDataSet

import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class Net(Network):
    def __init__(self, noise_dim, output_dim):
        super().__init__(
            input_tensor_spec=TensorSpec(
                shape=(noise_dim, ), dtype=torch.float32),
            name="Net")

        self.fc1 = FC(noise_dim, 256, activation=F.relu)
        self.fc2 = FC(256, 256, activation=F.relu)
        self.fc3 = FC(256, 256, activation=F.relu)
        self.fc4 = FC(256, output_dim)

    def forward(self, input, state=()):
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x, state


class Net2(Network):
    def __init__(self, input_dim):
        super().__init__(
            input_tensor_spec=TensorSpec(
                shape=(input_dim, ), dtype=torch.float32),
            name="Net2")

        self.fc1 = FC(input_dim, 256, activation=F.relu)
        self.fc2 = FC(256, 256, activation=F.relu)
        self.fc3 = FC(256, 256, activation=F.relu)
        self.fc4 = FC(256, 1)

    def forward(self, input, state=()):
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x.view(-1), state


def generate_image(generator, data, batch_size, i):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))
    points = torch.tensor(points).to(alf.get_default_device())
    critic_map = generator.critic(points)[0].cpu().numpy()

    noise = torch.randn(batch_size, 2).to(alf.get_default_device())
    samples = generator._generator._net(noise, data)[0]

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x, y, critic_map.reshape((len(x), len(y))).transpose())

    data = data.detach().cpu().numpy()
    samples = samples.detach().cpu().numpy()

    plt.scatter(data[:, 0], data[:, 1], c='orange', marker='+')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')
    save_dir = '/nfs/hpc/share/ratzlafn/alf-plots/gan/8Gaussians'
    os.makedirs(save_dir, exist_ok=True)
    print('saving to ', save_dir)
    plt.savefig('{}/gen_{}.png'.format(save_dir, i))
    plt.close('all')


class GenerativeAdversarialTest(parameterized.TestCase, alf.test.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(torch.max(abs(x - y))), eps)

    """
    @parameterized.parameters(
        dict(par_vi='svgd'),
    )"""

    def test_gan(self):
        """
        The generator is trained to match the likelihood of 8 Gaussian
        distributions
        """
        logging.info("GAN: 8 Gaussians")

        dim = 2
        noise_dim = 2
        batch_size = 256
        d_iters = 5
        par_vi = None
        net = Net(noise_dim, dim)
        critic = Net2(dim)

        trainset = Test8GaussiansDataSet(size=20000)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        generator = GenerativeAdversarialAlgorithm(
            TensorSpec(shape=(dim, )),
            net=net,
            critic=critic,
            grad_lambda=.10,
            critic_weight_clip=0.,
            critic_iter_num=d_iters,
            noise_dim=noise_dim,
            par_vi=par_vi,
            critic_optimizer=alf.optimizers.Adam(lr=1e-4, betas=(.5, .9)),
            optimizer=alf.optimizers.Adam(lr=1e-4, betas=(.5, .9)))

        generator.set_data_loader(train_loader, train_loader, None)
        logging.info("Generative Adversarial Network Test")

        def _train(i):
            alg_step = generator.train_iter(save_samples=False)

        for i in range(1000):
            _train(i)
            if i % 50 == 0:
                data = trainset.get_features()[:256]
                generate_image(generator, data, batch_size, i)


if __name__ == '__main__':
    alf.test.main()
