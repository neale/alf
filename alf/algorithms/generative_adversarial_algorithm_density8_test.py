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
from alf.utils.datagen import Test25GaussiansDataSet

from torch.cuda.amp import autocast, GradScaler

import os
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class Net(Network):
    def __init__(self, noise_dim, output_dim, hidden_size):
        super().__init__(
            input_tensor_spec=TensorSpec(
                shape=(noise_dim, ), dtype=torch.float32),
            name="Net")
        self.bn = True
        self.fc1 = nn.Linear(noise_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)

    def forward(self, input, state=()):
        if self.bn:
            x = F.relu(self.bn1(self.fc1(input)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
            #x = F.relu(self.bn4(self.fc4(x)))
        else:
            x = F.relu(self.fc1(input))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        x = self.fc5(x)
        return x, state


class Net2(Network):
    def __init__(self, input_dim, hidden_size, metric, use_sn):
        super().__init__(
            input_tensor_spec=TensorSpec(
                shape=(input_dim, ), dtype=torch.float32),
            name="Net2")
        self.metric = metric
        if use_sn:
            sn = nn.utils.spectral_norm
        else:
            sn = lambda x: x

        self.fc1 = sn(nn.Linear(input_dim, hidden_size))
        self.fc2 = sn(nn.Linear(hidden_size, hidden_size))
        self.fc3 = sn(nn.Linear(hidden_size, hidden_size))
        self.fc4 = sn(nn.Linear(hidden_size, 1))

    def forward(self, input, state=()):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x.view(-1), state


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def generate_image(generator,
                   noise_dim,
                   data,
                   step_num,
                   modes,
                   path,
                   final=True):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    if modes == 8:
        data_range = 3
    elif modes == 25:
        data_range = 9

    num_points = 1000
    points = np.zeros((num_points, num_points, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-data_range, data_range, num_points)[:, None]
    points[:, :, 1] = np.linspace(-data_range, data_range, num_points)[None, :]
    points = points.reshape((-1, 2))
    points = torch.tensor(points).to(alf.get_default_device())
    critic_map = generator.critic(points)[0].cpu().numpy()

    noise = torch.randn(num_points, noise_dim).to(alf.get_default_device())
    samples = generator._generator._net(noise)[0]  #, data)[0]

    x = y = np.linspace(-data_range, data_range, num_points)
    plt.contour(x, y, critic_map.reshape((len(x), len(y))).transpose())

    data = data.cpu().numpy()
    samples = samples.cpu().numpy()

    plt.scatter(data[:, 0], data[:, 1], c='orange', marker='+')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')
    save_dir = '/nfs/hpc/share/ratzlafn/alf-plots/gan/{}Gaussians'.format(
        modes)
    save_dir = save_dir + path
    os.makedirs(save_dir, exist_ok=True)
    print('saving step {} to {}'.format(step_num, save_dir))
    plt.savefig('{}/gen_{}.png'.format(save_dir, step_num))
    plt.close('all')

    # plot density
    np.save('{}/figure_samples_{}.npy'.format(save_dir, step_num), samples)
    data_samples = []
    for i in range(100):
        samples, _ = generator._generator._predict(
            batch_size=100, training=False)
        if isinstance(samples, tuple):
            _, samples = samples
        data_samples.append(samples)
    samples = torch.cat(data_samples, dim=0)
    samples = samples.cpu().numpy()
    if final == True:
        for i in range(100):
            final_samples, _ = generator._generator._predict(
                batch_size=100, training=False)
            if isinstance(final_samples, tuple):
                final_samples, _ = final_samples
            data_samples.append(final_samples)
        samples = torch.cat(data_samples, dim=0)
        samples = samples.cpu().numpy()
        np.save('{}/final_samples_{}.npy'.format(save_dir, step_num), samples)


class GenerativeAdversarialTest(parameterized.TestCase, alf.test.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(torch.max(abs(x - y))), eps)

    @parameterized.parameters(

        #dict(par_vi=None,
        #    functional_gradient=None,
        #    entropy_regularization=0.0,
        #    use_sn=True,
        #    metric='kl-w1'),

        #dict(par_vi=None,
        #    functional_gradient=None,
        #    entropy_regularization=0.0,
        #    metric='w1',
        #    grad_lambda=0.1),

        #dict(par_vi=None,
        #    functional_gradient=None,
        #    entropy_regularization=0.0,
        #    metric='jsd'),

        #dict(par_vi='svgd',
        #    functional_gradient='rkhs',
        #    entropy_regularization=.1,
        #    diag=.01,
        #    p_iters=1,
        #    p_hidden=100,
        #    metric='jsd'),

        #dict(par_vi='svgd',
        #    functional_gradient='rkhs',
        #    entropy_regularization=0.001,
        #    diag=.1,
        #    p_iters=1,
        #    p_hidden=256,
        #    use_sn=True,
        #    metric='kl-w1'),
        dict(
            par_vi='svgd',
            functional_gradient='rkhs',
            entropy_regularization=100 / 1e4,
            diag=.1,
            p_iters=1,
            p_hidden=256,
            metric='jsd'), )
    def test_gan(self,
                 par_vi='svgd',
                 functional_gradient='rkhs',
                 entropy_regularization=0.1,
                 num_modes=8,
                 metric='kl-w1',
                 grad_lambda=0.0,
                 use_sn=False,
                 diag=1.0,
                 p_iters=5,
                 p_hidden=5,
                 batch_size=100):
        """
        The generator is trained to match the likelihood of 8 Gaussian
        distributions
        """
        logging.info("{}-GAN: {} Gaussians".format(metric, num_modes))
        logging.info('ParVI: {}, functional gradient: {}'.format(
            par_vi, functional_gradient))
        if par_vi == None:
            if entropy_regularization != 0.:
                entropy_regularization = 0.
        else:
            if entropy_regularization == 0.:
                entropy_regularization = 1.0  # batch_size / 1e3

        dim = 2
        noise_dim = 2
        d_iters = 5

        metric = metric
        grad_lambda = grad_lambda  #1.0 # 10
        use_sn = use_sn

        g_hidden_size = 256
        d_hidden_size = 256
        glr = 1e-4
        dlr = 1e-4
        plr = 1e-4

        pinverse_hidden_size = p_hidden
        pinverse_solve_iters = p_iters
        fullrank_diag_weight = diag

        net = Net(noise_dim, dim, hidden_size=g_hidden_size)
        critic = Net2(
            dim, hidden_size=d_hidden_size, metric=metric, use_sn=use_sn)
        net.apply(weights_init)
        critic.apply(weights_init)

        std = 0.3
        if num_modes == 8:
            trainset = Test8GaussiansDataSet(size=10000, data_std=std)
        elif num_modes == 25:
            trainset = Test25GaussiansDataSet(size=10000, data_std=std)
        else:
            raise NotImplementedError

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        generator = GenerativeAdversarialAlgorithm(
            output_dim=dim,
            input_tensor_spec=TensorSpec(shape=(dim, )),
            net=net,
            critic=critic,
            grad_lambda=grad_lambda,
            metric=metric,
            critic_weight_clip=0.,
            critic_iter_num=d_iters,
            noise_dim=noise_dim,
            par_vi=par_vi,
            functional_gradient=functional_gradient,
            entropy_regularization=entropy_regularization,
            block_pinverse=True,
            force_fullrank=True,
            jac_autograd=True,
            scaler=GradScaler(enabled=False),
            expectation_logp=False,
            use_kernel_logp=False,
            fullrank_diag_weight=fullrank_diag_weight,
            pinverse_hidden_size=pinverse_hidden_size,
            pinverse_solve_iters=pinverse_solve_iters,
            pinverse_optimizer=alf.optimizers.Adam(lr=plr),
            critic_optimizer=torch.optim.Adam(
                critic.parameters(), lr=dlr, betas=(.5, .99)),
            optimizer=alf.optimizers.Adam(lr=glr, betas=(.5, .99)),
            logging_training=True)

        generator.set_data_loader(train_loader, train_loader, None)
        logging.info("Generative Adversarial Network Test")

        def _train(i):
            alg_step = generator.train_iter(save_samples=False)
            if num_modes == 8:
                trainset = Test8GaussiansDataSet(size=10000, data_std=std)
            elif num_modes == 25:
                trainset = Test25GaussiansDataSet(size=10000, data_std=std)
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True, num_workers=0)
            generator.set_data_loader(train_loader, train_loader, None)

        if functional_gradient is not None:
            path = f'/gpvi-{metric}_diag{fullrank_diag_weight}'
            path += f'-plr{plr}-inv{pinverse_hidden_size}-iters{pinverse_solve_iters}'
        elif par_vi is not None:
            path = f'/{par_vi}-{metric}'
        elif par_vi == None:
            path = f'/{metric}-gan'
        path += f'-dlr{dlr}_glr{glr}_(g{g_hidden_size}, d{d_hidden_size})'
        path += f'-gp{grad_lambda}-ent{entropy_regularization}'
        path += f'-datastd{std}_bs{batch_size}_logp'

        for i in range(10000):
            _train(i)
            if i % 100 == 0:
                data = trainset.get_features()[:1000]
                with torch.no_grad():
                    generate_image(generator, noise_dim, data, i, num_modes,
                                   path)

        data = trainset.get_features()
        with torch.no_grad():
            generate_image(
                generator, noise_dim, data, i, num_modes, path, final=True)


if __name__ == '__main__':
    alf.test.main()
