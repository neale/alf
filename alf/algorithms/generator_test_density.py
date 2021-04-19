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

import math

from absl import logging
from absl.testing import parameterized
import torch
import torch.nn as nn
import torch.nn.functional as F

import alf
from alf.algorithms.generator import Generator
from alf.networks import Network
from alf.networks.relu_mlp import ReluMLP
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops, common

import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class MoG(torch.distributions.Distribution):
    def __init__(self, loc, covariance_matrix):
        self.num_components = loc.size(0)
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        self.dists = [
            torch.distributions.MultivariateNormal(
                mu, covariance_matrix=sigma)
            for mu, sigma in zip(loc, covariance_matrix)
        ]

        super(MoG, self).__init__(torch.Size([]), torch.Size([loc.size(-1)]))

    @property
    def arg_constraints(self):
        return self.dists[0].arg_constraints

    @property
    def support(self):
        return self.dists[0].support

    @property
    def has_rsample(self):
        return False

    def log_prob(self, value):
        return torch.cat([p.log_prob(value).unsqueeze(-1) for p in self.dists],
                         dim=-1).logsumexp(dim=-1)

    def enumerate_support(self):
        return self.dists[0].enumerate_support()


class MoG8(MoG):
    def __init__(self):
        scale = 6
        sq2 = 1 / math.sqrt(2)
        loc = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1], [sq2, sq2],
                            [-sq2, sq2], [sq2, -sq2], [-sq2, -sq2]])
        loc = loc * scale
        cov = torch.Tensor([.5, .5]).diag().unsqueeze(0).repeat(8, 1, 1)
        super(MoG8, self).__init__(loc, cov)


def get_energy_function(name):
    w1 = lambda z: torch.sin(2 * math.pi * z[:, 0] / 4)
    w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6)**2)
    w3 = lambda z: 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)

    if name == 'twin_moons':
        target_dist = lambda z: 0.5 * ((torch.norm(
            z, p=2, dim=1) - 2) / 0.4)**2 - torch.log(
                torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6)**2) + torch.exp(
                    -0.5 * ((z[:, 0] + 2) / 0.6)**2) + 1e-10)
    elif name == 'sin':
        target_dist = lambda z: 0.5 * ((z[:, 1] - w1(z)) / 0.4)**2
    elif name == 'sin_section':
        target_dist = lambda z: -torch.log(
            torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35)**2) + torch.exp(
                -0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35)**2) + 1e-10)
    elif name == 'sin_split':
        target_dist = lambda z: -torch.log(
            torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4)**2) + torch.exp(
                -0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35)**2) + 1e-10)
    elif name == 'mog8':
        target_dist = MoG8()
    else:
        raise ValueError("given target distribution potential not defined")
    return target_dist


class Net(Network):
    def __init__(self, z_dim, output_dim, hidden_size):
        super().__init__(
            input_tensor_spec=TensorSpec(shape=(z_dim, )), name="Generator")

        self.fc1 = nn.Linear(z_dim, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc4 = nn.Linear(hidden_size, output_dim, bias=True)

    def forward(self, input, state=()):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, state


class GeneratorTest(parameterized.TestCase, alf.test.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(torch.max(abs(x - y))), eps)

    """
    @parameterized.parameters(
        dict(
            entropy_regularization=1.0,
            par_vi='svgd3',
            functional_gradient=None,
            energy='twin_moons',
            input_noise_stdev=6.0,
            hidden_size=64,
            batch_size=128),
        dict(entropy_regularization=1.0, par_vi='svgd3', energy='sin'),
        dict(entropy_regularization=1.0, par_vi='svgd3', energy='sin_section'),
        dict(entropy_regularization=1.0, par_vi='svgd3', energy='sin_split'),
        dict(entropy_regularization=1.0, par_vi='svgd3'),
        dict(entropy_regularization=1.0, par_vi='gfsf'),
        dict(
            entropy_regularization=1.0,
            par_vi='svgd3',
            functional_gradient='rkhs',
            energy='sin_split'),
        dict(entropy_regularization=1.0, par_vi='minmax'),
        dict(entropy_regularization=0.0, mi_weight=1),
    )"""

    def test_generator_unconditional(
            self,
            entropy_regularization=1.0,
            par_vi='svgd3',
            functional_gradient='rkhs',
            energy='twin_moons',
            batch_size=128,
            input_noise_stdev=2.0,
            n_layers=6,
            hidden_size=256,
            pinverse_hidden_size=64,
            mi_weight=None,
    ):
        r"""
        The generator is trained to match (STEIN) / maximize (ML) the likelihood
        of a Gaussian distribution with zero mean and diagonal variance :math:`(1, 4)`.
        After training, :math:`w^T w` is the variance of the distribution implied by the
        generator. So it should be :math:`diag(1,4)` for STEIN and 0 for 'ML'.
        """
        logging.info(
            "entropy_regularization: %s par_vi: %s mi_weight: %s dataset: %s" %
            (entropy_regularization, par_vi, mi_weight, energy))
        common.set_random_seed(None)
        dim = 2
        noise_dim = 1
        if functional_gradient is not None:
            input_dim = TensorSpec((noise_dim, ))
            hidden_sizes = [hidden_size] * n_layers
            net = ReluMLP(
                input_dim, hidden_layers=hidden_sizes, output_size=dim)
        else:
            net = Net(noise_dim, dim, hidden_size)

        print(net._fc_layers[0].weight.mean(), net._fc_layers[1].weight.mean(),
              net._fc_layers[2].weight.mean())
        ll_fn = get_energy_function(energy)
        fullrank_diag_weight = 1.
        generator = Generator(
            dim,
            noise_dim=noise_dim,
            entropy_regularization=entropy_regularization,
            net=net,
            mi_weight=mi_weight,
            par_vi=par_vi,
            functional_gradient=functional_gradient,
            exact_jac_inverse=False,
            pinverse_hidden_size=pinverse_hidden_size,
            force_fullrank=True,
            block_pinverse=True,
            jac_autograd=True,
            fullrank_diag_weight=fullrank_diag_weight,
            input_noise_stdev=input_noise_stdev,
            critic_hidden_layers=(hidden_size, hidden_size),
            optimizer=alf.optimizers.Adam(lr=1e-4),  #, weight_decay=1e-4),
            pinverse_optimizer=alf.optimizers.Adam(lr=1e-4),
            critic_optimizer=alf.optimizers.Adam(lr=1e-3))

        if functional_gradient is not None:
            par_vi = par_vi + '_fg_{}ps_{}hs_{}bs_stdev{}_l{}_lamb{}_lr4'.format(
                pinverse_hidden_size, hidden_size, batch_size,
                input_noise_stdev, n_layers, fullrank_diag_weight)
        else:
            par_vi = par_vi + '_{}hs_{}bs_stdev{}_l{}'.format(
                hidden_size, batch_size, input_noise_stdev, n_layers)

        def _neglogprob(x):
            if energy == 'mog8':
                y = -ll_fn.log_prob(x)
            else:
                y = ll_fn(x)
            return y

        def _train(i):
            alg_step = generator.train_step(
                inputs=None, loss_func=_neglogprob, batch_size=batch_size)
            generator.update_with_gradient(alg_step.info)
            p_loss = alg_step.info.extra.pinverse
            if i % 500 == 0:
                print(p_loss)

        for i in range(30000):
            _train(i)
            if i % 1000 == 0:
                x, _ = generator._predict(
                    batch_size=batch_size, training=False)
                if isinstance(x, tuple):
                    x, _ = x
                print('mean', x.mean(), 'var', x.var())
                nll = _neglogprob(x).sum()
                print('[{}] [iter {}], nll: {}'.format(par_vi, i, nll))
                x, _ = generator._predict(batch_size=20000, training=False)
                if isinstance(x, tuple):
                    x, _ = x
                basedir = '/nfs/hpc/share/ratzlafn/alf-plots/density_outputs/{}/{}'.format(
                    energy, par_vi)
                os.makedirs(basedir, exist_ok=True)
                #if i % 10000 == 0:
                #    torch.save(x.detach().cpu(), basedir+'/outputs_{}.pt'.format(i))
                x = x.detach().cpu().numpy()
                plt.hist2d(x[:, 0], x[:, 1], bins=4 * 50, cmap=plt.cm.jet)
                plt.tight_layout()
                plt.savefig(basedir + '/output_{}'.format(i))
                plt.close('all')

        x, _ = generator._predict(batch_size=batch_size, training=False)
        if isinstance(x, tuple):
            x, _ = x
        nll = _neglogprob(x).sum()
        print('nll', nll)


if __name__ == '__main__':
    alf.test.main()
