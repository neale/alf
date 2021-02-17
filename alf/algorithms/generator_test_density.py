# copyright (c) 2019 horizon robotics. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#      http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

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
from alf.utils import math_ops
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class MoG(torch.distributions.Distribution):
  def __init__(self, loc, covariance_matrix):
    self.num_components = loc.size(0)
    self.loc = loc
    self.covariance_matrix = covariance_matrix
    self.dists = [
      torch.distributions.MultivariateNormal(mu, covariance_matrix=sigma)
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
    return torch.cat(
      [p.log_prob(value).unsqueeze(-1) for p in self.dists], dim=-1).logsumexp(dim=-1)

  def enumerate_support(self):
    return self.dists[0].enumerate_support()


class MoG8(MoG):
    def __init__(self):
        scale = 6
        sq2 = 1/math.sqrt(2)
        loc = torch.tensor([[1,0], [-1,0], [0,1], [0,-1], [sq2,sq2],
                            [-sq2,sq2], [sq2,-sq2], [-sq2,-sq2]])
        loc = loc * scale
        cov = torch.Tensor([.5, .5]).diag().unsqueeze(0).repeat(8, 1, 1)
        super(MoG8, self).__init__(loc, cov)



def get_energy_function(name):
    w1 = lambda z: torch.sin(2 * math.pi * z[:,0] / 4)
    w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:,0] - 1)/0.6)**2)
    w3 = lambda z: 3 * torch.sigmoid((z[:,0] - 1) / 0.3)
             
    if name == 'twin_moons':
        target_dist = lambda z: 0.5 * ((torch.norm(
            z, p=2, dim=1) - 2) / 0.4)**2 - torch.log(torch.exp(
            -0.5*((z[:,0] - 2) / 0.6)**2) + torch.exp(-0.5*((
            z[:,0] + 2) / 0.6)**2) + 1e-10)
    elif name == 'sin':
        target_dist = lambda z: 0.5 * ((z[:,1] - w1(z)) / 0.4)**2
    elif name == 'sin_section':
        target_dist = lambda z: - torch.log(torch.exp(-0.5*((
            z[:,1] - w1(z))/0.35)**2) + torch.exp(
                -0.5*((z[:,1] - w1(z) + w2(z))/0.35)**2) + 1e-10)
    elif name == 'sin_split':
        target_dist = lambda z: - torch.log(torch.exp(-0.5*((
            z[:,1] - w1(z))/0.4)**2) + torch.exp(-0.5*((
                z[:,1] - w1(z) + w3(z))/0.35)**2) + 1e-10)
    elif name == 'mog8':
        target_dist = MoG8()
    else:
        raise ValueError("given target distribution potential not defined")
    return target_dist


class Net(Network):
    def __init__(self, dim=2, init_w=None):
        super().__init__(
            input_tensor_spec=TensorSpec(shape=(dim, )), name="Net")

        self.fc1 = nn.Linear(3, 1000, bias=True)
        self.fc2 = nn.Linear(1000, 1000, bias=True)
        self.fc3 = nn.Linear(1000, dim, bias=True)

    def forward(self, input, state=()):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, ()


class GeneratorTest(parameterized.TestCase, alf.test.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(torch.max(abs(x - y))), eps)

    #@parameterized.parameters(
        #dict(entropy_regularization=1.0, par_vi='svgd3', energy='twin_moons'),
        #dict(entropy_regularization=1.0, par_vi='svgd3', energy='sin'),
        #dict(entropy_regularization=1.0, par_vi='svgd3', energy='sin_section'),
        #dict(entropy_regularization=1.0, par_vi='svgd3', energy='sin_split'),
        #dict(entropy_regularization=1.0, par_vi='svgd3'),
        #dict(entropy_regularization=1.0, par_vi='gfsf'),
        #dict(entropy_regularization=1.0, par_vi='svgd3', functional_gradient='rkhs', energy='sin_section'),
        #dict(entropy_regularization=1.0, par_vi='svgd3', functional_gradient='rkhs', energy='twin_moons'),
        #dict(entropy_regularization=1.0, par_vi='svgd3', functional_gradient='rkhs', energy='sin'),
        #dict(entropy_regularization=1.0, par_vi='svgd3', functional_gradient='rkhs', energy='sin_split'),
        #dict(entropy_regularization=1.0, par_vi='minmax'),
        #dict(entropy_regularization=0.0, mi_weight=1),
    #)
    def test_generator_unconditional(self,
                                     entropy_regularization=1.0,
                                     par_vi='svgd3',
                                     functional_gradient='rkhs',
                                     energy='sin_split',
                                     mi_weight=None):
        r"""
        The generator is trained to match (STEIN) / maximize (ML) the likelihood
        of a Gaussian distribution with zero mean and diagonal variance :math:`(1, 4)`.
        After training, :math:`w^T w` is the variance of the distribution implied by the
        generator. So it should be :math:`diag(1,4)` for STEIN and 0 for 'ML'.
        """
        logging.info("entropy_regularization: %s par_vi: %s mi_weight: %s dataset: %s" %
                     (entropy_regularization, par_vi, mi_weight, energy))
        dim = 2
        batch_size = 100
        hidden_size = 2
        noise_dim = 3
        init_w = torch.randn(noise_dim, dim)
        if functional_gradient is not None:
            noise_dim = 2
            input_dim = TensorSpec((noise_dim, ))
            net = ReluMLP(input_dim, hidden_layers=(500,500), output_size=dim)
            critic_relu_mlp = True
        else:
            net = Net(dim, init_w)

        ll_fn = get_energy_function(energy)
        generator = Generator(
            dim,
            noise_dim=noise_dim,
            entropy_regularization=entropy_regularization,
            net=net,
            mi_weight=mi_weight,
            par_vi=par_vi,
            functional_gradient=functional_gradient,
            pinverse_hidden_size=500,
            critic_hidden_layers=(hidden_size, hidden_size),
            optimizer=alf.optimizers.AdamTF(lr=1e-4),
            pinverse_optimizer=alf.optimizers.AdamTF(lr=1e-4),
            critic_optimizer=alf.optimizers.AdamTF(lr=1e-3))
        if functional_gradient is not None:
            par_vi = par_vi+'_fg_500p_500hx2_b100_stdev4'
        else:
            par_vi = par_vi+'_1000x2_b100'
        
        def _neglogprob(x):
            if energy == 'mog8':
                y = -ll_fn.log_prob(x)
            else:
                y = ll_fn(x)
            return y

        def _train(i):
            alg_step = generator.train_step(
                inputs=None,
                loss_func=_neglogprob,
                batch_size=batch_size)
            generator.update_with_gradient(alg_step.info)

        for i in range(200001):
            _train(i)
            if i % 1000 == 0:
                x, _ = generator._predict(batch_size=batch_size, training=False)
                nll = _neglogprob(x).sum()
                print ('[{}] [iter {}], nll: {}'.format(par_vi, i, nll))
                
                x, _ = generator._predict(batch_size=20000, training=False)
                basedir = 'plots/density_outputs/{}/{}'.format(energy, par_vi)
                os.makedirs(basedir, exist_ok=True)
                if i % 10000 == 0:
                    torch.save(x.detach().cpu(), basedir+'/outputs_{}.pt'.format(i))
                
                x = x.detach().cpu().numpy()
                
                #f, ax = plt.subplots(1, 2)
                #ax[0].hist2d(x[:, 0], x[:, 1], bins=4*50, cmap=plt.cm.jet)
                
                plt.hist2d(x[:, 0], x[:, 1], bins=4*50, cmap=plt.cm.jet)
                
                #x = torch.linspace(-7, 7, 200)
                #xx, yy = torch.meshgrid((x, x))
                #zz = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze()
                #ax[1].pcolormesh(xx.cpu(), yy.cpu(),
                #    torch.exp(ll_fn.log_prob(zz)).view(200,200).data.cpu(), cmap=plt.cm.jet)
                    #torch.exp(-ll_fn.log_prob(zz)).view(200,200).data.cpu(), cmap=plt.cm.jet)

                plt.tight_layout()
                plt.savefig(basedir+'/output_{}'.format(i))
                plt.close('all')

        x, _ = generator._predict(batch_size=batch_size, training=False)
        nll = _neglogprob(x).sum()
        print ('nll', nll)

if __name__ == '__main__':
    alf.test.main()

