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

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.generator2 import Generator
from alf.networks import Network, ReluMLP
from alf.tensor_specs import TensorSpec
from alf.utils.math_ops import identity
from alf.utils import common


class Net(Network):
    def __init__(self, noise_dim=3, dim=2, hidden_size=3):
        super().__init__(
            input_tensor_spec=TensorSpec(shape=(dim, )), name="Net")

        #self.fc = nn.Linear(noise_dim, noise_dim, bias=False)
        #self.fc2 = nn.Linear(noise_dim, dim, bias=False)
        self.fc = nn.Linear(noise_dim, dim, bias=False)

    def forward(self, input, state=()):
        #return self.fc2(self.fc(input)), ()
        return self.fc(input), ()


class Net2(Network):
    def __init__(self, dim=2):
        super().__init__(
            input_tensor_spec=[
                TensorSpec(shape=(dim, )),
                TensorSpec(shape=(dim, ))
            ],
            name="Net")
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        w = torch.tensor([[1, 2], [1, 1]], dtype=torch.float32)
        u = torch.zeros((dim, dim), dtype=torch.float32)
        self.fc1.weight = nn.Parameter(w.t())
        self.fc2.weight = nn.Parameter(u.t())

    def forward(self, input, state=()):
        return self.fc1(input[0]) + self.fc2(input[1]), ()


class GeneratorTest(parameterized.TestCase, alf.test.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(torch.max(abs(x - y))), eps)

    """
    @parameterized.parameters(
        #dict(entropy_regularization=1.0, par_vi='gfsf'),
        #dict(entropy_regularization=1.0, par_vi='svgd'),
        #dict(entropy_regularization=1.0, par_vi='svgd2'),
        dict(entropy_regularization=1.0, par_vi='svgd3'),
        #dict(entropy_regularization=1.0, par_vi='minmax'),
        dict(
            entropy_regularization=1.0,
            par_vi='svgd',
            functional_gradient='rkhs'),
        #dict(
        #    entropy_regularization=1.0,
        #    par_vi='minmax',
        #    functional_gradient='minmax'),
        #dict(entropy_regularization=0.0),
        #dict(entropy_regularization=0.0, mi_weight=1),
    )"""

    def test_generator_unconditional(self,
                                     entropy_regularization=1.0,
                                     par_vi='svgd',
                                     functional_gradient='rkhs',
                                     mi_weight=None):
        r"""
        The generator is trained to match (STEIN) / maximize (ML) the likelihood
        of a Gaussian distribution with zero mean and diagonal variance :math:`(1, 4)`.
        After training, :math:`w^T w` is the variance of the distribution implied by the
        generator. So it should be :math:`diag(1,4)` for STEIN and 0 for 'ML'.
        """
        logging.info("entropy_regularization: %s par_vi: %s mi_weight: %s" %
                     (entropy_regularization, par_vi, mi_weight))
        dim = 4
        noise_dim = 3
        batch_size = 64
        hidden_size = 10
        block_pinverse = True
        jac_autograd = False
        if functional_gradient is not None:
            input_dim = TensorSpec((noise_dim, ))
            if block_pinverse and not jac_autograd:
                head_size = (noise_dim, dim - noise_dim)
            else:
                head_size = None
            net = ReluMLP(
                input_dim,
                hidden_layers=(),  #(dim,),
                head_size=head_size,
                activation=identity,
                output_size=dim)
            #net = Net(noise_dim, dim, hidden_size)
        else:
            net = Net(noise_dim, dim, hidden_size)
        common.set_random_seed(0)
        generator = Generator(
            dim,
            noise_dim=noise_dim,
            entropy_regularization=entropy_regularization,
            net=net,
            mi_weight=mi_weight,
            par_vi=par_vi,
            functional_gradient=functional_gradient,
            block_pinverse=block_pinverse,
            force_fullrank=True,
            pinverse_hidden_size=10,
            fullrank_diag_weight=1.0,
            exact_jac_inverse=False,
            jac_autograd=jac_autograd,
            critic_hidden_layers=(hidden_size, hidden_size),
            pinverse_solve_iters=2,
            optimizer=alf.optimizers.Adam(lr=1e-2, name='generator'),
            pinverse_optimizer=alf.optimizers.Adam(lr=1e-3, name='pinverse'),
            critic_optimizer=alf.optimizers.Adam(lr=2e-3))

        var = torch.rand(dim, dim).float()
        var = torch.mm(var, var.t())
        precision = torch.inverse(var)
        print('True Var:', var)

        def _neglogprob(x):
            y = 0.5 * torch.einsum('bi,ij,bj->b', x, precision, x)
            return y

        def _train():
            alg_step = generator.train_step(
                inputs=None, loss_func=_neglogprob, batch_size=batch_size)
            generator.update_with_gradient(alg_step.info)
            return alg_step.info.extra

        for i in range(20000):
            step = _train()
            if functional_gradient is not None:
                #learned_var = torch.matmul(net.fc.weight, net.fc.weight.t())
                #learned_var = torch.matmul(net._fc_layers[0].weight,
                #                           net._fc_layers[0].weight.t())
                #learned_var1 = torch.matmul(net._fc_layers[0].weight,
                #                            net._fc_layers[0].weight.t())
                if block_pinverse == True and jac_autograd == False:
                    weight = torch.cat(
                        (net._heads[0].weight, net._heads[1].weight), dim=0)
                else:
                    try:
                        weight = net._fc_layers[0].weight
                    except:
                        weight = net.fc.weight
                learned_var = torch.matmul(weight, weight.t())
                #learned_var = torch.matmul(torch.matmul(weight, learned_var1), weight.t())
            else:
                learned_var = torch.matmul(net.fc.weight, net.fc.weight.t())
                #learned_var1 = torch.matmul(net.fc.weight, net.fc.weight.t())
                #learned_var = torch.matmul(
                #    torch.matmul(net.fc2.weight, learned_var1), net.fc2.weight.t())
            if i % 500 == 0:
                if functional_gradient is not None:
                    print("pinverse loss", step.pinverse.mean())
                print(i, "learned var=", learned_var)
                print('[{}] avg per dim variance error: {}'.format(
                    i, (var - learned_var).mean()))
                print('[{}] max dim error: {}'.format(
                    i, (float(torch.max(abs(var - learned_var))))))
        print('[{}] avg per dim variance error: {}'.format(
            i, (var - learned_var).mean()))
        if entropy_regularization == 1.0:
            self.assertArrayEqual(var, learned_var, 0.2)
        else:
            if mi_weight is None:
                self.assertArrayEqual(torch.zeros(dim, dim), learned_var, 0.2)
            else:
                self.assertGreater(
                    float(torch.sum(torch.abs(learned_var))), 0.5)

    """
    @parameterized.parameters(
        dict(entropy_regularization=1.0),
        dict(entropy_regularization=0.0),
        dict(entropy_regularization=0.0, mi_weight=1),
    )"""

    def test_generator_conditional(self,
                                   entropy_regularization=0.0,
                                   par_vi='svgd',
                                   mi_weight=None):
        r"""
        The target conditional distribution is :math:`N(\mu; diag(1, 4))`. After training
        net._u should be u for both STEIN and ML. And :math:`w^T w` should be :math:`diag(1, 4)`
        for STEIN and 0 for ML.
        """

        logging.info("entropy_regularization: %s mi_weight: %s" %
                     (entropy_regularization, mi_weight))
        dim = 2
        batch_size = 128
        net = Net2(dim)
        generator = Generator(
            dim,
            noise_dim=dim,
            entropy_regularization=entropy_regularization,
            net=net,
            mi_weight=mi_weight,
            par_vi=par_vi,
            input_tensor_spec=TensorSpec((dim, )),
            optimizer=alf.optimizers.Adam(lr=1e-3))

        var = torch.tensor([1, 4], dtype=torch.float32)
        precision = 1. / var
        u = torch.tensor([[-0.3, 1], [1, 2]], dtype=torch.float32)

        def _neglogprob(xy):
            x, y = xy
            d = x - torch.matmul(y, u)
            return torch.squeeze(
                0.5 * torch.matmul(d * d, torch.reshape(precision, (dim, 1))),
                axis=-1)

        def _train():
            y = torch.randn(batch_size, dim)
            alg_step = generator.train_step(inputs=y, loss_func=_neglogprob)
            generator.update_with_gradient(alg_step.info)

        for i in range(0):
            _train()
            learned_var = torch.matmul(net.fc1.weight, net.fc1.weight.t())
            if i % 500 == 0:
                print(i, "learned var=", learned_var)
                print("u=", net.fc2.weight.t())

        if mi_weight is not None:
            self.assertGreater(float(torch.sum(torch.abs(learned_var))), 0.5)
        elif entropy_regularization == 1.0:
            self.assertArrayEqual(net.fc2.weight.t(), u, 0.2)
            self.assertArrayEqual(torch.diag(var), learned_var, 0.2)
        else:
            self.assertArrayEqual(net.fc2.weight.t(), u, 0.2)
            self.assertArrayEqual(torch.zeros(dim, dim), learned_var, 0.2)


if __name__ == '__main__':
    alf.test.main()
