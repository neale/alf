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

import alf
from alf.algorithms.generator import Generator
from alf.networks import Network
from alf.networks.relu_mlp import ReluMLP
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops


class Net(Network):
    def __init__(self, dim=2, init_w=None):
        super().__init__(
            input_tensor_spec=TensorSpec(shape=(dim, )), name="Net")

        self.fc = nn.Linear(3, dim, bias=False)
        if init_w is None:
            w = torch.randn(3, 2, dtype=torch.float32)
        else:
            w = init_w
        #w = torch.tensor([[1, 2], [-1, 1], [1, 1]], dtype=torch.float32)
        self.fc.weight = nn.Parameter(w.t())

    def forward(self, input, state=()):
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

    #@parameterized.parameters(
        #dict(entropy_regularization=1.0, par_vi='svgd'),
        #dict(entropy_regularization=1.0, par_vi='svgd2'),
        #dict(entropy_regularization=1.0, par_vi='svgd3'),
        #dict(entropy_regularization=1.0, par_vi='gfsf'),
        #dict(entropy_regularization=1.0, par_vi='svgd3', functional_gradient='rkhs'),
        #dict(entropy_regularization=1.0, par_vi='minmax'),
        #dict(entropy_regularization=0.0),
        #dict(entropy_regularization=0.0, mi_weight=1),
    #)
    def test_generator_unconditional(self,
                                     entropy_regularization=1.0,
                                     par_vi='svgd3',
                                     functional_gradient=None,
                                     mi_weight=None):
        r"""
        The generator is trained to match (STEIN) / maximize (ML) the likelihood
        of a Gaussian distribution with zero mean and diagonal variance :math:`(1, 4)`.
        After training, :math:`w^T w` is the variance of the distribution implied by the
        generator. So it should be :math:`diag(1,4)` for STEIN and 0 for 'ML'.
        """
        logging.info("entropy_regularization: %s par_vi: %s mi_weight: %s" %
                     (entropy_regularization, par_vi, mi_weight))
        dim = 5
        batch_size = 512
        hidden_size = 5
        noise_dim = 5
        init_w = torch.randn(noise_dim, dim)
        init_w = torch.randn(noise_dim, dim)
        if functional_gradient is not None:
            noise_dim = 5
            input_dim = TensorSpec((noise_dim, ))
            net = ReluMLP(input_dim, hidden_layers=(), output_size=dim)
            net._fc_layers[0].weight.data = torch.tensor(torch.randn(dim, dim), dtype=torch.float32).T
                #[[1, 2], [1, 1]],
            critic_relu_mlp = True
        else:
            net = Net(dim, init_w)
        print (par_vi)
        generator = Generator(
            dim,
            noise_dim=noise_dim,
            entropy_regularization=entropy_regularization,
            net=net,
            mi_weight=mi_weight,
            par_vi=par_vi,
            functional_gradient=functional_gradient,
            pinverse_hidden_size=300,
            critic_hidden_layers=(hidden_size, hidden_size),
            optimizer=alf.optimizers.AdamTF(lr=1e-3),
            critic_optimizer=alf.optimizers.AdamTF(lr=1e-3))
        
        #var = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        var = torch.rand(dim, dim).float()
        var = torch.mm(var, var.t())
        precision = torch.inverse(var)
        print ('true var', var)

        def _neglogprob(x):
            y = 0.5 * torch.einsum('bi,ij,bj->b', x, precision, x)
            return y

        def _train(i):
            alg_step = generator.train_step(
                inputs=None,
                loss_func=_neglogprob,
                batch_size=batch_size)
            generator.update_with_gradient(alg_step.info)

        for i in range(20000):
            _train(i)
            if functional_gradient is not None:
                learned_var = torch.matmul(
                    net._fc_layers[0].weight,
                    net._fc_layers[0].weight.t())
                
            else:
                learned_var = torch.matmul(net.fc.weight, net.fc.weight.t())
            if i % 500 == 0:
                print(i, "learned var=", learned_var)
        if entropy_regularization == 1.0:
            #self.assertArrayEqual(torch.diag(var), learned_var, 0.1)
            print(par_vi, ': ', float(torch.max(abs(var - learned_var))))
            self.assertArrayEqual(var, learned_var, 0.1)
        else:
            if mi_weight is None:
                self.assertArrayEqual(torch.zeros(dim, dim), learned_var, 0.1)
            else:
                self.assertGreater(
                    float(torch.sum(torch.abs(learned_var))), 0.5)

    @parameterized.parameters(
        dict(entropy_regularization=1.0),
        #dict(entropy_regularization=0.0),
        #dict(entropy_regularization=0.0, mi_weight=1),
    )
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
        batch_size = 512
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

        for i in range(5000):
            _train()
            learned_var = torch.matmul(net.fc1.weight, net.fc1.weight.t())
            if i % 500 == 0:
                print(i, "learned var=", learned_var)
                print("u=", net.fc2.weight.t())

        if mi_weight is not None:
            self.assertGreater(float(torch.sum(torch.abs(learned_var))), 0.5)
        elif entropy_regularization == 1.0:
            self.assertArrayEqual(net.fc2.weight.t(), u, 0.1)
            self.assertArrayEqual(torch.diag(var), learned_var, 0.1)
        else:
            self.assertArrayEqual(net.fc2.weight.t(), u, 0.1)
            self.assertArrayEqual(torch.zeros(dim, dim), learned_var, 0.1)

if __name__ == '__main__':
    alf.test.main()
