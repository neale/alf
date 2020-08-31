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
import torch
import torch.nn as nn
import torch.nn.functional as F

import alf
from alf.algorithms.generator import Generator
from alf.networks import Network
from alf.tensor_specs import TensorSpec

from alf.trainers.policy_trainer import create_dataset

class Net(Network):
    def __init__(self, noise_dim, output_dim):
        super().__init__(
            input_tensor_spec=TensorSpec(shape=(dim, )), name="Net")

        self.fc1 = nn.Linear(noise_dim, 256, bias=False)
        self.fc2 = nn.Linear(256, 512, bias=False)
        self.fc3 = nn.Linear(512, output_dim, bias=False)

    def forward(self, input, state=()):
        x = torch.relu_(self.fc1(input))
        x = torch.relu_(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x, ()


class ImageGeneratorTest(parameterized.TestCase, alf.test.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(torch.max(abs(x - y))), eps)

    @parameterized.parameters(
        dict(par_vi='minmax'),
        dict(par_vi='gfsf'),
        dict(par_vi='svgd'),
    )
    def test_image_generator_mnist(self, par_vi=None)
        """
        The generator is trained to match(STEIN)/maximize(ML) the likelihood
        of a Gaussian distribution with zero mean and diagonal variance :math:`(1, 4)`.
        After training, :math:`w^T w` is the variance of the distribution implied by the
        generator. So it should be :math:`diag(1,4)` for STEIN and 0 for 'ML'.
        """
        logging.info("par_vi: %s dataset: %s" %
                     (par_vi, dataset))

        trainset, _ = create_dataset()
        dim = 784
        noise_dim = 256
        batch_size = 100
        d_iters = 5

        net = Net(noise_dim, dim)
        generator = Generator(
            dim,
            noise_dim=256,
            net=net,
            par_vi=par_vi,
            optimizer=alf.optimizers.AdamTF(lr=1e-3))

        var = torch.tensor([1, 4], dtype=torch.float32)
        precision = 1. / var

        def _neglogprob(x):
            return torch.squeeze(
                0.5 * torch.matmul(x * x, torch.reshape(precision, (dim, 1))),
                axis=-1)

        def _train(i):
            for batch_idx, (data, _) in enumerate(trainset):
                if par_vi == 'minmax':
                    if i % (d_iters+1):
                        model = 'critic'
                    else:
                        model = 'generator'
                else:
                    model = None
                alg_step = generator.train_step(
                    inputs=None,
                    loss_func=_neglogprob,
                    batch_size=batch_size,
                    model=model)
                generator.update_with_gradient(alg_step.info)
                generator.after_update(alg_step.info)

        for i in range(6000):
            _train(i)
            learned_var = torch.matmul(net.fc.weight, net.fc.weight.t())
            if i % 500 == 0:
                print(i, "learned var=", learned_var)

        if entropy_regularization == 1.0:
            self.assertArrayEqual(torch.diag(var), learned_var, 0.1)
        else:
            if mi_weight is None:
                self.assertArrayEqual(torch.zeros(dim, dim), learned_var, 0.1)
            else:
                self.assertGreater(
                    float(torch.sum(torch.abs(learned_var))), 0.5)

    @parameterized.parameters(
        dict(entropy_regularization=1.0),
        dict(entropy_regularization=0.0),
        dict(entropy_regularization=0.0, mi_weight=1),
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
