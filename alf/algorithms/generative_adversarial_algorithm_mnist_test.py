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
import os

from absl import logging
from absl.testing import parameterized
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import alf
from alf.algorithms.generator import Generator
from alf.algorithms.generative_adversarial_algorithm import GenerativeAdversarialAlgorithm
from alf.networks import Network
from alf.layers import FC
from alf.tensor_specs import TensorSpec
from alf.utils.datagen import load_mnist1k, load_mnist
from alf.utils.sl_utils import classification_loss, predict_dataset
# torch.backends.cudnn.benchmark = True
from torch.cuda.amp import autocast


class Generator(Network):
    def __init__(self, dim, noise_dim, metric, flat):
        super().__init__(
            input_tensor_spec=TensorSpec(
                shape=(noise_dim, ), dtype=torch.float32),
            name="Generator")

        self.dim = dim
        self.flat = flat
        self.metric = metric

        if metric == 'jsd':
            self.conv1 = nn.ConvTranspose2d(
                noise_dim, 4 * dim, 4, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(4 * dim)
            self.conv2 = nn.ConvTranspose2d(
                4 * dim, 2 * dim, 3, 2, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(2 * dim)
            self.conv3 = nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(dim)
        elif metric == 'wasserstein':
            self.fc1 = nn.Linear(noise_dim, 8 * 2 * 2 * dim)
            self.bn0 = nn.BatchNorm1d(8 * 2 * 2 * dim)
            self.conv1 = nn.ConvTranspose2d(
                8 * dim, 4 * dim, 4, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(4 * dim)
            self.conv2 = nn.ConvTranspose2d(
                4 * dim, 2 * dim, 4, 2, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(2 * dim)
            self.conv3 = nn.ConvTranspose2d(2 * dim, dim, 4, 2, 2, bias=False)
            self.bn3 = nn.BatchNorm2d(dim)

        self.conv_out = nn.ConvTranspose2d(
            dim, 1, 4, stride=2, padding=1, bias=False)

    def forward(self, input, state=(), enable_autocast=False):
        with autocast(enabled=enable_autocast):
            if self.metric == 'jsd':
                x = input.view(input.shape[0], -1, 1, 1)
            elif self.metric == 'wasserstein':
                x = F.relu(self.bn0(self.fc1(input)))
                x = x.view(-1, 8 * self.dim, 2, 2)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.conv_out(x)
            x = torch.sigmoid(x)
            if self.flat:
                x = x.reshape(input.shape[0], -1)
        return x, state


class Discriminator(Network):
    def __init__(self, dim, input_dim, metric):
        super().__init__(
            input_tensor_spec=TensorSpec(
                shape=(input_dim, ), dtype=torch.float32),
            name="Discriminator")
        self.dim = dim
        self.metric = metric
        if metric == 'jsd':
            self.conv1 = nn.Conv2d(1, dim, 5, stride=2, padding=2, bias=False)
            self.conv2 = nn.Conv2d(
                dim, 2 * dim, 5, stride=2, padding=2, bias=False)
            self.conv3 = nn.Conv2d(
                2 * dim, 4 * dim, 5, stride=2, padding=2, bias=False)
        elif metric == 'wasserstein':
            self.conv1 = nn.Conv2d(1, dim, 5, stride=2, padding=2, bias=False)
            self.conv2 = nn.Conv2d(
                dim, 2 * dim, 5, stride=2, padding=2, bias=False)
            self.conv3 = nn.Conv2d(
                2 * dim, 4 * dim, 5, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(4 * dim, 1, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(2 * dim)
        self.bn3 = nn.BatchNorm2d(4 * dim)
        self.linear_out = nn.Linear(4 * 4 * 4 * dim, 1)

    def forward(self, input, state=(), enable_autocast=False):
        with autocast(enabled=enable_autocast):
            if input.ndim < 3:
                try:
                    input = input.reshape(-1, 1, 28, 28)
                except:
                    raise ValueError
            if self.metric == 'jsd':
                x = F.leaky_relu(self.bn1(self.conv1(input)), .2)
                x = F.leaky_relu(self.bn2(self.conv2(x)), .2)
                x = F.leaky_relu(self.bn3(self.conv3(x)), .2)
                x = self.conv4(x)
            elif self.metric == 'wasserstein':
                x = F.elu(self.conv1(input))
                x = F.elu(self.conv2(x))
                x = F.elu(self.conv3(x))
                x = x.view(-1, 4 * 4 * 4 * self.dim)
                x = self.linear_out(x)

            #if self.metric == 'jsd':
            #    x = torch.sigmoid(x)
        return x.view(-1), state


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def generate_image(generator, batch_size, epoch, path, noise_dim, final=False):
    """Generates a batch of superimposed MNIST 1k images"""
    generator.eval()
    noise = torch.randn(batch_size, noise_dim)
    samples, _ = generator._generator._net(noise, enable_autocast=False)
    if isinstance(samples, tuple):
        samples, _ = samples
    samples = samples.cpu()
    samples = samples.reshape(-1, 1, 28, 28).float()

    save_dir = '/nfs/hpc/share/ratzlafn/alf-plots/gan/MNIST'
    save_dir = save_dir + path
    os.makedirs(save_dir, exist_ok=True)
    logging.info('saving step {} to {}'.format(epoch, save_dir))
    save_image(samples, '{}/mnist_gen_samples_{}.png'.format(save_dir, epoch))
    if final == True:
        np.save('{}/final_samples.npy'.format(save_dir), samples.numpy())
    generator.train()


class GenerativeAdversarialTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(
        dict(
            par_vi='svgd',
            functional_gradient='rkhs',
            entropy_regularization=.1,
            metric='jsd',
            noise_dim=784,
            d_cap=1.0,
            grad_lambda=0.0,
            dlr=1e-4,
            glr=1e-4), )
    def test_gan(
            self,
            par_vi=None,  #'svgd3',
            functional_gradient=None,  #'rkhs',
            entropy_regularization=1.0,
            batch_size=64,
            d_cap=.25,
            metric='jsd',
            noise_dim=128,
            dlr=1e-2,
            glr=1e-2,
            plr=1e-4,
            grad_lambda=1.):
        """
        The generator is trained to match the likelihood of 8 Gaussian
        distributions
        """
        logging.info("GAN: MNIST")
        if par_vi == None:
            if entropy_regularization > 0.:
                entropy_regularization = 0.
            flat = False
        else:
            if entropy_regularization == 0.:
                entropy_regularization = 1.0  # batch_size / 50e3
            flat = True

        dim = 64
        d_cap = d_cap
        noise_dim = noise_dim
        d_iters = 5
        input_dim = 784
        metric = metric
        #metric = 'jsd'
        grad_lambda = grad_lambda
        pinverse_hidden_size = 128
        pinverse_solve_iters = 1
        fullrank_diag_weight = .01
        glr = glr
        dlr = dlr
        plr = plr
        net = Generator(dim, noise_dim, metric, flat)
        critic = Discriminator(int(dim * d_cap), input_dim, metric=metric)

        net.apply(weights_init)
        critic.apply(weights_init)
        train_loader, test_loader = load_mnist(
            train_bs=batch_size,
            test_bs=batch_size,
            normalize=False,
            num_workers=0)

        generator = GenerativeAdversarialAlgorithm(
            output_dim=28 * 28 * 1,
            input_tensor_spec=TensorSpec(shape=(1, 28, 28)),
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
            scaler=torch.cuda.amp.GradScaler(enabled=False),
            fullrank_diag_weight=fullrank_diag_weight,
            pinverse_hidden_size=pinverse_hidden_size,
            pinverse_solve_iters=pinverse_solve_iters,
            pinverse_optimizer=alf.optimizers.Adam(lr=plr),
            expectation_logp=False,
            use_kernel_logp=False,
            critic_optimizer=torch.optim.Adam(
                critic.parameters(), lr=dlr, betas=(.5, .9),
                weight_decay=1e-4),
            optimizer=alf.optimizers.Adam(
                lr=glr, betas=(.5, .9), weight_decay=1e-4),
            logging_training=True)

        generator.set_data_loader(train_loader, train_loader)
        logging.info("Generative Adversarial Network Test")

        def _train(i):
            alg_step = generator.train_iter(save_samples=False)

        if functional_gradient is not None:
            path = f'/gpvi-ldiag{fullrank_diag_weight}'
            path += f'-plr{plr}-inv{pinverse_hidden_size}-iters{pinverse_solve_iters}'
        elif par_vi != None:
            path = f'/{par_vi}'
        elif metric == 'jsd':
            path = '/gan_bn'
        elif metric == 'wasserstein':
            path = '/wgan'
        path += f'-{metric}_dlr{dlr}_glr{glr}_bs{batch_size}_diters{d_iters}'
        path += f'-gp{grad_lambda}-ent{entropy_regularization}_dcap{d_cap}'
        path += f'-z{noise_dim}_noavg'

        for epoch in range(1000):
            _train(epoch)
            if epoch % 5 == 0:
                with torch.no_grad():
                    generate_image(generator, 64, epoch, path, noise_dim)
        with torch.no_grad():
            generate_image(generator, 64, epoch, path, noise_dim, final=True)


if __name__ == '__main__':
    alf.test.main()
