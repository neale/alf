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
#torch.backends.cudnn.benchmark = True
from torch.cuda.amp import autocast


class Generator(Network):
    def __init__(self, dim, noise_dim, flat):
        super().__init__(
            input_tensor_spec=TensorSpec(
                shape=(noise_dim, ), dtype=torch.float32),
            name="Generator")

        self.dim = dim
        self.flat = flat
        self.fc1 = nn.Linear(noise_dim, 4 * 4 * 4 * dim, bias=False)
        self.bn1 = nn.BatchNorm1d(4 * 4 * 4 * dim)

        self.conv1 = nn.ConvTranspose2d(4 * dim, 2 * dim, 5, bias=False)
        self.bn2 = nn.BatchNorm2d(2 * dim, 2 * dim)

        self.conv2 = nn.ConvTranspose2d(2 * dim, dim, 5, bias=False)
        self.bn3 = nn.BatchNorm2d(dim, dim)

        self.conv_out = nn.ConvTranspose2d(dim, 1, 8, stride=2, padding=1)

    #@autocast()
    def forward(self, input, state=()):
        x = F.relu(self.bn1(self.fc1(input)))
        x = x.view(-1, 4 * self.dim, 4, 4)
        x = F.relu(self.bn2(self.conv1(x)))
        x = F.relu(self.bn3(self.conv2(x)))
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
        self.conv1 = nn.Conv2d(1, dim, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(2 * dim, 4 * dim, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(dim, dim)
        self.bn2 = nn.BatchNorm2d(2 * dim, 2 * dim)
        self.bn3 = nn.BatchNorm2d(4 * dim, 4 * dim)
        self.linear_out = nn.Linear(4 * 4 * 4 * dim, 1)

    #@autocast()
    def forward(self, input, state=()):
        if input.ndim < 3:
            try:
                input = input.reshape(-1, 1, 28, 28)
            except:
                raise ValueError
        if self.metric == 'jsd':
            x = F.leaky_relu(self.bn1(self.conv1(input)))
            x = F.leaky_relu(self.bn2(self.conv2(x)))
            x = F.leaky_relu(self.bn3(self.conv3(x)))
        elif self.metric == 'wasserstein':
            x = F.leaky_relu(self.conv1(input))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
        x = x.view(-1, 4 * 4 * 4 * self.dim)
        x = self.linear_out(x)

        if self.metric == 'jsd':
            x = torch.sigmoid(x)
        return x.view(-1), state


def generate_image(generator, batch_size, epoch, path, final=False):
    """Generates a batch of superimposed MNIST 1k images"""
    noise = torch.randn(batch_size, 128)
    samples, _ = generator._generator._net(noise)
    if isinstance(samples, tuple):
        samples, _ = samples
        #
    samples = samples.cpu()
    samples = samples.reshape(-1, 1, 28, 28).float()

    save_dir = '/nfs/hpc/share/ratzlafn/alf-plots/gan/MNIST'
    save_dir = save_dir + path
    os.makedirs(save_dir, exist_ok=True)
    logging.info('saving step {} to {}'.format(epoch, save_dir))
    save_image(samples, '{}/mnist_gen_samples_{}.png'.format(save_dir, epoch))
    if final == True:
        np.save('{}/final_samples.npy'.format(save_dir), samples.numpy())


def test_gan(par_vi='svgd',
             functional_gradient='rkhs',
             entropy_regularization=1.0,
             batch_size=128):
    """
    The generator is trained to match the likelihood of 8 Gaussian
    distributions
    """
    logging.info("GAN: MNIST")
    if par_vi == None:
        entropy_regularization = 0.
        flat = False
    else:
        entropy_regularization = .1  # batch_size / 50e3
        flat = True

    dim = 16
    noise_dim = 64
    d_iters = 3
    input_dim = 784
    metric = 'wasserstein'
    # metric = 'jsd'
    net = Generator(dim, noise_dim, flat)
    critic = Discriminator(dim, input_dim, metric=metric)

    train_loader, test_loader = load_mnist(
        train_bs=batch_size,
        test_bs=batch_size,
        normalize=False,
        num_workers=0)

    grad_lambda = 0.1
    pinverse_hidden_size = 128
    pinverse_solve_iters = 1
    fullrank_diag_weight = 1.0
    glr = 1e-4
    dlr = 1e-4
    plr = 1e-4

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
        scaler=torch.cuda.amp.GradScaler(),
        fullrank_diag_weight=fullrank_diag_weight,
        pinverse_hidden_size=pinverse_hidden_size,
        pinverse_solve_iters=pinverse_solve_iters,
        pinverse_optimizer=alf.optimizers.Adam(lr=plr),
        critic_optimizer=alf.optimizers.Adam(lr=dlr, betas=(.5, .99)),
        optimizer=alf.optimizers.Adam(lr=glr, betas=(.5, .99)),
        logging_training=True)

    generator.set_data_loader(train_loader, train_loader)
    logging.info("Generative Adversarial Network Test")

    def _train(i):
        print('train')
        alg_step = generator.train_iter(save_samples=False)

    if functional_gradient is not None:
        path = f'/gpvi-ldiag{fullrank_diag_weight}_z{noise_dim}'
        path += f'-plr{plr}-inv{pinverse_hidden_size}-iters{pinverse_solve_iters}'
    elif par_vi == 'svgd3':
        path = '/svgd3'
    elif metric == 'jsd':
        path = '/gan_bn'
    elif metric == 'wasserstein':
        path = '/wgan'
    path += f'-{metric}_dlr{dlr}_glr{glr}'
    path += f'-gp{grad_lambda}-ent{entropy_regularization}'
    path += f'_profile_amp'
    for epoch in range(2):
        _train(epoch)
        #if epoch % 5 == 0:
        #with torch.no_grad():
        #    generate_image(generator, 128, epoch, path)
    #with torch.no_grad():
    #    generate_image(generator, 128, epoch, path, final=True)


test_gan()
