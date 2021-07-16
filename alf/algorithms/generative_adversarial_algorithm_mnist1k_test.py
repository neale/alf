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

from torch.cuda.amp import autocast, GradScaler
from generative_adversarial_algorithm_mnist1k_dcgan_test import Generator64, Discriminator64


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
        elif metric in ['w1', 'kl-w1']:
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
            dim, 3, 4, stride=2, padding=1, bias=False)

    def forward(self, input, state=(), enable_autocast=False):
        with autocast(enabled=enable_autocast):
            if self.metric == 'jsd':
                x = input.view(input.shape[0], -1, 1, 1)
            elif self.metric in ['w1', 'kl-w1']:
                x = F.relu(self.bn0(self.fc1(input)))
                x = x.view(-1, 8 * self.dim, 2, 2)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.conv_out(x)
            if self.metric == 'jsd':
                x = torch.tanh(x)
            else:
                x = torch.sigmoid(x)

            if self.flat:
                x = x.reshape(input.shape[0], -1)
        return x, state


class Discriminator(Network):
    def __init__(self, dim, input_dim, metric, use_sn):
        super().__init__(
            input_tensor_spec=TensorSpec(
                shape=(input_dim, ), dtype=torch.float32),
            name="Discriminator")
        self.dim = dim
        self.metric = metric
        if use_sn:
            sn = torch.nn.utils.spectral_norm
        else:
            sn = lambda x: x

        if metric == 'jsd':
            self.conv1 = nn.Conv2d(3, dim, 5, stride=2, padding=2, bias=False)
            self.conv2 = nn.Conv2d(
                dim, 2 * dim, 5, stride=2, padding=2, bias=False)
            self.conv3 = nn.Conv2d(
                2 * dim, 4 * dim, 5, stride=2, padding=2, bias=False)
        elif metric in ['w1', 'kl-w1']:
            self.conv1 = sn(
                nn.Conv2d(3, dim, 5, stride=2, padding=2, bias=False))
            self.conv2 = sn(
                nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2, bias=False))
            self.conv3 = sn(
                nn.Conv2d(
                    2 * dim, 4 * dim, 5, stride=2, padding=2, bias=False))
        self.conv4 = sn(nn.Conv2d(4 * dim, 1, 4, bias=False))
        self.linear_out = sn(nn.Linear(4 * 4 * 4 * dim, 1))

        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(2 * dim)
        self.bn3 = nn.BatchNorm2d(4 * dim)

    def forward(self, input, state=(), enable_autocast=False):
        with autocast(enabled=enable_autocast):
            if input.ndim < 3:
                try:
                    input = input.reshape(-1, 3, 28, 28)
                except:
                    raise ValueError
            if self.metric == 'jsd':
                x = F.leaky_relu(self.bn1(self.conv1(input)), .2)
                x = F.leaky_relu(self.bn2(self.conv2(x)), .2)
                x = F.leaky_relu(self.bn3(self.conv3(x)), .2)
                x = self.conv4(x)
            elif self.metric in ['w1', 'kl-w1']:
                x = F.elu(self.conv1(input))
                x = F.elu(self.conv2(x))
                x = F.elu(self.conv3(x))
                x = x.view(-1, 4 * 4 * 4 * self.dim)
                x = self.linear_out(x)
        return x.view(-1), state


class MNISTNet(nn.Module):
    def __init__(self, input_dim):
        super(MNISTNet, self).__init__()
        self.input_dim = input_dim
        self.block1 = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1), nn.ReLU())
        self.pool1 = nn.AvgPool2d(2, stride=2)

        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=1), nn.ReLU())
        self.pool2 = nn.AvgPool2d(2, stride=2)

        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1), nn.ReLU())
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1), nn.ReLU())
        self.linear1 = nn.Linear(128, 84, bias=True)
        self.linear_out = nn.Linear(84, 10, bias=True)

    def forward(self, inputs, state=()):
        x = self.block1(inputs)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(-1, 128)
        x = F.relu(self.linear1(x))
        x = self.linear_out(x)
        return x, state


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def make_rgb_batch(data, target):
    new_batch = []
    new_targets = []
    for i, item in enumerate(data):
        batch = torch.cat((data[:i], data[i:]), dim=0)
        batch_y = torch.cat((target[:i], target[i:]), dim=0)
        indices = np.random.randint(0, len(batch) - 1, size=(2, ))
        new_dim_item = torch.cat((item, batch[indices[0]], batch[indices[1]]),
                                 dim=0)
        new_dim_target = target[i] * 100 + batch_y[indices[0]] * 10 + batch_y[
            indices[1]]
        new_batch.append(new_dim_item)
        new_targets.append(new_dim_target)
    data = torch.stack(new_batch)
    targets = torch.tensor(new_targets)
    return data, targets


def train_classifier():
    net = MNISTNet(input_dim=28)
    train_loader, test_loader = load_mnist(
        train_bs=100, test_bs=100, normalize=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(20):
        cum_loss = 0.
        avg_acc = 0.
        for i, (data, target) in enumerate(train_loader):
            data = data.to(alf.get_default_device())
            target = target.to(alf.get_default_device())
            preds = net(data)[0]
            target = target.reshape(target.shape[0], 1)
            alg_step = classification_loss(preds, target)
            loss = alg_step.loss
            acc = alg_step.extra
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
            avg_acc += acc
        avg_acc /= i
        logging.info('[Epoch {}] classifier cum loss {}'.format(epoch, loss))
        logging.info('[Epoch {}] classifier avg acc {}'.format(epoch, avg_acc))
    model_state = net.state_dict()
    optim_state = optimizer.state_dict()
    logging.info('saving model and optimizer')
    torch.save({'model': model_state, 'optim': optim_state}, '1k_clf.pt')
    return net


def load_classifier():
    state = torch.load('1k_clf.pt')
    net_state = state['model']
    optim_state = state['optim']
    net = MNISTNet(input_dim=28)
    net.load_state_dict(net_state)
    net.eval()
    print('Loaded mnist classifier')
    return net


def test_generator_coverage(epoch, generator, classifier, path):
    gen_classes = []
    for _ in range(1000):
        samples, _ = generator._generator._predict(
            batch_size=100, training=False)
        if isinstance(samples, tuple):
            _, samples = samples
        if samples.ndim < 3:
            try:
                samples = samples.view(-1, 3, 28, 28)
            except:
                raise ValueError

        for sample in samples:
            sample = sample.unsqueeze(1)
            class_idx = torch.argmax(classifier(sample)[0], dim=-1)
            class_id = int(
                str(class_idx[0].item()) + str(class_idx[1].item()) +
                str(class_idx[2].item()))
            gen_classes.append(class_id)

    gen_classes = torch.tensor(gen_classes).view(-1).long()
    num_classes = len(torch.unique(gen_classes))
    logging.info('[Epoch {}] Generated {} classes'.format(epoch, num_classes))
    save_dir = '/nfs/hpc/share/ratzlafn/alf-plots/gan/MNIST1K'
    save_dir = save_dir + path
    os.makedirs(save_dir, exist_ok=True)
    np.save('{}/class_preds_{}.npy'.format(save_dir, epoch),
            gen_classes.detach().cpu().numpy())


def generate_image(generator, batch_size, epoch, path, final=False):
    """Generates a batch of superimposed MNIST 1k images"""
    samples, _ = generator._generator._predict(
        batch_size=batch_size, training=False)
    if isinstance(samples, tuple):
        _, samples = samples
    samples = samples.cpu()
    samples = samples.reshape(-1, 3, 28, 28)

    save_dir = '/nfs/hpc/share/ratzlafn/alf-plots/gan/MNIST1K'
    save_dir = save_dir + path
    os.makedirs(save_dir, exist_ok=True)
    logging.info('saving step {} to {}'.format(epoch, save_dir))
    save_image(samples, '{}/mnist_1k_gen_samples_{}.png'.format(
        save_dir, epoch))
    final_samples = []
    for _ in range(100):
        samples, _ = generator._generator._predict(
            batch_size=100, training=False)
        if isinstance(samples, tuple):
            _, samples = samples
        samples = samples.cpu()
        samples = samples.reshape(-1, 3, 28, 28)
        final_samples.append(samples)
    final_samples = torch.stack(final_samples).view(-1, 3, 28, 28)

    if final == True:
        print('saved samples: ', final_samples.shape)
        np.save('{}/final_samples.npy'.format(save_dir), final_samples.numpy())


class GenerativeAdversarialTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(

        # JSD
        #dict(
        #    par_vi=None, functional_gradient=None, entropy_regularization=0.,
        #    metric='jsd', noise_dim=128, d_cap=.25),
        # WGAN-GP
        #dict(
        #    par_vi=None, functional_gradient=None, entropy_regularization=0.,
        #    metric='w1', noise_dim=128, d_cap=1.0, grad_lambda=10.,
        #    batch_size=64),

        # GPVI+ JSD
        dict(
            par_vi='svgd',
            functional_gradient='rkhs',
            entropy_regularization=.1,
            metric='jsd',
            noise_dim=128,
            grad_lambda=0.0,
            d_cap=0.25,
            diag=.01,
            p_hidden=256,
            p_iters=1,
            glr=1e-4,
            batch_size=64),

        # GPVI+ WGAN
        #dict(
        #    par_vi='svgd',
        #    functional_gradient='rkhs',
        #    entropy_regularization=.1,
        #    metric='w1',
        #    noise_dim=128,
        #    grad_lambda=1.0,
        #    use_sn=True,
        #    d_cap=1.0,
        #    diag=.01,
        #    p_hidden=512,
        #    p_iters=1,
        #    glr=1e-4,
        #    batch_size=64),
    )
    def test_gan(self,
                 par_vi='svgd',
                 functional_gradient='rkhs',
                 entropy_regularization=0.1,
                 metric='jsd',
                 noise_dim=128,
                 d_cap=1.0,
                 grad_lambda=0.0,
                 dlr=1e-4,
                 glr=1e-4,
                 use_sn=False,
                 diag=.1,
                 p_hidden=256,
                 p_iters=1,
                 batch_size=64):
        """
        The generator is trained to match the likelihood of 8 Gaussian
        distributions
        """
        logging.info("{}: 1k MNIST".format(metric))
        logging.info(
            f"par vi: {par_vi}, functional_gradient: {functional_gradient}")

        if par_vi == None:
            assert entropy_regularization == 0., "ent reg must be 0 for GAN"
            flat = False
        else:
            assert entropy_regularization > 0., "ent reg must be > 0 for par vi"
            flat = True

        dim = 64
        noise_dim = noise_dim
        d_iters = 5
        input_dim = 64 * 64  # 784
        d_cap = d_cap
        metric = metric
        use_sn = use_sn
        grad_lambda = grad_lambda
        net = Generator64(dim, noise_dim, metric, flat)
        net.apply(weights_init)
        critic = Discriminator64(
            int(d_cap * dim), input_dim, metric=metric, use_sn=use_sn)
        critic.apply(weights_init)

        train_loader, test_loader = load_mnist1k(
            train_bs=batch_size, test_bs=batch_size, scale=64)

        pinverse_hidden_size = p_hidden
        pinverse_solve_iters = p_iters
        fullrank_diag_weight = diag
        glr = glr
        dlr = dlr
        plr = 1e-4

        # mnist_clf = train_classifier()
        mnist_clf = load_classifier()

        generator = GenerativeAdversarialAlgorithm(
            output_dim=64 * 64 * 3,  #28 * 28 * 3,
            input_tensor_spec=TensorSpec(shape=(3, 64, 64)),  ##(3, 28, 28)),
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
            expectation_logp=False,
            use_kernel_logp=False,
            scaler=GradScaler(enabled=False),
            fullrank_diag_weight=fullrank_diag_weight,
            pinverse_hidden_size=pinverse_hidden_size,
            pinverse_solve_iters=pinverse_solve_iters,
            pinverse_optimizer=alf.optimizers.Adam(lr=plr),
            critic_optimizer=torch.optim.Adam(
                critic.parameters(), lr=dlr, betas=(.5, .99)),
            optimizer=alf.optimizers.Adam(lr=glr, betas=(.5, .99)),
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
        elif par_vi == None:
            path = '/gan'
        path += f'-{metric}_dlr{dlr}_glr{glr}_bs{batch_size}_diters{d_iters}'
        path += f'-gp{grad_lambda}-ent{entropy_regularization}_dcap{d_cap}'
        path += f'-z{noise_dim}_tanh_forKL'

        for epoch in range(100):
            _train(epoch)
            if epoch % 5 == 0:
                with torch.no_grad():
                    generate_image(generator, 64, epoch, path, final=True)
            if epoch % 10 == 0:
                test_generator_coverage(epoch, generator, mnist_clf, path)
        with torch.no_grad():
            generate_image(generator, 64, epoch, path, final=True)


if __name__ == '__main__':
    alf.test.main()
