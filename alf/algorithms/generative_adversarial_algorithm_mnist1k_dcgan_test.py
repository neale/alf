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


class Generator(nn.Module):
    def __init__(self, ngf, z_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        input = input.view(-1, 128, 1, 1)
        output = self.main(input)
        print(output.shape)
        return output


class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(8, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )

    def forward(self, input):
        print(input.shape)
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


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
                samples = samples.view(-1, 3, 64, 64)
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
    samples = samples.reshape(-1, 3, 64, 64)

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
        samples = samples.reshape(-1, 3, 64, 64)
        final_samples.append(samples)
    final_samples = torch.stack(final_samples).view(-1, 3, 64, 64)

    if final == True:
        print('saved samples: ', final_samples.shape)
        np.save('{}/final_samples.npy'.format(save_dir), final_samples.numpy())


class GenerativeAdversarialTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(

        # JSD
        dict(
            par_vi=None,
            functional_gradient=None,
            entropy_regularization=0.,
            metric='jsd',
            noise_dim=128,
            d_cap=1.0),
        # WGAN-GP
        #dict(
        #    par_vi=None, functional_gradient=None, entropy_regularization=0.,
        #    metric='w1', noise_dim=128, d_cap=1.0, grad_lambda=10.,
        #    batch_size=64),

        # GPVI+ JSD
        #dict(
        #    par_vi='svgd', functional_gradient='rkhs', entropy_regularization=.1,
        #    metric='jsd', noise_dim=128, grad_lambda=0.0, d_cap=1.0, diag=.1,
        #    p_hidden=512, p_iters=1, glr=1e-4, batch_size=64),

        # GPVI+ WGAN
        #dict(par_vi='svgd', functional_gradient='rkhs', entropy_regularization=.1,
        #     metric='kl-w1', noise_dim=128, grad_lambda=0.0, use_sn=True, d_cap=1.0,
        #     diag=.1, p_hidden=512, p_iters=1, glr=1e-4, batch_size=64),
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
        input_dim = 64 * 64
        d_cap = d_cap
        metric = metric
        use_sn = use_sn
        grad_lambda = grad_lambda
        net = Generator(dim, noise_dim)
        net.apply(weights_init)
        critic = Discriminator(input_dim)
        critic.apply(weights_init)

        train_loader = load_mnist1k(
            train_bs=batch_size, as_ds=True, normalize=True, scale=64)
        train_loader = torch.utils.data.DataLoader(
            train_loader, batch_size=batch_size, shuffle=True)

        pinverse_hidden_size = p_hidden
        pinverse_solve_iters = p_iters
        fullrank_diag_weight = diag
        glr = glr
        dlr = dlr
        plr = 1e-4

        # mnist_clf = train_classifier()
        mnist_clf = load_classifier()

        generator = GenerativeAdversarialAlgorithm(
            output_dim=64 * 64 * 3,
            input_tensor_spec=TensorSpec(shape=(3, 64, 64)),
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
