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

import os
import absl
from absl.testing import parameterized
import numpy as np
import torch
import torch.nn.functional as F

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.hypernetwork_algorithm import HyperNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops
from alf.utils.datagen import load_nclass_test
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_classification(i, algorithm, n_classes, data, tag=''):
    basedir = '/nfs/hpc/share/ratzlafn/alf-plots/gpvi/classification/{}/'.format(
        tag)
    print(basedir)
    os.makedirs(basedir, exist_ok=True)
    x = torch.linspace(-12, 12, 100)
    y = torch.linspace(-12, 12, 100)
    gridx, gridy = torch.meshgrid(x, y)
    grid = torch.stack((gridx.reshape(-1), gridy.reshape(-1)), -1)
    outputs = algorithm.predict_step(grid, num_particles=100).output.cpu()
    outputs = F.softmax(outputs, -1).detach()  # [B, D]
    std_outputs = outputs.std(1).cpu()
    conf_std = std_outputs.max(-1)[0] * 1.94
    p1 = plt.scatter(
        grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_std, cmap='rainbow')
    p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black', alpha=0.1)
    cbar = plt.colorbar(p1)
    cbar.set_label("confidance (std)")
    plt.savefig(basedir + 'std-map_{}.png'.format(i))
    plt.close('all')


class HyperNetworkClassificationTest(parameterized.TestCase,
                                     alf.test.TestCase):
    """ 
    HyperNetwork Classification Test
        A 2/4 class classification problem, where the classes are distributed
        as 2/4 symmetric Normal distributions with non overlapping support. 
        The hypernetwork is trained to sample classification functions that
        fit the data, for the purpose of observing the predictive
        distributions of sampled funcitons on data outside the training
        distribution. 
    """

    @parameterized.parameters(
        #('svgd2', False, None),
        #('svgd3', False, None, 32, 32),
        #('gfsf', False, None),
        #('minmax', False, None),
        #('svgd2', True, None),
        #('svgd3', True, None),
        #('gfsf', True, None),
        #('svgd3', False, 'rkhs', 24, 64, 64, .3, 1),
        #('svgd3', False, 'rkhs', 24, 64, 64, .35, 1),
        #('svgd3', False, 'rkhs', 24, 64, 64, .25, 1),
        #('svgd3', False, 'rkhs', 24, 100, 100, .3, 1),
        #('svgd3', False, 'rkhs', 24, 100, 100, .25, 1),
        #('svgd3', False, 'rkhs', 24, 100, 100, .35, 1),
        #('svgd3', False, 'rkhs', 10, 10, 10, .3, 1),
        #('svgd3', False, 'rkhs', 24, 32, 32, .35, 1),
        #('svgd3', False, 'rkhs', 24, 32, 32, .25, 1),
        #('svgd3', False, 'rkhs', 151, 151, 151, .2, 1),
        #('svgd3', False, 'rkhs', 32, 32, 64, .5),
        ('svgd3', False, 'rkhs', 151, 151, 151, .1, 0),
        #('svgd3', False, 'rkhs', 64, 64, 24),
        #('svgd3', False, 'rkhs', 64, 64, 32),
        #('svgd3', False, 'rkhs', 64, 64, 16),

        #('svgd3', False, 'rkhs', 24, 24, 24, .3, 1),
        #('svgd3', False, 'rkhs', 24, 24, 24, .3, 2),
        #('svgd3', False, 'rkhs', 24, 24, 24, .4, 1),
        #('svgd3', False, 'rkhs', 24, 24, 24, .4, 2),
        #('svgd3', False, 'rkhs', 24, 24, 24, .5, 1),
        #('svgd3', False, 'rkhs', 24, 24, 24, .5, 2),
        #('svgd3', False, 'rkhs', 24, 24, 24, .6, 1),
        #('svgd3', False, 'rkhs', 24, 24, 24, .6, 2),
        #('svgd3', False, 'rkhs', 24, 24, 64, .3, 1),
        #('svgd3', False, 'rkhs', 24, 24, 64, .3, 2),
        #('svgd3', False, 'rkhs', 24, 24, 64, .4, 1),
        #('svgd3', False, 'rkhs', 24, 24, 64, .4, 2),
        #('svgd3', False, 'rkhs', 24, 24, 64, .5, 1),
        #('svgd3', False, 'rkhs', 24, 24, 64, .5, 2),
        #('svgd3', False, 'rkhs', 24, 24, 64, .6, 1),
        #('svgd3', False, 'rkhs', 24, 24, 64, .6, 2),
        #('svgd3', False, 'rkhs', 24, 24, 128, .3, 1),
        #('svgd3', False, 'rkhs', 24, 24, 128, .3, 2),
        #('svgd3', False, 'rkhs', 24, 24, 128, .4, 1),
        #('svgd3', False, 'rkhs', 24, 24, 128, .4, 2),
        #('svgd3', False, 'rkhs', 24, 24, 128, .5, 1),
        #('svgd3', False, 'rkhs', 24, 24, 128, .5, 2),
        #('svgd3', False, 'rkhs', 24, 24, 128, .6, 1),
        #('svgd3', False, 'rkhs', 24, 24, 128, .6, 2),"""
        #2
        #('svgd3', False, 'rkhs', 64, 64, 64, .3, 1),
        #('svgd3', False, 'rkhs', 64, 64, 64, .3, 2),
        #('svgd3', False, 'rkhs', 64, 64, 64, .4, 1),
        #('svgd3', False, 'rkhs', 64, 64, 64, .4, 2),
        #('svgd3', False, 'rkhs', 64, 64, 64, .5, 1),
        #('svgd3', False, 'rkhs', 64, 64, 64, .5, 2),
        #('svgd3', False, 'rkhs', 64, 64, 64, .6, 1),
        #('svgd3', False, 'rkhs', 64, 64, 64, .6, 2),
        #('svgd3', False, 'rkhs', 64, 64, 256, .3, 1),
        #('svgd3', False, 'rkhs', 64, 64, 256, .3, 2),
        #('svgd3', False, 'rkhs', 64, 64, 256, .4, 1),
        #('svgd3', False, 'rkhs', 64, 64, 256, .4, 2),
        #('svgd3', False, 'rkhs', 64, 64, 256, .5, 1),
        #('svgd3', False, 'rkhs', 64, 64, 256, .5, 2),
        #('svgd3', False, 'rkhs', 64, 64, 256, .6, 1),
        #('svgd3', False, 'rkhs', 64, 64, 256, .6, 2),
        #('svgd3', False, 'rkhs', 64, 64, 128, .3, 1),
        #('svgd3', False, 'rkhs', 64, 64, 128, .3, 2),
        #('svgd3', False, 'rkhs', 64, 64, 128, .4, 1),
        #('svgd3', False, 'rkhs', 64, 64, 128, .4, 2),
        #('svgd3', False, 'rkhs', 64, 64, 128, .5, 1),
        #('svgd3', False, 'rkhs', 64, 64, 128, .5, 2),
        #('svgd3', False, 'rkhs', 64, 64, 128, .6, 1),
        #('svgd3', False, 'rkhs', 64, 64, 128, .6, 2),
        #3

        #('svgd3', False, 'rkhs', 128, 128, 128, .3, 1),
        #('svgd3', False, 'rkhs', 128, 128, 128, .3, 2),
        #('svgd3', False, 'rkhs', 128, 128, 128, .4, 1),
        #('svgd3', False, 'rkhs', 128, 128, 128, .4, 2),
        #('svgd3', False, 'rkhs', 128, 128, 128, .5, 1),
        #('svgd3', False, 'rkhs', 128, 128, 128, .5, 2),
        #('svgd3', False, 'rkhs', 128, 128, 128, .6, 1),
        #('svgd3', False, 'rkhs', 128, 128, 128, .6, 2),

        #('svgd3', False, 'rkhs', 128, 128, 256, .5, 1),
        #('svgd3', False, 'rkhs', 128, 128, 256, .5, 2),
        #('svgd3', False, 'rkhs', 128, 128, 256, .6, 1),
        #('svgd3', False, 'rkhs', 128, 128, 256, .6, 2),
        #('svgd3', False, 'rkhs', 128, 128, 256, .4, 1),
        #('svgd3', False, 'rkhs', 128, 128, 256, .4, 2),
        #('svgd3', False, 'rkhs', 128, 128, 256, .3, 1),
        #('svgd3', False, 'rkhs', 128, 128, 256, .3, 2),

        #('svgd3', False, 'rkhs', 128, 128, 512, .5, 1),
        #('svgd3', False, 'rkhs', 128, 128, 512, .5, 2),
        #('svgd3', False, 'rkhs', 128, 128, 512, .3, 1),
        #('svgd3', False, 'rkhs', 128, 128, 512, .3, 2),
        #('svgd3', False, 'rkhs', 128, 128, 512, .6, 1),
        #('svgd3', False, 'rkhs', 128, 128, 512, .6, 2),
        #('svgd3', False, 'rkhs', 128, 128, 512, .4, 1),
        #('svgd3', False, 'rkhs', 128, 128, 512, .4, 2),

        #('minmax', False, 'minmax'),
    )
    def test_classification_hypernetwork(self,
                                         par_vi='svgd3',
                                         function_vi=False,
                                         functional_gradient='rkhs',
                                         noise_dim=16,
                                         hidden_size=16,
                                         pinverse_hidden_size=16,
                                         lamb=1.0,
                                         n_hidden=1,
                                         num_classes=2,
                                         num_particles=100):
        """
        Symmetric 4-class classification problem. The training data are drawn
        from standard normal distributions, each class is given by one of
        these distributions. The hypernetwork is trained to generate parameters
        that achieves low loss / high accuracy on this data.
        """

        input_size = 2
        output_dim = num_classes
        input_spec = TensorSpec((input_size, ), torch.float64)
        train_size = 100
        test_size = 200
        batch_size = train_size
        train_batch_size = train_size
        train_loader, test_loader = load_nclass_test(
            num_classes,
            train_size=train_size,
            test_size=test_size,
            train_bs=train_batch_size,
            test_bs=train_batch_size)
        test_inputs = test_loader.dataset.get_features()
        test_targets = test_loader.dataset.get_targets()

        #noise_dim = 32
        #hidden_size = 32
        #pinverse_hidden_size = pinverse
        lr = 1e-4
        plr = 1e-4
        pinverse_solve_iters = 2
        config = TrainerConfig(root_dir='dummy')
        algorithm = HyperNetwork(
            input_tensor_spec=input_spec,
            fc_layer_params=((10, True), (10, True)),
            last_layer_param=(output_dim, True),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            hidden_layers=(hidden_size, ) * n_hidden,  #hidden_size),
            num_particles=num_particles,
            loss_type='classification',
            par_vi=par_vi,
            function_vi=function_vi,
            function_bs=train_batch_size,
            functional_gradient=functional_gradient,
            block_pinverse=True,
            force_fullrank=True,
            jac_autograd=False,
            fullrank_diag_weight=lamb,
            pinverse_hidden_size=pinverse_hidden_size,
            critic_hidden_layers=(hidden_size, hidden_size),
            critic_iter_num=5,
            pinverse_solve_iters=pinverse_solve_iters,
            optimizer=alf.optimizers.Adam(lr=lr, weight_decay=0),
            critic_optimizer=alf.optimizers.Adam(lr=lr),
            pinverse_optimizer=alf.optimizers.Adam(lr=plr),
            #logging_training=True,
            config=config)

        algorithm.set_data_loader(
            train_loader,
            test_loader,
            entropy_regularization=batch_size / train_size)
        absl.logging.info(
            'Hypernetwork: Fitting {} Classes'.format(num_classes))
        absl.logging.info('{} - {} particles'.format(par_vi, num_particles))

        def _test(i):
            outputs = algorithm.predict_step(test_inputs).output
            probs = F.softmax(outputs, dim=-1)
            preds = probs.mean(1).cpu().argmax(-1)
            mean_acc = preds.eq(test_targets.cpu().view_as(preds)).float()
            mean_acc = mean_acc.sum() / len(test_targets)

            sample_preds = probs.cpu().argmax(-1).reshape(-1, 1)
            targets_unrolled = test_targets.unsqueeze(1).repeat(
                1, num_particles).reshape(-1, 1)

            sample_acc = sample_preds.eq(
                targets_unrolled.cpu().view_as(sample_preds)).float()
            sample_acc = sample_acc.sum() / len(targets_unrolled)

            absl.logging.info('-' * 86)
            absl.logging.info('iter {}'.format(i))
            absl.logging.info('mean particle acc: {}'.format(mean_acc.item()))
            absl.logging.info('all particles acc: {}'.format(
                sample_acc.item()))
            tag = f'2cls_gpvi_g2{noise_dim}_h{hidden_size}_lr{lr}_p{pinverse_hidden_size}_l{lamb}_nh{n_hidden}_plr{plr}_iters{pinverse_solve_iters}'
            plot_classification(i, algorithm, num_classes, test_inputs, tag)

        train_iter = 1000000
        for i in range(train_iter):
            algorithm.train_iter()
            if i % 5000 == 0:
                _test(i)

        algorithm.evaluate()

        outputs = algorithm.predict_step(test_inputs).output
        probs = F.softmax(outputs, dim=-1)
        preds = probs.mean(1).cpu().argmax(-1)
        mean_acc = preds.eq(test_targets.cpu().view_as(preds)).float()
        mean_acc = mean_acc.sum() / len(test_targets)

        sample_preds = probs.cpu().argmax(-1).reshape(-1, 1)
        targets_unrolled = test_targets.unsqueeze(1).repeat(
            1, num_particles).reshape(-1, 1)

        sample_acc = sample_preds.eq(
            targets_unrolled.cpu().view_as(sample_preds)).float()
        sample_acc = sample_acc.sum() / len(targets_unrolled)
        absl.logging.info('-' * 86)
        absl.logging.info('iter {}'.format(i))
        absl.logging.info('mean particle acc: {}'.format(mean_acc.item()))
        absl.logging.info('all particles acc: {}'.format(sample_acc.item()))
        self.assertGreater(mean_acc, 0.95)
        self.assertGreater(sample_acc, 0.95)


if __name__ == "__main__":
    alf.test.main()
