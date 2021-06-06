# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import alf
from alf.algorithms import functional_particle_vi_algorithm
from alf.trainers import policy_trainer

conv_net = ((32, 3, 1, 0, 2), (64, 3, 1, 0, 2), (64, 3, 1, 0, 2))
fc_net = ((128, True), )

alf.config(
    'create_dataset',
    dataset_name='cifar10',
    train_batch_size=16,
    test_batch_size=100)

alf.config(
    'FuncParVIAlgorithm',
    conv_layer_params=conv_net,
    fc_layer_params=fc_net,
    num_particles=10,
    par_vi='svgd',
    loss_type='classification',
    entropy_regularization=1.5,
    optimizer=alf.optimizers.Adam(lr=1e-3, weight_decay=1e-4),
    logging_training=True,
    logging_evaluate=True)

alf.config('ParamConvNet', use_bias=True)

alf.config(
    'TrainerConfig',
    algorithm_ctor=functional_particle_vi_algorithm.FuncParVIAlgorithm,
    #ml_type='sl',
    num_iterations=200,
    num_checkpoints=1,
    evaluate=True,
    eval_uncertainty=True,
    eval_interval=1,
    summary_interval=1,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    hold_out_dataset='cifar10',
    train_classes=[0, 1, 2, 3, 4, 5],
    hold_out_classes=[6, 7, 8, 9],
    random_seed=None)
