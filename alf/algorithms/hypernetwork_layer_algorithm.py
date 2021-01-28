# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

from absl import logging
import functools
import gin
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable
from scipy.stats import entropy as entropy_fn
from sklearn.metrics import roc_auc_score

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.config import TrainerConfig
from alf.networks.relu_mlp import ReluMLP
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.algorithms.generator import Generator
from alf.algorithms.hypernetwork_layer_generator import ParamLayers
from alf.networks import EncodingNetwork, ParamNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops, summary_utils
from alf.utils.summary_utils import record_time


HyperNetworkLossInfo = namedtuple("HyperNetworkLossInfo", ["loss", "extra"])


def classification_loss(output, target):
    pred = output.max(-1)[1]
    acc = pred.eq(target).float().mean(0)
    avg_acc = acc.mean()
    if output.ndim == 2:  # function_vi
        target = target.reshape(-1)
        output = output.reshape(target.shape[0], -1)
    else:
        output = output.transpose(1, 2)
    loss = F.cross_entropy(output, target)
    return HyperNetworkLossInfo(loss=loss, extra=avg_acc)


def regression_loss(output, target):
    out_shape = output.shape[-1]
    assert (target.shape[-1] == out_shape), (
        "feature dimension of output and target does not match.")
    loss = 0.5 * F.mse_loss(
        output.reshape(-1, out_shape),
        target.reshape(-1, out_shape),
        reduction='sum')
    return HyperNetworkLossInfo(loss=loss, extra=())


@gin.configurable
class HyperNetwork(Algorithm):
    """HyperNetwork 

    HyperNetwork algorithm maintains a generator that generates a set of 
    parameters for a predefined neural network from a random noise input. 
    It is based on the following work:

    https://github.com/neale/HyperGAN

    Ratzlaff and Fuxin. "HyperGAN: A Generative Model for Diverse, 
    Performant Neural Networks." International Conference on Machine Learning. 2019.

    Major differences versus the original paper are:

    * A single genrator that generates parameters for all network layers.

    * Remove the mixer and the distriminator.

    * The generator is trained with Amortized particle-based variational 
      inference (ParVI) methods, please refer to generator.py for details.

    """

    def __init__(self,
                 input_tensor_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 last_layer_param=None,
                 last_activation=None,
                 noise_dim=32,
                 hidden_layers=(64, 64),
                 use_fc_bn=False,
                 num_particles=10,
                 entropy_regularization=1.,
                 critic_optimizer=None,
                 critic_hidden_layers=(100, 100),
                 critic_iter_num=2,
                 critic_l2_weight=10.,
                 function_vi=False,
                 function_bs=None,
                 function_extra_bs_ratio=0.1,
                 function_extra_bs_sampler='uniform',
                 function_extra_bs_std=1.,
                 functional_gradient=None,
                 force_fullrank=True,
                 fullrank_diag_weight=1.0,
                 pinverse_solve_iters=1,
                 pinverse_hidden_size=100,
                 use_jac_regularization=False,
                 pinverse_optimizer=None,
                 parameterization='network',
                 loss_type="classification",
                 voting="soft",
                 par_vi="svgd",
                 optimizer=None,
                 logging_network=False,
                 logging_training=False,
                 logging_evaluate=False,
                 config: TrainerConfig = None,
                 name="HyperNetwork"):
        """
        Args:
            Args for the generated parametric network
            ====================================================================
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format 
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            fc_layer_params (tuple[tuple]): a tuple of tuples where each tuple
                takes a format ``(FC layer sizes. use_bias)``, where 
                ``use_bias`` is optional.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            last_layer_param (tuple): an optional tuple of the format
                ``(size, use_bias)``, where ``use_bias`` is optional,
                it appends an additional layer at the very end. 
                Note that if ``last_activation`` is specified, 
                ``last_layer_param`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_param``. Note that if
                ``last_layer_param`` is not None, ``last_activation`` has to be
                specified explicitly.

            Args for the generator
            ====================================================================
            noise_dim (int): dimension of noise
            hidden_layers (tuple): size of hidden layers.
            use_fc_bn (bool): whether use batnch normalization for fc layers.
            num_particles (int): number of sampling particles
            entropy_regularization (float): weight of entropy regularization
            parameterization (str): choice of parameterization for the
                hypernetwork. Choices are [``network``, ``layer``].
                A parameterization of ``network`` uses a single generator to
                generate all the weights at once. A parameterization of ``layer``
                uses one generator for each layer of output parameters.

            Args for training and testing
            ====================================================================
            loss_type (str): loglikelihood type for the generated functions,
                types are [``classification``, ``regression``]
            voting (str): types of voting results from sampled functions,
                types are [``soft``, ``hard``]
            par_vi (str): types of particle-based methods for variational inference,
                types are [``svgd``, ``svgd2``, ``svgd3``, ``gfsf``]
            optimizer (torch.optim.Optimizer): The optimizer for training.
            logging_network (bool): whether logging the archetectures of networks.
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            config (TrainerConfig): configuration for training
            name (str):
        """
        param_net = ParamNetwork(
            input_tensor_spec=input_tensor_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            last_layer_param=last_layer_param,
            last_activation=last_activation)

        gen_output_dim = param_net.param_length
        noise_spec = TensorSpec(shape=(noise_dim, ))

        assert parameterization in ['network', 'layer'], "Hypernetwork " \
                "can only be parameterized by \"network\" or \"layer\" " \
                "generators"
        
        if parameterization == 'network':
            if functional_gradient:
                net = ReluMLP(
                    noise_spec,
                    hidden_layers=hidden_layers,
                    output_size=gen_output_dim,
                    name='Generator')
            else:
                net = EncodingNetwork(
                    noise_spec,
                    fc_layer_params=hidden_layers,
                    use_fc_bn=use_fc_bn,
                    last_layer_size=gen_output_dim,
                    last_activation=math_ops.identity,
                    name="Generator")
        else:
            net = ParamLayers(
                noise_dim=noise_dim,
                particles=num_particles,
                input_tensor_spec=input_tensor_spec,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                last_layer_param=last_layer_param,
                last_activation=math_ops.identity,
                hidden_layers=hidden_layers,
                activation=activation,
                use_fc_bn=use_fc_bn,
                use_bias=True,
                optimizer=optimizer,
                name="Generator")

        if logging_network:
            logging.info("Generated network")
            logging.info("-" * 68)
            logging.info(param_net)

            logging.info("Generator network")
            logging.info("-" * 68)
            logging.info(net)

        if par_vi == 'svgd':
            par_vi = 'svgd3'

        if function_vi:
            assert par_vi in ('svgd2', 'svgd3', 'gfsf', 'minmax'), (
                "Function_vi is not support for par_vi method: %s" % par_vi)
            assert function_bs is not None, (
                "Need to specify batch_size of function outputs.")
            assert function_extra_bs_sampler in ('uniform', 'normal'), (
                "Unsupported sampling type %s for extra training batch" %
                (function_extra_bs_sampler))
            self._function_extra_bs = math.ceil(
                function_bs * function_extra_bs_ratio)
            self._function_extra_bs_sampler = function_extra_bs_sampler
            self._function_extra_bs_std = function_extra_bs_std
            critic_input_dim = (
                function_bs + self._function_extra_bs) * last_layer_param[0]
        else:
            critic_input_dim = gen_output_dim

        super().__init__(train_state_spec=(), optimizer=optimizer, name=name) 

        self._generator = Generator(
            gen_output_dim,
            noise_dim=noise_dim,
            net=net,
            entropy_regularization=entropy_regularization,
            par_vi=par_vi,
            critic_input_dim=critic_input_dim,
            critic_hidden_layers=critic_hidden_layers,
            critic_relu_mlp=functional_gradient,
            critic_iter_num=critic_iter_num,
            critic_l2_weight=critic_l2_weight,
            functional_gradient=functional_gradient,
            force_fullrank=force_fullrank,
            fullrank_diag_weight=fullrank_diag_weight,
            pinverse_solve_iters=pinverse_solve_iters,
            pinverse_hidden_size=pinverse_hidden_size,
            use_jac_regularization=use_jac_regularization,
            critic_optimizer=critic_optimizer,
            pinverse_optimizer=pinverse_optimizer,
            optimizer=None,
            name=name)
        
        self._param_net = param_net
        self._num_particles = num_particles
        self._entropy_regularization = entropy_regularization
        self._train_loader = None
        self._test_loader = None
        self._use_fc_bn = use_fc_bn
        self._loss_type = loss_type
        self._parameterization = parameterization
        self._function_vi = function_vi
        self._functional_gradient = functional_gradient
        self._logging_training = logging_training
        self._logging_evaluate = logging_evaluate
        self._config = config
        assert (voting in ['soft',
                           'hard']), ('voting only supports "soft" and "hard"')
        self._voting = voting
        if loss_type == 'classification':
            self._loss_func = classification_loss
            self._vote = self._classification_vote
        elif loss_type == 'regression':
            self._loss_func = regression_loss
            self._vote = self._regression_vote
        else:
            raise ValueError("Unsupported loss_type: %s" % loss_type)

    def set_data_loader(self, train_loader, test_loader=None, outlier=None):
        """ Set data loaders for training, testing, and uncertainty
            quantification 
        """
        self._train_loader = train_loader
        self._test_loader = test_loader
        if self._entropy_regularization is None:
            self._entropy_regularization = 1/ len(train_loader)
        if outlier is not None:
            assert isinstance(outlier, tuple), "outlier dataset must be " \
                "provided in the format (outlier_train, outlier_test)"
            self._outlier_train = outlier[0]
            self._outlier_test = outlier[1]
        else: 
            self._outlier_train = self._outlier_test = None
 
    def set_num_particles(self, num_particles):
        """Set the number of particles to sample through one forward
        pass of the hypernetwork. """
        self._num_particles = num_particles
    
    @property
    def num_particles(self):
        return self._num_particles

    def sample_parameters(self, noise=None, num_particles=None, training=True):
        "Sample parameters for an ensemble of networks." ""
        if noise is None and num_particles is None:
            num_particles = self._num_particles
        if self._functional_gradient:
            output, _ = self._generator._predict(batch_size=num_particles)
        else:
            output = self._generator.predict_step(
                noise=noise, batch_size=num_particles,
                training=training).output
        return output

    def predict_step(self, inputs, params=None, num_particles=None,
                     state=None):
        """Predict ensemble outputs for inputs using the hypernetwork model.
        
        Args:
            inputs (Tensor): inputs to the ensemble of networks.
            params (Tensor): parameters of the ensemble of networks,
                if None, will resample.
            num_particles (int): size of sampled ensemble.
            state: not used.

        Returns:
            AlgorithmStep: outputs with shape (batch_size, output_dim)
        """
        if params is None:
            params = self.sample_parameters(num_particles=num_particles)      
            if self._functional_gradient is not None:
                if self._generator._use_jac_regularization:
                    params, _ = params
        self._param_net.set_parameters(params)
        outputs, _ = self._param_net(inputs)
        print (outputs.shape)
        return AlgStep(output=outputs, state=(), info=())
     
    def train_iter(self, num_particles=None, state=None):
        """ Perform a single (iteration) epoch of training"""
        assert self._train_loader is not None, "Must set data_loader first"
        alf.summary.increment_global_counter()
        with record_time("time/train"):
            loss = 0.
            pinverse_loss = 0
            if self._loss_type == 'classification':
                avg_acc = []
            for batch_idx, (data, target) in enumerate(self._train_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                alg_step = self.train_step((data, target),
                                           num_particles=num_particles,
                                           state=state)
                loss_info, params = self.update_with_gradient(alg_step.info)
                loss += loss_info.extra.generator.loss
                if self._functional_gradient:
                    pinverse_loss += loss_info.extra.pinverse
                if self._loss_type == 'classification':
                    avg_acc.append(alg_step.info.extra.generator.extra)
        acc = None
        if self._loss_type == 'classification':
            acc = torch.as_tensor(avg_acc).mean() * 100
        if self._logging_training:
            if self._loss_type == 'classification':
                logging.info("Avg acc: {}".format(acc))
            logging.info("Cum loss: {}".format(loss))
            if pinverse_loss != 0:
                pinverse_loss /= batch_idx
                logging.info("Avg pinverse loss: {}".format(pinverse_loss))
        self.summarize_train(loss_info, params, cum_loss=loss, avg_acc=acc)
        return batch_idx + 1

    def train_step(self,
                   inputs,
                   num_particles=None,
                   entropy_regularization=None,
                   state=None):
        """Perform one batch of training computation.

        Args:
            inputs (nested Tensor): input training data. 
            num_particles (int): number of sampled particles. 
            state: not used

        Returns:
            AlgorithmStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
        if num_particles is None:
            num_particles = self._num_particles
        if entropy_regularization is None:
            entropy_regularization = self._entropy_regularization
        
        if self._function_vi:
            data, target = inputs
            return self._generator.train_step(
                inputs=None,
                loss_func=functools.partial(self._function_neglogprob,
                                            target.view(-1)),
                batch_size=num_particles,
                entropy_regularization=entropy_regularization,
                transform_func=functools.partial(self._function_transform,
                                                 data),
                state=())
        else:
            return self._generator.train_step(
                inputs=None,
                loss_func=functools.partial(self._neglogprob, inputs),
                batch_size=num_particles,
                entropy_regularization=entropy_regularization,
                state=())
    
    def _function_transform(self, data, params):
        """
        Transform the generator outputs to its corresponding function values
        evaluated on the training batch. Used when function_vi is True.
        Args:
            data (torch.Tensor): training batch input.
            params: tensor params or tuple of tensors (params, extra_samples)
                - params: of shape ``[D]`` or ``[B, D]``, sampled outputs 
                    from the generator
                - extra_samples: sampled extra data
        Returns:
            outputs (torch.Tensor): outputs of param_net under params
                evaluated on data.
            density_outputs (torch.Tensor): outputs of param_net under
                params evaluated on sampled extra data.
            extra_samples (torch.Tensor): sampled extra data.
        """
        # sample extra data
        if isinstance(params, tuple):
            params, extra_samples = params
        else:
            sample = data[-self._function_extra_bs:]
            noise = torch.zeros_like(sample)
            if self._function_extra_bs_sampler == 'uniform':
                noise.uniform_(0., 1.)
            else:
                noise.normal_(mean=0., std=self._function_extra_bs_std)
            extra_samples = sample + noise

        num_particles = params.shape[0]
        self._param_net.set_parameters(params)
        aug_data = torch.cat([data, extra_samples], dim=0)
        aug_outputs, _ = self._param_net(aug_data)  # [B+b, P, D]

        outputs = aug_outputs[:data.shape[0]]  # [B, P, D]
        outputs = outputs.transpose(0, 1)
        outputs = outputs.view(num_particles, -1)  # [P, B * D]

        density_outputs = aug_outputs[-extra_samples.shape[0]:]  # [b, P, D]
        density_outputs = density_outputs.transpose(0, 1)
        density_outputs = density_outputs.view(num_particles, -1)

        return outputs, density_outputs, extra_samples

    def _function_neglogprob(self, targets, outputs):
        """
        Function computing negative log_prob loss for function outputs.
        Used when function_vi is True.
        Args:
            targets (torch.Tensor): target values of the training batch.
            outputs (torch.Tensor): function outputs to evaluate the loss.
        Returns:
            negative log_prob for outputs evaluated on current training batch.
        """
        num_particles = outputs.shape[0]
        targets = targets.unsqueeze(0).expand(num_particles, *targets.shape)

        return self._loss_func(outputs, targets)

    def _neglogprob(self, inputs, params):
        """
        Function computing negative log_prob loss for generator outputs.
        Used when function_vi is False.
        Args:
            inputs (torch.Tensor): (data, target) of training batch.
            params (torch.Tensor): generator outputs to evaluate the loss.
        Returns:
            negative log_prob for params evaluated on current training batch.
        """
        self._param_net.set_parameters(params)
        num_particles = params.shape[0]
        data, target = inputs
        output, _ = self._param_net(data)  # [B, P, D]
        target = target.unsqueeze(1).expand(*target.shape[:1], num_particles,
                                            *target.shape[1:])
        return self._loss_func(output, target)

    def evaluate(self, num_particles=None):
        """Evaluate on a randomly drawn network. """

        assert self._test_loader is not None, "Must set test_loader first."
        logging.info("==> Begin testing")
        if self._use_fc_bn:
            self._generator.eval()
        params = self.sample_parameters(num_particles=num_particles)
        #if self._functional_gradient:
        #    params, _ = params
        self._param_net.set_parameters(params)
        if self._use_fc_bn:
            self._generator.train()
        with record_time("time/test"):
            if self._loss_type == 'classification':
                test_acc = 0.
            test_loss = 0.
            for i, (data, target) in enumerate(self._test_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                output, _ = self._param_net(data)  # [B, N, D]
                loss, extra = self._vote(output, target)
                if self._loss_type == 'classification':
                    test_acc += extra.item()
                test_loss += loss.loss.item()

        if self._loss_type == 'classification':
            test_acc /= len(self._test_loader.dataset)
            alf.summary.scalar(name='eval/test_acc', data=test_acc * 100)
        if self._logging_evaluate:
            if self._loss_type == 'classification':
                logging.info("Test acc: {}".format(test_acc * 100))
            logging.info("Test loss: {}".format(test_loss))
        alf.summary.scalar(name='eval/test_loss', data=test_loss)

    def eval_uncertainty(self, num_particles=None):
        # Soft voting for now
        if num_particles is None:
            num_particles = 100
        params = self.sample_parameters(num_particles=num_particles)
       # if self._functional_gradient:
       #     params, _ = params
        if self._generator._par_vi == 'minmax':
            params = params[0]
        self._param_net.set_parameters(params)
        with torch.no_grad():
            outputs = self._predict_dataset(self._test_loader, num_particles)
        probs = F.softmax(outputs, -1).mean(0)
        entropy = entropy_fn(probs.T.cpu().detach().numpy())
        with torch.no_grad():
            outputs_outlier = self._predict_dataset(self._outlier_test,
                                                    num_particles)
        probs_outlier = F.softmax(outputs_outlier, -1).mean(0)
        entropy_outlier = entropy_fn(probs_outlier.T.cpu().detach().numpy())
        auroc_entropy = self._auc_score(entropy, entropy_outlier)
        logging.info("AUROC score: {}".format(auroc_entropy))

        if auroc_entropy >= 0.98:
            torch.save(outputs_outlier, 'probs_outlier_{}.pt'.format(auroc_entropy))
            torch.save(outputs, 'probs_inlier_{}.pt'.format(auroc_entropy))

    def _classification_vote(self, output, target):
        """ensmeble the ooutputs from sampled classifiers."""
        num_particles = output.shape[1]
        probs = F.softmax(output, dim=-1)  # [B, N, D]
        if self._voting == 'soft':
            pred = probs.mean(1).cpu()  # [B, D]
            vote = pred.argmax(-1)
        elif self._voting == 'hard':
            pred = probs.argmax(-1).cpu()  # [B, N, 1]
            vote = []
            for i in range(pred.shape[0]):
                values, counts = torch.unique(
                    pred[i], sorted=False, return_counts=True)
                modes = (counts == counts.max()).nonzero()
                label = values[torch.randint(len(modes), (1, ))]
                vote.append(label)
            vote = torch.as_tensor(vote, device='cpu')
        correct = vote.eq(target.cpu().view_as(vote)).float().cpu().sum()
        target = target.unsqueeze(1).expand(*target.shape[:1], num_particles,
                                            *target.shape[1:])
        loss = classification_loss(output, target)
        return loss, correct

    def _regression_vote(self, output, target):
        """ensemble the outputs for sampled regressors."""
        num_particles = output.shape[1]
        pred = output.mean(1)  # [B, D]
        loss = regression_loss(pred, target)
        target = target.unsqueeze(1).expand(*target.shape[:1], num_particles,
                                            *target.shape[1:])
        total_loss = regression_loss(output, target)
        return loss, total_loss

    def _auc_score(self, inliers, outliers):
        """
        Computes the AUROC score w.r.t network outputs. the ROC (curve) plots
        true positive rate against false positive rate. Thus the area under
        this curve gives the degree of separability between two dataset. 
        An AUROC score of 1.0 means that the classifier means that the
        classifier can totally discriminate between the two input datasets
        
        Args: 
            inliers (np.array): set of predictions on inlier (training) data
            outliers (np.array): set of predictions on outlier data
        
        Returns:
            AUROC score 
        """
        y_true = np.array([0] * len(inliers) + [1] * len(outliers))
        y_score = np.concatenate([inliers, outliers])
        return roc_auc_score(y_true, y_score)
    
    def _predict_dataset(self, testset, predictor_size=None):
        """
        Computes predictions for an input dataset. 

        Args: 
            testset (iterable): dataset for which to get predictions
            predictor_size (int): optional parameter indicating how many 
                predictors are used in an ensemble. Useful for nonstandard
                implementations.
        Returns:
            model_outputs (torch.tensor): a tensor of shape [N, S, D] where
            N refers to the number of predictors, S is the number of data
            points, and D is the output dimensionality. 
        """
        if hasattr(testset.dataset, 'dataset'):
            cls = len(testset.dataset.dataset.classes)
        else:
            cls = len(testset.dataset.classes)
        outputs = []
        for batch, (data, target) in enumerate(testset):
            data = data.to(alf.get_default_device())
            target = target.to(alf.get_default_device())
            output, _ = self._param_net(data)
            if output.dim() == 2:
                output = output.unsqueeze(1)
            output = output.transpose(0, 1)
            outputs.append(output)
        model_outputs = torch.cat(outputs, dim=1)  # [N, B, D]
        return model_outputs

    def summarize_train(self, loss_info, params, cum_loss=None, avg_acc=None):
        """Generate summaries for training & loss info after each gradient update.
        The default implementation of this function only summarizes params
        (with grads) and the loss. An algorithm can override this for additional
        summaries. See ``RLAlgorithm.summarize_train()`` for an example.
        Args:
            experience (nested Tensor): samples used for the most recent
                ``update_with_gradient()``. By default it's not summarized.
            train_info (nested Tensor): ``AlgStep.info`` returned by either
                ``rollout_step()`` (on-policy training) or ``train_step()``
                (off-policy training). By default it's not summarized.
            loss_info (LossInfo): loss
            params (list[Parameter]): list of parameters with gradients
        """
        if self._config.summarize_grads_and_vars:
            summary_utils.summarize_variables(params)
            summary_utils.summarize_gradients(params)
        if self._config.debug_summaries:
            summary_utils.summarize_loss(loss_info)
        if cum_loss is not None:
            alf.summary.scalar(name='train_epoch/neglogprob', data=cum_loss)
        if avg_acc is not None:
            alf.summary.scalar(name='train_epoch/avg_acc', data=avg_acc)
