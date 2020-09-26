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
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable
from scipy.stats import entropy as entropy_fn
from sklearn.metrics import roc_auc_score

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.config import TrainerConfig
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.algorithms.generator import Generator
from alf.algorithms.hypernetwork_networks import ParamNetwork
from alf.algorithms.hypernetwork_layer_generator import ParamLayers
from alf.algorithms.forward_network_algorithm import ForwardNetwork
from alf.networks import EncodingNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops, summary_utils
from alf.utils.summary_utils import record_time

HyperNetworkLossInfo = namedtuple("HyperNetworkLossInfo", ["loss", "extra"])


def classification_loss(output, target):
    pred = output.max(-1)[1]
    acc = pred.eq(target).float().mean(0)
    avg_acc = acc.mean()
    loss = F.cross_entropy(output.transpose(1, 2), target)
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


def neglogprob(inputs, param_net, loss_type, params):
    if loss_type == 'regression':
        loss_func = regression_loss
    elif loss_type == 'classification':
        loss_func = classification_loss
    else:
        raise ValueError("Unsupported loss_type: %s" % loss_type)
    
    param_net.set_parameters(params)
    particles = params.shape[0]
    data, target = inputs
    output, _ = param_net(data)  # [B, N, D]
    target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                        *target.shape[1:])
    return loss_func(output, target)

def neglogprob2(inputs, outputs, loss_type, params=None):
    """ 
    computes the NLL for the corresponding model. This version
    does not use the ``params`` argument, and takes predictions as inputs
    """
    if loss_type == 'regression':
        loss_func = regression_loss
    elif loss_type == 'classification':
        loss_func = classification_loss
    else:
        raise ValueError("Unsupported loss_type: %s" % loss_type)
    
    data, target = inputs
    particles = outputs.shape[1]
    target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                        *target.shape[1:])
    return loss_func(outputs, target)


@gin.configurable
class HyperNetwork(Algorithm):
    """HyperNetwork 

    HyperrNetwork algorithm maintains a generator that generates a set of 
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
                 particles=10,
                 entropy_regularization=1.,
                 parameterization='layer',
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
            particles (int): number of sampling particles
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
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)

        param_net = ParamNetwork(
            input_tensor_spec=input_tensor_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            last_layer_param=last_layer_param,
            last_activation=last_activation)

        gen_output_dim = param_net.param_length
        noise_spec = TensorSpec(shape=(noise_dim, ))
        self._parameterization = parameterization
        assert self._parameterization in ['network', 'layer'], "Hypernetwork " \
                "can only be parameterized by \"network\" or \"layer\" " \
                "generators"
        if self._parameterization == 'network':
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
                particles=particles,
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

        if par_vi == 'minmax':
            print (gen_output_dim)
            critic = ForwardNetwork(
                TensorSpec(shape=(gen_output_dim, )),
                conv_layer_params=None,
                fc_layer_params=(16, 16),
                activation=torch.nn.functional.relu,
                last_layer_param=gen_output_dim,
                last_activation=math_ops.identity,
                optimizer=alf.optimizers.Adam(lr=1e-4),
                name="Critic")
            self._d_iters = 5
        else:
            critic = None

        if logging_network:
            logging.info("Generated network")
            logging.info("-" * 68)
            logging.info(param_net)

            logging.info("Generator network")
            logging.info("-" * 68)
            logging.info(net)

        if par_vi == 'svgd':
            par_vi = 'svgd3'

        self._generator = Generator(
            gen_output_dim,
            noise_dim=noise_dim,
            net=net,
            entropy_regularization=entropy_regularization,
            par_vi=par_vi,
            optimizer=None,
            name=name)
        
        self._param_net = param_net
        self._critic = critic
        self._particles = particles
        self._entropy_regularization = entropy_regularization
        self._train_loader = None
        self._test_loader = None
        self._use_fc_bn = use_fc_bn
        self._loss_type = loss_type
        self._logging_training = logging_training
        self._logging_evaluate = logging_evaluate
        self._config = config
        assert (voting in ['soft', 'hard'
                           ]), ("voting only supports \"soft\" and \"hard\"")
        self._voting = voting
        if loss_type == 'classification':
            self._vote = self._classification_vote
        elif loss_type == 'regression':
            self._vote = self._regression_vote
        else:
            raise ValueError("Unsupported loss_type: %s" % loss_type)
    
    def set_data_loader(self, train_loader, test_loader=None, outlier=None):
        """Set data loadder for training and testing."""
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._outlier_train = outlier[0]
        self._outlier_test = outlier[1]
        self._entropy_regularization = 1 / len(train_loader)

    def set_particles(self, particles):
        """Set the number of particles to sample through one forward
        pass of the hypernetwork. """
        self._particles = particles

    @property
    def particles(self):
        return self._particles

    def sample_parameters(self, noise=None, particles=None, training=True):
        "Sample parameters for an ensemble of networks." ""
        if noise is None and particles is None:
            particles = self.particles
        generator_step = self._generator.predict_step(
            noise=noise, batch_size=particles, training=training)
        return generator_step.output

    def predict_step(self, inputs, params=None, particles=None, state=None):
        """Predict ensemble outputs for inputs using the hypernetwork model.
        
        Args:
            inputs (Tensor): inputs to the ensemble of networks.
            params (Tensor): parameters of the ensemble of networks,
                if None, will resample.
            particles (int): size of sampled ensemble.
            state: not used.

        Returns:
            AlgorithmStep: outputs with shape (batch_size, output_dim)
        """
        if params is None:
            params = self.sample_parameters(particles=particles)
        self._param_net.set_parameters(params)
        outputs, _ = self._param_net(inputs)
        return AlgStep(output=outputs, state=(), info=())

    def train_iter(self, particles=None, state=None):
        """Perform one epoch (iteration) of training."""

        assert self._train_loader is not None, "Must set data_loader first."
        alf.summary.increment_global_counter()
        with record_time("time/train"):
            loss = 0.
            if self._loss_type == 'classification':
                avg_acc = []
            for batch_idx, (data, target) in enumerate(self._train_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                alg_step = self.train_step((data, target),
                                           particles=particles,
                                           state=state)
                loss_info, params = self.update_with_gradient(alg_step.info)
                loss += loss_info.extra.generator.loss
                if self._loss_type == 'classification':
                    avg_acc.append(alg_step.info.extra.generator.extra)
        acc = None
        if self._loss_type == 'classification':
            acc = torch.as_tensor(avg_acc).mean() * 100
        if self._logging_training:
            if self._loss_type == 'classification':
                logging.info("Avg acc: {}".format(acc))
            logging.info("Cum loss: {}".format(loss))
        self.summarize_train(loss_info, params, cum_loss=loss, avg_acc=acc)
        return batch_idx + 1

    def train_step(self,
                   inputs,
                   particles=None,
                   entropy_regularization=None,
                   state=None):
        """Perform one batch of training computation.

        Args:
            inputs (nested Tensor): input training data. 
            particles (int): number of sampled particles. 
            state: not used

        Returns:
            AlgorithmStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
        params = self.sample_parameters(particles=particles)

        if entropy_regularization is None:
            entropy_regularization = self._entropy_regularization
        

        outputs = params
        loss_func = functools.partial(neglogprob, inputs, self._param_net,
            self._loss_type)

        if self._generator._par_vi == 'minmax':
            for i in range(self._d_iters):
                self._critic.predict_with_update(outputs, loss_func)
                outputs = self.sample_parameters(particles=particles)
            
            outputs = self.sample_parameters(particles=particles)
            critic_outputs = self._critic._net(outputs)[0]
            outputs = (outputs, critic_outputs.detach())
        
        return self._generator.train_step(
            inputs=None,
            loss_func=loss_func,
            outputs=outputs,
            entropy_regularization=entropy_regularization,
            state=())

    def evaluate(self, particles=None):
        """Evaluate on a randomly drawn network. """

        assert self._test_loader is not None, "Must set test_loader first."
        logging.info("==> Begin testing")
        if self._use_fc_bn:
            self._generator.eval()
        params = self.sample_parameters(particles=particles)
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

    def _classification_vote(self, output, target):
        """ensmeble the ooutputs from sampled classifiers."""
        particles = output.shape[1]
        probs = F.softmax(output, dim=-1)  # [B, N, D]
        if self._voting == 'soft':
            pred = probs.mean(1).cpu()  # [B, D]
            vote = pred.argmax(-1)
        elif self._voting == 'hard':
            pred = probs.argmax(-1).cpu()  # [B, N, 1]
            vote = pred.mode(1)[0]  # [B, 1]
        correct = vote.eq(target.cpu().view_as(vote)).float().cpu().sum()
        target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                            *target.shape[1:])
        loss = classification_loss(output, target)
        return loss, correct

    def _regression_vote(self, output, target):
        """ensemble the outputs for sampled regressors."""
        particles = output.shape[1]
        pred = output.mean(1)  # [B, D]
        loss = regression_loss(pred, target)
        target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                            *target.shape[1:])
        total_loss = regression_loss(output, target)
        return loss, total_loss

    def _spectral_norm(self, module):
        if 'weight' in module._parameters:
            torch.nn.utils.spectral_norm(module)

    def _auc_score(self, inliers, outliers):
        y_true = np.array([0] * len(inliers) + [1] * len(outliers))
        y_score = np.concatenate([inliers, outliers])
        return roc_auc_score(y_true, y_score)
    
    def _predict_dataset(self, testset, particles=None):
        if particles is None:
            particles = self.particles
        cls = len(testset.dataset.dataset.classes)
        model_outputs = torch.zeros(particles, len(testset.dataset), cls)
        for batch, (data, target) in enumerate(testset):
            data = data.to(alf.get_default_device())
            target = target.to(alf.get_default_device())
            output, _ = self._param_net(data)
            output = output.transpose(0, 1)
            model_outputs[:, batch*len(data): (batch+1)*len(data), :] = output
        return model_outputs

    def eval_uncertainty(self, particles=None):
        # Soft voting for now
        if particles is None:
            particles = 100
        params = self.sample_parameters(particles=particles)
        if self._generator._par_vi == 'minmax':
            params = params[0]
        self._param_net.set_parameters(params)
        with torch.no_grad():
            outputs = self._predict_dataset(
                self._test_loader,
                particles)
        probs = F.softmax(outputs, -1).mean(0)
        entropy = entropy_fn(probs.T.cpu().detach().numpy())
        with torch.no_grad():
            outputs_outlier = self._predict_dataset(
                self._outlier_test,
                particles)
        probs_outlier = F.softmax(outputs_outlier, -1).mean(0)
        entropy_outlier = entropy_fn(probs_outlier.T.cpu().detach().numpy())
        auroc_entropy = self._auc_score(entropy, entropy_outlier)
        logging.info("AUROC score: {}".format(auroc_entropy))

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
