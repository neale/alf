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

from absl.testing import parameterized
import torch

import alf
from alf.networks.relu_mlp import ReluMLP
from alf.tensor_specs import TensorSpec


def jacobian(y, x, create_graph=False):
    """It is from Adam Paszke's implementation:
    https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
    """
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(
            flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.

    return torch.stack(jac).reshape(y.shape + x.shape)


def jvp_autograd(x, y, v):
    """ Double backward jvp trick from:
  https://j-towns.github.io/2017/06/12/A-new-trick.html
  """
    grad_y = torch.zeros_like(y, requires_grad=True)
    dy = torch.autograd.grad(y, x, grad_outputs=grad_y, create_graph=True)
    dyT = dy[0].transpose(1, 0)
    jvpT = torch.autograd.grad(dyT, grad_y, grad_outputs=v, retain_graph=True)
    jvp = jvpT[0].transpose(1, 0)

    return jvp


class ReluMLPTest(parameterized.TestCase, alf.test.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(torch.max(abs(x - y))), eps)

    @parameterized.parameters(
        dict(hidden_layers=(2, )),
        dict(hidden_layers=(2, 3), batch_size=1),
        dict(hidden_layers=(2, 3, 4)),
    )
    def test_compute_jac(self, hidden_layers=(2, ), batch_size=2,
                         input_size=5):
        """
        Check that the input-output Jacobian computed by the direct(autograd-free)
        approach is consistent with the one computed by calling autograd.
        """
        spec = TensorSpec((input_size, ))
        mlp = ReluMLP(spec, output_size=4, hidden_layers=hidden_layers)

        # compute jac using direct approach
        x = torch.randn(batch_size, input_size, requires_grad=True)
        x1 = x.detach().clone()
        x1.requires_grad = True
        # z, state = mlp(x1, requires_jac=True)
        jac = mlp.compute_jac(x1)

        # compute jac using autograd
        y, _ = mlp(x)
        jac_ad = jacobian(y, x)
        jac2 = []
        for i in range(batch_size):
            jac2.append(jac_ad[i, :, i, :])
        jac2 = torch.stack(jac2, dim=0)

        self.assertArrayEqual(jac, jac2, 1e-6)

    @parameterized.parameters(
        dict(hidden_layers=(2, )),
        dict(hidden_layers=(2, 3), batch_size=1),
        dict(hidden_layers=(2, 3, 4)),
    )
    def test_compute_jac_diag(self,
                              hidden_layers=(2, ),
                              batch_size=2,
                              input_size=5):
        """
        Check that the diagonal of input-output Jacobian computed by
        the direct (autograd-free) approach is consistent with the one
        computed by calling autograd.
        """
        spec = TensorSpec((input_size, ))
        mlp = ReluMLP(spec, hidden_layers=hidden_layers)

        # compute jac diag using direct approach
        x = torch.randn(batch_size, input_size, requires_grad=True)
        x1 = x.detach().clone()
        x1.requires_grad = True
        jac_diag = mlp.compute_jac_diag(x1)

        # compute jac using autograd
        y, _ = mlp(x)
        jac = jacobian(y, x)
        jac_diag2 = []
        for i in range(batch_size):
            jac_diag2.append(torch.diag(jac[i, :, i, :]))
        jac_diag2 = torch.stack(jac_diag2, dim=0)

        self.assertArrayEqual(jac_diag, jac_diag2, 1e-6)

    @parameterized.parameters(
        dict(hidden_layers=(2, )),
        dict(hidden_layers=(2, 3), batch_size=1),
        dict(hidden_layers=(2, 3, 4)),
    )
    def test_compute_vjp(self, hidden_layers=(2, ), batch_size=2,
                         input_size=5):
        """
        Check that the vector-Jacobian product computed by the direct(autograd-free)
        approach is consistent with the one computed by calling autograd.
        """
        output_size = 4
        spec = TensorSpec((input_size, ))
        mlp = ReluMLP(
            spec, output_size=output_size, hidden_layers=hidden_layers)

        # compute vjp using direct approach
        x = torch.randn(batch_size, input_size, requires_grad=True)
        vec = torch.randn(batch_size, output_size)
        x1 = x.detach().clone()
        x1.requires_grad = True
        vjp, _ = mlp.compute_vjp(x1, vec)

        # # compute jac using autograd
        y, _ = mlp(x)
        vjp2 = torch.autograd.grad(y, x, grad_outputs=vec)[0]

        self.assertArrayEqual(vjp, vjp2, 1e-6)

    @parameterized.parameters(
        dict(hidden_layers=(4, )),
        dict(hidden_layers=(4, 6), batch_size=1),
        dict(hidden_layers=(4, 6, 8)),
    )
    def test_compute_vjp_partial(self,
                                 hidden_layers=(2, ),
                                 batch_size=2,
                                 input_size=5):
        """
        Check that the vector-Jacobian product computed by the direct(autograd-free)
        approach is consistent with the one computed by calling autograd.
        """
        output_size = 6
        spec = TensorSpec((input_size, ))
        mlp = ReluMLP(
            spec,
            output_size=output_size,
            hidden_layers=hidden_layers,
            head_size=(2, output_size - 2))

        # compute vjp using direct approach
        x = torch.randn(batch_size, input_size, requires_grad=True)
        vec = torch.randn(batch_size, output_size)
        x1 = x.detach().clone()
        x1.requires_grad = True
        vjp_0, _ = mlp.compute_vjp_partial(x1, vec[:, :2], partial_idx=0)
        vjp_1, _ = mlp.compute_vjp_partial(x1, vec[:, 2:], partial_idx=1)
        vjp_3, _ = mlp.compute_vjp_partial(x1, vec, partial_idx=-1)

        # # compute jac using autograd
        y, _ = mlp(x)
        vjp2_0 = torch.autograd.grad(
            y[:, :2], x, grad_outputs=vec[:, :2], retain_graph=True)[0]
        vjp2_1 = torch.autograd.grad(
            y[:, 2:], x, grad_outputs=vec[:, 2:], retain_graph=True)[0]
        vjp2_3 = torch.autograd.grad(
            y, x, grad_outputs=vec, retain_graph=True)[0]

        self.assertArrayEqual(vjp_0, vjp2_0, 1e-6)
        self.assertArrayEqual(vjp_1, vjp2_1, 1e-6)
        self.assertArrayEqual(vjp_3, vjp2_3, 1e-6)

    @parameterized.parameters(
        dict(hidden_layers=(2, )),
        dict(hidden_layers=(2, 3), batch_size=1),
        dict(hidden_layers=(2, 3, 4)),
    )
    def test_compute_jvp(self, hidden_layers=(2, ), batch_size=2,
                         input_size=5):
        """
        Check that the vector-Jacobian product computed by the direct(autograd-free)
        approach is consistent with the one computed by calling autograd.
        """
        output_size = 4
        spec = TensorSpec((input_size, ))
        mlp = ReluMLP(
            spec, output_size=output_size, hidden_layers=hidden_layers)

        # compute vjp using direct approach
        x = torch.randn(batch_size, input_size, requires_grad=True)
        vec = torch.randn(batch_size, input_size, requires_grad=True).t()
        x1 = x.detach().clone()
        x1.requires_grad = True
        jvp, _ = mlp.compute_jvp(x1, vec)

        # # compute jac using autograd
        y, _ = mlp(x)
        jvp2 = jvp_autograd(x, y, vec)

        self.assertArrayEqual(jvp, jvp2, 1e-6)

    @parameterized.parameters(
        dict(hidden_layers=(4, )),
        dict(hidden_layers=(4, 6), batch_size=1),
        dict(hidden_layers=(4, 6, 8)),
    )
    def test_compute_jvp_partial(self,
                                 hidden_layers=(2, ),
                                 batch_size=2,
                                 input_size=5):
        """
        Check that the vector-Jacobian product computed by the direct(autograd-free)
        approach is consistent with the one computed by calling autograd.
        """
        output_size = 6
        spec = TensorSpec((input_size, ))
        mlp = ReluMLP(
            spec,
            output_size=output_size,
            hidden_layers=hidden_layers,
            head_size=(2, output_size - 2))

        # compute vjp using direct approach
        x = torch.randn(batch_size, input_size, requires_grad=True)
        vec = torch.randn(batch_size, input_size, requires_grad=True).t()
        x1 = x.detach().clone()
        x1.requires_grad = True
        jvp_0, _ = mlp.compute_jvp_partial(x1, vec, partial_idx=0)
        jvp_1, _ = mlp.compute_jvp_partial(x1, vec, partial_idx=1)
        jvp_2, _ = mlp.compute_jvp_partial(x1, vec, partial_idx=-1)

        # # compute jac using autograd
        y, _ = mlp(x)
        jvp2_0 = jvp_autograd(x, y[:, :2], vec)
        jvp2_1 = jvp_autograd(x, y[:, 2:], vec)
        jvp2_2 = jvp_autograd(x, y, vec)

        self.assertArrayEqual(jvp_0, jvp2_0, 1e-6)
        self.assertArrayEqual(jvp_1, jvp2_1, 1e-6)
        self.assertArrayEqual(jvp_2, jvp2_2, 1e-6)


if __name__ == "__main__":
    alf.test.main()
