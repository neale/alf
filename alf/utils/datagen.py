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
"""Set of methods for loading datasets for supervised learning.
Adapted from the following:

https://github.com/neale/HyperGAN/blob/master/datagen.py
"""
import numpy as np
import random
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Subset


class TestDataSet(torch.utils.data.Dataset):
    """Test dataset for Bayesian linear regression tests

    initializes a dataset of ``size`` pairs (X, Y). 
    inputs X are drawn from a standard normal distribution of dimension 
    ``input_dim``. The targets Y are computed as:
        :math:`y = w^T x + e`. Where e is drawn from a normal distribution, 
    and w is drawn from a uniform distribution.
    """

    def __init__(self, input_dim=3, output_dim=1, size=1000, weight=None):
        """
        Args: 
            input_dim (int): dimensionality of samples X.
            output_dim (int): dimensionality of targets Y.
            size (int): number of samples.
            weight (torch.tensor): optional weight w of shape
                [``input_dim``, ``output_dim``].
        """
        self._features = torch.randn(size, input_dim)
        if weight is None:
            self._weight = torch.rand(input_dim, output_dim) + 5.
        else:
            self._weight = weight
        noise = torch.randn(size, output_dim)
        self._values = self._features @ self._weight + noise

    def __getitem__(self, index):
        return self._features[index], self._values[index]

    def __len__(self):
        return len(self._features)

    def get_features(self):
        """Returns a tensor of the input samples."""
        return self._features

    def get_targets(self):
        """Returns a tensor of the target values."""
        return self._values


def load_test(train_bs=50, test_bs=10, num_workers=0):
    """Loads the ``TestDataset`` into a pytorch dataloader.

    Args:
        train_bs (int): training batch size.
        test_bs (int): testing batch size.
        num_workers (int): number of processes to allocate for loading data.

    Returns:
        train_loader (torch.utils.data.DataLoader): training data loader.
        test_loader (torch.utils.data.DataLoader): test data loader.
    """
    input_dim = 3
    output_dim = 1
    weight = torch.rand(input_dim, output_dim) + 5.
    trainset = TestDataSet(
        input_dim=input_dim, output_dim=output_dim, size=1000, weight=weight)
    testset = TestDataSet(
        input_dim=input_dim, output_dim=output_dim, size=500, weight=weight)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_bs, shuffle=True, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        trainset, batch_size=test_bs, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader


class TestNClassDataSet(torch.utils.data.Dataset):
    """Test dataset for 2 and 4 class classification tests.
    
    Classes are given by the number of components in a mixture model. Each
    component is an independent standard normal distribution. A point x
    sampled from component i is given label i. Tests use mixture models with
    2 or 4 components.
    """

    def __init__(self, num_classes=2, size=200):
        """
        Args:
            num_classes (int): number of mixture components (classes).
                ``num_classes`` must be 2 or 4.
            size (int): dataset size: number of sampled points per class.
        """
        self.num_classes = num_classes
        assert num_classes in [2, 4], "``num_classes`` must be set to 2 or 4,"\
            " other values not supported."
        if num_classes == 4:
            means = [(2., 2.), (-2., 2.), (2., -2.), (-2., -2.)]
        else:
            means = [(2., 2.), (-2., -2.)]
        data = torch.zeros(size, 2)
        labels = torch.zeros(size)
        size = size // len(means)
        for i, (x, y) in enumerate(means):
            dist = torch.distributions.Normal(torch.tensor([x, y]), .3)
            samples = dist.sample([size])
            data[size * i:size * (i + 1)] = samples
            labels[size * i:size * (i + 1)] = torch.ones(len(samples)) * i

        self._features = data
        self._labels = labels.long()

    def __getitem__(self, index):
        return self._features[index], self._labels[index]

    def __len__(self):
        return len(self._features)

    def get_features(self):
        """Returns a tensor of the input samples."""
        return self._features

    def get_targets(self):
        """Returns a tensor of the target values."""
        return self._labels


class Test8GaussiansDataSet(torch.utils.data.Dataset):
    def __init__(self, size=10000):
        scale = 2.
        centers = [((1, 0), 0), ((-1, 0), 1), ((0, 1), 2), ((0, -1), 3),
                   ((1. / np.sqrt(2), 1. / np.sqrt(2)), 4),
                   ((1. / np.sqrt(2), -1. / np.sqrt(2)), 5),
                   ((-1. / np.sqrt(2), 1. / np.sqrt(2)), 6),
                   ((-1. / np.sqrt(2), -1. / np.sqrt(2)), 7)]
        centers = [((scale * x[0], scale * x[1]), y) for x, y in centers]
        samples = []
        labels = []
        for _ in range(size):
            point = np.random.randn(2) * .02
            center, label = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            samples.append(point)
            labels.append(label)
        samples = torch.tensor(samples)
        samples /= 1.414  # stdev
        self._features = samples.float()
        self._labels = torch.tensor(labels).long()

    def __getitem__(self, index):
        return self._features[index], self._labels[index]

    def __len__(self):
        return len(self._features)

    def get_features(self):
        return self._features

    def get_targets(self):
        return self._labels


def load_nclass_test(num_classes,
                     train_size,
                     test_size,
                     train_bs=100,
                     test_bs=100,
                     num_workers=0):
    """Loads the ``TestNClassDataset`` into a pytorch dataloader.

    Args:
        train_size (int): number of samples to generate for training set.
        test_size (int): number of samples to generate for test set.
        train_bs (int): training batch size.
        test_bs (int): testing batch size.
        num_workers (int): number of processes to allocate for loading data.

    Returns:
        train_loader (torch.utils.data.DataLoader): training data loader.
        test_loader (torch.utils.data.DataLoader): test data loader.
    """
    trainset = TestNClassDataSet(num_classes, size=train_size)
    testset = TestNClassDataSet(num_classes, size=test_size)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_bs, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_bs, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader


def get_classes(target, labels):
    """Helper function to subclass a dataloader, i.e. select only given
        classes from target dataset.

    Args:
        target (torch.utils.data.Dataset): the dataset that should be filtered.
        labels (list[int]): list of labels to filter on.
    
    Returns:
        label_indices (list[int]): indices of examples with label in
            ``labels``. 
    """
    label_indices = []
    for i in range(len(target)):
        if target[i][1] in labels:
            label_indices.append(i)
    return label_indices


def load_mnist(label_idx=None,
               train_bs=100,
               test_bs=100,
               num_workers=0,
               scale=None,
               normalize=True):
    """ Loads the MNIST dataset. 
    
    Args:
        label_idx (list[int]): class indices to load from the dataset.
        train_bs (int): training batch size.
        test_bs (int): testing batch size. 
        num_workers (int): number of processes to allocate for loading data.
        
    Returns:
        train_loader (torch.utils.data.DataLoader): training data loader.
        test_loader (torch.utils.data.DataLoader): test data loader.
    """

    kwargs = {
        'num_workers': num_workers,
        'pin_memory': False,
        'drop_last': False
    }
    path = 'data_m/'

    if scale is None:
        scale = 28
    if normalize:
        normalize = transforms.Normalize((0.1307, ), (0.3081, ))
    else:
        normalize = transforms.Normalize((0, ), (1, ))

    data_transform = transforms.Compose(
        [transforms.Resize(scale),
         transforms.ToTensor(), normalize])

    trainset = datasets.MNIST(
        root=path, train=False, download=True, transform=data_transform)
    testset = datasets.MNIST(root=path, train=False, transform=data_transform)

    if label_idx is not None:
        trainset = Subset(trainset, get_classes(trainset, label_idx))
        testset = Subset(testset, get_classes(testset, label_idx))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_bs, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_bs, shuffle=False, **kwargs)

    return train_loader, test_loader


def load_cifar10(label_idx=None,
                 train_bs=100,
                 test_bs=100,
                 num_workers=0,
                 scale=None,
                 normalize=True):
    """ Loads the CIFAR-10 dataset.

    Args:
        label_idx (list[int]): classes to be loaded from the dataset.
        train_bs (int): training batch size.
        test_bs (int): testing batch size. 
        num_workers (int): number of processes to allocate for loading data.
        
    Returns:
        train_loader (torch.utils.data.DataLoader): training data loader.
        test_loader (torch.utils.data.DataLoader): test data loader.
    """
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': False,
        'drop_last': False
    }
    path = 'data_c10/'

    if scale is None:
        scale = 32
    if normalize:
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    else:
        normalize = transforms.Normalize((0, 0, 0), (1, 1, 1))

    data_transform = transforms.Compose(
        [transforms.Resize(scale),
         transforms.ToTensor(), normalize])

    trainset = datasets.CIFAR10(
        root=path, train=True, download=True, transform=data_transform)

    testset = datasets.CIFAR10(
        root=path, train=False, download=True, transform=data_transform)

    if label_idx is not None:
        trainset = Subset(trainset, get_classes(trainset, label_idx))
        testset = Subset(testset, get_classes(testset, label_idx))

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_bs, shuffle=False, **kwargs)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_bs, shuffle=True, **kwargs)

    return train_loader, test_loader
