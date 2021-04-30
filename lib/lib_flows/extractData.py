"""Module that define data extracting flows."""


## Import
import os

from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10


## Data extracting flows definition
def extract_MNIST_flow(dataset):
    train_path = dataset['train_path']
    test_path = dataset['test_path']

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    MNIST(train_path, train=True, download=True)
    MNIST(test_path, train=False, download=True)