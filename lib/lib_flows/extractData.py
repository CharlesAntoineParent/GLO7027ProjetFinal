"""Module that define data extracting flows."""


## Import
import os

from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10


## Data extracting flows definition
def extract_MNIST_flow(dataset):
    path = dataset['data_separated_path']

    MNIST(path, download=True)