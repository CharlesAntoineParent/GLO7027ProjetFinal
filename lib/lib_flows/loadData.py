"""Module that define data loading flows."""


## Import
import torchvision

from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from sklearn.model_selection import train_test_split


## Data loading flows definition
def load_CUB200_flow(dataset, transformation, split):
    train_path = dataset['train_path']
    test_path = dataset['test_path']

    if split == 0:
        dataset_train = torchvision.datasets.ImageFolder(root=train_path, transform=transformation)
        dataset_test = torchvision.datasets.ImageFolder(root=test_path, transform=transformation)

        return [dataset_train, dataset_test]

    else:
        dataset_trainVal = torchvision.datasets.ImageFolder(root=train_path, transform=transformation)
        dataset_test = torchvision.datasets.ImageFolder(root=test_path, transform=transformation)
        dataset_train, dataset_val = train_test_split(dataset_trainVal, test_size=split, random_state=0)

        return [dataset_train, dataset_test, dataset_val]

def load_MNIST_flow(dataset, transformation, split):
    path = dataset['data_separated_path']

    if split == 0:
        dataset_train = MNIST(path, train=True, download=False, transform=transformation)
        dataset_test = MNIST(path, train=False, download=False, transform=transformation)

        return [dataset_train, dataset_test]

    else:
        dataset_trainVal = MNIST(path, train=True, download=False, transform=transformation)
        dataset_test = MNIST(path, train=False, download=False, transform=transformation)
        dataset_train, dataset_val = train_test_split(dataset_trainVal, test_size=split, random_state=0)

        return [dataset_train, dataset_test, dataset_val]