"""Module defining loading processes."""


## Import
import torchvision

from sklearn.model_selection import train_test_split


## Loading processes definition
def load_CUB200(train_path, test_path, split, transformation):
    dataset_trainVal = torchvision.datasets.ImageFolder(root=train_path, transform=transformation)
    dataset_test = torchvision.datasets.ImageFolder(root=test_path, transform=transformation)
    dataset_train, dataset_val = train_test_split(dataset_trainVal, test_size=split, random_state=0)

    return dataset_train, dataset_val, dataset_test
