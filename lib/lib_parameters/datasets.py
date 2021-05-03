"""Module defining parameters of the possible datasets."""


## Parameters definition
data_CUB200 = {
  "data_path": "data_raw\\CUB200\\",
  "data_separated_path": "data_separated\\",
  "train_path": "data_separated\\CUB200\\train\\",
  "test_path": "data_separated\\CUB200\\test\\",
  "nb_classes": 200,
  "shape": [224, 224]
}

data_MNIST = {
  "data_path": "data_raw\\MNIST\\",
  "data_separated_path": "data_separated\\",
  "train_path": "data_separated\\MNIST\\train\\",
  "test_path": "data_separated\\MNIST\\test\\",
  "nb_classes": 10,
  "shape": [1, 28, 28]
}

data_CIFAR10 = {
  "data_path": "data_raw\\CIFAR10\\",
  "data_separated_path": "data_separated\\",
  "train_path": "data_separated\\CIFAR10\\train\\",
  "test_path": "data_separated\\CIFAR10\\test\\",
  "nb_classes": 10,
  "shape": [3, 32, 32]
}