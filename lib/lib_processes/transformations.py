"""Module defining parameters of the possible transformations."""


## Import
import torchvision


## Parameters definition
default_transformation = torchvision.transforms.Compose(
  [torchvision.transforms.ToTensor()]
)

normalize_with_CUB200 = torchvision.transforms.Compose(
  [torchvision.transforms.Resize([224, 224]),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    mean=[0.4750, 0.4917, 0.4242], 
    std=[0.2287, 0.2244, 0.2656])])

normalize_with_ImageNet = torchvision.transforms.Compose(
  [torchvision.transforms.Resize([224, 224]),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])])

normalize_with_MNIST = torchvision.transforms.Compose(
  [torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    mean=(0.1307,), 
    std=(0.3081,))
  ])
