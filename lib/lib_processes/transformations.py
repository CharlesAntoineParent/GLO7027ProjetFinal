"""Module defining parameters of the possible transformations."""


## Import
import torchvision
import torch


## Utils
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

        
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

normalize_and_add_gaussian_noise_with_MNIST = torchvision.transforms.Compose(
  [torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    mean=(0.1307,), 
    std=(0.3081,)),
    AddGaussianNoise(0.1307, 0.3081)
  ])

normalize_with_celebA = torchvision.transforms.Compose(
  [torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    mean=(0.50612009, 0.42543493, 0.38282761), 
    std=(0.26589054, 0.24521921, 0.24127836))
  ])
