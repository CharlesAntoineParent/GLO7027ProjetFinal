"""Module that define model initialization flows."""


## Import
import torch
import torchvision
import numpy as np


## Model initialization flows definition
def initialize_resnet18_flow(initialization, dataset, network = None):
    if initialization == "pretrained":
        model = torchvision.models.resnet18(True)
    else:
        model = torchvision.models.resnet18(False)
        model.apply(initialization)

    model.fc = torch.nn.Linear(512, dataset['nb_classes'])
    model.type = "Classic"

    return model

def initialize_classicAutoEncoder_HalfHalf_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 1.5, 1.5)
    model.apply(initialization)

    return model