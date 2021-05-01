"""Module that define model initialization flows."""


## Import
import torch
import torchvision
import numpy as np


## Model initialization flows definition
def initialize_torchvision_model_flow(initialization, dataset, network):
    if initialization == "pretrained":
        if network == "resnet18":
            model = torchvision.models.resnet18(True)
    else:
        if network == "resnet18":
            model = torchvision.models.resnet18(False)
            model.apply(initialization)

    model.fc = torch.nn.Linear(512, dataset['nb_classes'])
    model.type = "Torchvision"

    return model

def initialize_autoencoder_base_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 1.1)
    model.apply(initialization)

    return model

def initialize_irma_autoencoder_base_base_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 1.1, 4)
    model.apply(initialization)

    return model