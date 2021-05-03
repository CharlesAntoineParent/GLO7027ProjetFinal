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

def initialize_autoencoder_low_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 0.7)
    model.apply(initialization)

    return model

def initialize_autoencoder_base_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 1)
    model.apply(initialization)

    return model

def initialize_autoencoder_high_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 1.3)
    model.apply(initialization)

    return model

def initialize_irma_autoencoder_low_low_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 0.7, 4)
    model.apply(initialization)

    return model

def initialize_irma_autoencoder_base_low_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 1, 4)
    model.apply(initialization)

    return model

def initialize_irma_autoencoder_high_low_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 1.3, 4)
    model.apply(initialization)

    return model

def initialize_irma_autoencoder_low_base_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 0.7, 6)
    model.apply(initialization)

    return model

def initialize_irma_autoencoder_base_base_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 1, 6)
    model.apply(initialization)

    return model

def initialize_irma_autoencoder_high_base_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 1.3, 6)
    model.apply(initialization)

    return model

def initialize_irma_autoencoder_low_high_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 0.7, 8)
    model.apply(initialization)

    return model

def initialize_irma_autoencoder_base_high_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 1, 8)
    model.apply(initialization)

    return model

def initialize_irma_autoencoder_high_high_flow(initialization, dataset, network):
    shape = dataset['shape']
    capacity = np.prod(shape)
    
    model = network(capacity, 1.3, 8)
    model.apply(initialization)

    return model