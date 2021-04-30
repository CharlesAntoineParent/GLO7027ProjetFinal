"""Module that define model initialization flows."""


## Import
import torch
import torchvision
import numpy as np


## Model initialization flows definition
def initialize_resnet18_flow(initialization, dataset, networks = None):
    if initialization == "pretrained":
        model = torchvision.models.resnet18(True)
    else:
        model = torchvision.models.resnet18(False)
        model.apply(initialization)

    model.fc = torch.nn.Linear(512, dataset['nb_classes'])

    return model

def initialize_classicAutoEncoder_flow(initialization, dataset, networks):
    shape = dataset['shape']
    capacity = np.prod(shape)
    models = []

    for network in networks:
        model =  network(capacity)
        model.apply(initialization)
        models.append(model)

    return models

def initialize_irma4_AutoEncoder_flow(initialization, dataset, networks):
    shape = dataset['shape']
    capacity = np.prod(shape)
    models = []

    for network in networks:
        if network.label == "irma.Linear":
            model =  network(capacity, 4)
        else:
            model = network(capacity)

        model.apply(initialization)
        models.append(model)

    return models

def initialize_irma8_AutoEncoder_flow(initialization, dataset, networks):
    shape = dataset['shape']
    capacity = np.prod(shape)
    models = []

    for network in networks:
        if network.label == "irma.Linear":
            model =  network(capacity, 8)
        else:
            model = network(capacity)

        model.apply(initialization)
        models.append(model)

    return models