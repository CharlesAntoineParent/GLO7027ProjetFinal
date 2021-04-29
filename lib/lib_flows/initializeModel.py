"""Module that define model initialization flows."""


## Import
import torch
import torchvision


## Model initialization flows definition
def initialize_resnet18_flow(initialization, dataset, network):
    if initialization == "pretrained":
        model = torchvision.models.resnet18(True)
    else:
        model = torchvision.models.resnet18(False)
        model.apply(initialization)

    model.fc = torch.nn.Linear(512, dataset['nb_classes'])

    return model

def initialize_customUNet_flow(initialization, dataset, network):
    model = network()
    model.apply(initialization)

    return model