"""Module defining initializing processes."""


## Import
import torch
import torchvision


## Loading process definition
def initializing_process(dataset, process, initialization):
    return process(initialization, dataset['nb_classes'])

def initialize_resnet18(initialization, nb_classes):
    if initialization == "pretrained":
        model = torchvision.models.resnet18(True)
    else:
        model = torchvision.models.resnet18(False)
        model.apply(initialization)

    model.fc = torch.nn.Linear(512, nb_classes)

    return model
