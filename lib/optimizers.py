"""Module defining optimizer."""


## Import
import torch


## Optimizers definition
def Adam_optimizer(parameters, hyperparameters_optimizer):

    return torch.optim.Adam(parameters, **hyperparameters_optimizer)