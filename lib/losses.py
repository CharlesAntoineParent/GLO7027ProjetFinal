"""Module defining losses."""


## Import
import torch


## Losses definition
def CrossEntropy_criterion(input_batch, target_batch):
    loss = torch.nn.CrossEntropyLoss()

    return loss(input_batch, target_batch)