# pylint:disable=no-member
"""Module defining losses."""


## Import
import torch


## Losses definition
def CrossEntropy_criterion(input_batch, target_batch):
    loss = torch.nn.CrossEntropyLoss()
  
    return loss(input_batch, target_batch)

def MSE_criterion(input_batch, target_batch):
    loss = torch.nn.MSELoss()

    return loss(input_batch, target_batch)

def contractive_criterion(input_batch, target_batch):
    output = input_batch[1]
    output_encoder = input_batch[0]
    output_encoder_size = output_encoder.size()

    ones_tensor = torch.ones(output_encoder_size)
    output_encoder.backward(ones_tensor, retain_graph=True)

    loss = torch.nn.MSELoss(output, target_batch)
    loss += torch.mean(pow(target_batch.grad, 2))
    
    return loss 