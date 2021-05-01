# pylint:disable=no-member
"""Module defining losses."""


## Import
import torch

from torch.autograd import Variable


## Losses definition
def crossEntropy_criterion(input_batch, target_batch):
    loss = torch.nn.CrossEntropyLoss()
  
    return loss(input_batch, target_batch)

def mse_criterion(input_batch, target_batch):
    loss = torch.nn.MSELoss()

    return loss(input_batch, target_batch)

def contractive_criterion(input_batch, target_batch):
    lam = 1e-4
    weigths = input_batch[0]
    output_encoder = input_batch[1]
    output = input_batch[2]

    criterion1 = torch.nn.MSELoss() 
    mse_loss = criterion1(output, target_batch)

    output_encoder_product = output_encoder*(1 - output_encoder)
    weigths_sum = torch.sum(Variable(weigths)**2, dim=1)
    weigths_sum = weigths_sum.unsqueeze(1)

    contractive_loss = torch.sum(torch.mm(output_encoder_product**2, weigths_sum), 0)

    loss = mse_loss + contractive_loss.mul_(lam)
    
    return loss 