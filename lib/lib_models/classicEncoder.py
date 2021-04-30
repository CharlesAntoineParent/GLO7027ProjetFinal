"""Module defining classic Encoder model."""


## Import
import torch.nn as nn


## Model definition
class Encoder(nn.Module):
    def __init__(self, capacity):
        self.capacity = capacity
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(self.capacity, int(self.capacity*1.5), bias = False)
        
    def forward(self, x):
        fc1 = self.relu(self.fc1(x.view(-1, self.capacity)))
        return fc1