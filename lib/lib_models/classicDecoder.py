"""Module defining classic Decoder model."""


## Import
import torch.nn as nn


## Model definition
class Decoder(nn.Module):
    def __init__(self, capacity):
        self.capacity = capacity
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(int(self.capacity*1.5), self.capacity, bias = False)
        
    def forward(self, x):
        fc1 = self.relu(self.fc1(x))
        return fc1