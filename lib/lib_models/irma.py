"""Module defining Irma model."""


## Import
import torch.nn as nn
import torch.nn.functional as F


## Model definition
class Linear(nn.Module):
    def __init__(self, capacity, num_of_linear_layers = 4, type = "linear"):
        self.capacity = capacity
        self.num_of_linear_layers = num_of_linear_layers
        super(Linear, self).__init__()

        if type == "linear":
            self.linear_layer = nn.ModuleList([nn.Linear(in_features=int(self.capacity*1.5), 
                                                        out_features=int(self.capacity*1.5)) 
                                                        for _ in range(num_of_linear_layers)])
            self.forward = self.forward_linear

        if type == "weight_share":
            self.linear_layer = nn.Linear(in_features=int(self.capacity*1.5), 
                                            out_features=int(self.capacity*1.5))
            self.forward = self.forward_weight_share

        if type == "fixed":
            self.linear_layer = nn.ModuleList([nn.Linear(in_features=int(self.capacity*1.5), 
                                                        out_features=int(self.capacity*1.5)) 
                                                        for _ in range(num_of_linear_layers)])
            for layer in self.linear_layer:
                for param in layer.parameters():
                    param.requires_grad = False
            self.forward = self.forward_fixed

        if type == "nonlinear":
            self.linear_layer = nn.ModuleList([nn.Linear(in_features=int(self.capacity*1.5), 
                                                        out_features=int(self.capacity*1.5)) 
                                                        for _ in range(num_of_linear_layers)])
            self.forward = self.forward_nonlinear
        
    def forward_linear(self, x):
        for layer in self.linear_layer:
            x = layer(x)
        return x

    def forward_weight_share(self, x):
        for _ in range(self.num_of_linear_layers):
            x = self.linear_layer(x)
        return x
    
    def forward_fixed(self, x):
        for layer in self.linear_layer:
            x = layer(x)
        return x
    
    def forward_nonlinear(self, x):
        for layer in self.linear_layer:
            x = F.relu(layer(x))
        return x

    @property
    def label(self):
        return "irma.Linear"