"""Module defining Irma model."""


## Import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


## Model definition
class Encoder(nn.Module):
    def __init__(self,capacity):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 900, bias = False)
        
    def forward(self, x):
        h1 = self.relu(self.fc1(x.view(-1, 784)))
        return h1


class Decoder(nn.Module):
    def __init__(self,capacity):
        super(Decoder, self).__init__()
        self.fc2 = nn.Linear(900, 784, bias = False)
        
            
    def forward(self, x):
        h2 = self.sigmoid(self.fc2(z))
        return h2


class LNN(nn.Module):
    def __init__(self, num_of_linear_layers, type):
        self.num_of_linear_layers = num_of_linear_layers
        super(LNN, self).__init__()

        if type == "linear":
            self.linear_layer = nn.ModuleList([nn.Linear(in_features=900, out_features=900) for _ in range(num_of_linear_layers)])
            self.forward = self.forward_linear

        if type == "weight_share":
            self.linear_layer = nn.Linear(in_features=900, out_features=900)
            self.forward = self.forward_weight_share

        if type == "fixed":
            self.linear_layer = nn.ModuleList([nn.Linear(in_features=900, out_features=900) for _ in range(num_of_linear_layers)])
            for layer in self.linear_layer:
                for param in layer.parameters():
                    param.requires_grad = False
            self.forward = self.forward_fixed

        if type == "nonlinear":
            self.linear_layer = nn.ModuleList([nn.Linear(in_features=900, out_features=900) for _ in range(num_of_linear_layers)])
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



class Irma(nn.Module):
    def __init__(self,capacity, *args, **kwargs):
        super(Irma, self).__init__()
        self.encoder = Encoder(capacity)
        self.lnn = LNN(*args, **kwargs)
        self.decoder = Decoder(capacity)
    
    def forward(self, x):
        h1 = self.encoder(x)
        min_rank_latent = self.lnn(h1)
        h2 = self.decoder(min_rank_latent)
        return h1,h2



class AE(nn.Module):
    def __init__(self,capacity, *args, **kwargs):
        super(Irma, self).__init__()
        self.encoder = Encoder(capacity)
        self.decoder = Decoder(capacity)
    
    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1,h2        