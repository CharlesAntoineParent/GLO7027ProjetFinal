"""Module defining classic Autoencoder model."""


## Import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


## Model definition
class Autoencoder(nn.Module):
    def __init__(self, capacity, multiplier):
        super(Autoencoder, self).__init__()

        nb_neurons = int(multiplier*capacity)

        self.encoder = nn.Linear(
            in_features=capacity, out_features=nb_neurons, bias=False
        )
        self.decoder = nn.Linear(
            in_features=nb_neurons, out_features=capacity, bias=False
        )

    def forward(self, input):
        shape = input.shape

        input = input.view(input.shape[0], -1)

        output_encoder = self.encoder(input)
        output_encoder = F.relu(output_encoder)
        dimensionLatente = output_encoder.detach()
        output_decoder = self.decoder(output_encoder)
        output_decoder = F.relu(output_decoder)

        output = output_decoder.view(*shape)
 
        return output

    @property
    def type(self):
        return "Autoencoder"