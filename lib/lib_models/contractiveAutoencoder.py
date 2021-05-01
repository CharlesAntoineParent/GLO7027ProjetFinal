# pylint:disable=no-member
"""Module defining contractive Autoencoder model."""


## Import
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        output_decoder = self.decoder(output_encoder)
        output_decoder = torch.sigmoid(output_decoder)

        output = output_decoder.view(*shape)
 
        return [self.state_dict()['encoder.weight'] , output_encoder, output]

    @property
    def type(self):
        return "Autoencoder"