# pylint:disable=no-member
"""Module defining contractive irma Autoencoder model."""


## Import
import torch
import torch.nn as nn
import torch.nn.functional as F


## Model definition
class Autoencoder(nn.Module):
    def __init__(self, capacity, multiplier, nb_linear_layer):
        super(Autoencoder, self).__init__()

        nb_neurons = int(multiplier*capacity)

        self.encoder = nn.Linear(
            in_features=capacity, out_features=nb_neurons, bias=False
        )
        self.linear_layers = nn.ModuleList([
            nn.Linear(
                in_features=nb_neurons, out_features=nb_neurons )
                                                        for _ in range(nb_linear_layer)]
        )
        self.decoder = nn.Linear(
            in_features=nb_neurons, out_features=capacity, bias=False
        )

    def forward(self, input):
        shape = input.shape

        input = input.view(input.shape[0], -1)

        output_encoder = self.encoder(input)
        output_linear = F.relu(output_encoder)

        for layer in self.linear_layers:
            output_linear = layer(output_linear)

        output_decoder = self.decoder(output_linear)
        output_decoder = torch.sigmoid(output_decoder)

        output = output_decoder.view(*shape)
 
        return [self.state_dict()['encoder.weight'] , output_encoder, output]

    @property
    def type(self):
        return "Autoencoder"