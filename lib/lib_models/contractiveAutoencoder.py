"""Module defining contractive Autoencoder model."""


## Import
import torch
import torch.nn as nn
import torch.nn.functional as F


## Model definition
class Autoencoder(nn.Module):
    def __init__(self, capacity, multiplier_encoder, multiplier_decoder):
        super(Autoencoder, self).__init__()

        nb_neurons_encoder = int(multiplier_encoder*capacity)
        nb_neurons_decoder = int(multiplier_decoder*capacity)

        self.encoder = nn.Linear(
            in_features=capacity, out_features=nb_neurons_encoder, bias=False
        )
        self.link = nn.Linear(
            in_features=nb_neurons_encoder, out_features=nb_neurons_decoder, bias=False
        )
        self.decoder = nn.Linear(
            in_features=nb_neurons_decoder, out_features=capacity, bias=False
        )

    def forward(self, input):
        shape = input.shape

        input = input.view(input.shape[0], -1)

        output_encoder = self.encoder(input)
        output_encoder = F.relu(output_encoder)

        output_link = self.link(output_encoder)
        output_link = F.relu(output_link)

        output_decoder = self.decoder(output_link)
        output_decoder = F.sigmoid(output_decoder)

        output = output_decoder.view(*shape)
 
        return [output_encoder, output]