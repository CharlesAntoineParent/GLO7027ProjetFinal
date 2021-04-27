"""Module defining parameters of the possible initializations."""


## Import
import torch
import torch.nn as nn


## Parameters definition
def default_init(network):
    pass

def uniform_init(network):
    if type(network) == nn.Linear:
        network.reset_parameters()
        nn.init.uniform_(network.weight, -1, 1)

def normal_init(network):
    if type(network) == nn.Linear:
        network.reset_parameters()
        torch.nn.init.normal_(network.weight, 0, 1)

def constant_init(network):
    if type(network) == nn.Linear:
        network.reset_parameters()
        torch.nn.init.constant_(network.weight, 0.1)

def XavierNormal_init(network):
    if type(network) == nn.Linear:
        network.reset_parameters()
        torch.nn.init.xavier_normal_(network.weight, 1)

def KaimingUniform_init(network):
    if type(network) == nn.Linear:
        network.reset_parameters()
        torch.nn.init.kaiming_uniform_(network.weight, 1)

pretrained = "pretrained"