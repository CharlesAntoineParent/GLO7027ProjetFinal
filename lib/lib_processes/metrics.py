# pylint:disable=no-member
"""Module defining metrics."""


## Import
import torch
import numpy as np


## Metrics definition
def accuracy_metric(input_batch, target_batch):

    return float((input_batch.argmax(-1) == target_batch).sum()/len(target_batch))

def frechet_metric(input_batch, target_batch):
    input_batch_flatten = torch.flatten(input_batch, start_dim=1)
    target_batch_flatten = torch.flatten(target_batch, start_dim=1)

    input_batch_array = np.transpose(np.asarray(input_batch_flatten.cpu()))
    target_batch_array = np.transpose(np.asarray(target_batch_flatten.cpu()))

    input_batch_mean = np.mean(input_batch_array, axis=1)
    target_batch_mean = np.mean(target_batch_array, axis=1)

    input_batch_cov = np.cov(input_batch_array)
    target_batch_cov = np.cov(target_batch_array)

    frechet_distance = (np.dot(input_batch_mean - target_batch_mean, input_batch_mean - target_batch_mean) + 
                np.trace(input_batch_cov + target_batch_cov - 2*np.sqrt(np.matmul(input_batch_cov,target_batch_cov))))

    return frechet_distance
    






