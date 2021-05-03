# pylint:disable=no-member
"""Module defining metrics."""


## Import
import torch
import numpy as np
import scipy.linalg as linalg


## Metrics definition
def accuracy_metric(input_batch, target_batch):

    return float((input_batch.argmax(-1) == target_batch).sum()/len(target_batch))

def frechet_metric(input_batch, target_batch):
    eps = 1e-6

    input_batch_flatten = torch.flatten(input_batch, start_dim=1)
    target_batch_flatten = torch.flatten(target_batch, start_dim=1)

    input_batch_array = np.transpose(np.asarray(input_batch_flatten.cpu()))
    target_batch_array = np.transpose(np.asarray(target_batch_flatten.cpu()))

    input_batch_mean = np.mean(input_batch_array, axis=1)
    target_batch_mean = np.mean(target_batch_array, axis=1)

    input_batch_cov = np.cov(input_batch_array)
    target_batch_cov = np.cov(target_batch_array)

    covmean, _ = linalg.sqrtm(np.dot(input_batch_cov, target_batch_cov), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(input_batch_cov.shape[0])*eps
        covmean = linalg.sqrtm(np.dot(input_batch_cov + offset, target_batch_cov + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    frechet_distance = (np.dot(input_batch_mean - target_batch_mean, input_batch_mean - target_batch_mean) + 
                np.trace(input_batch_cov + target_batch_cov - 2*covmean))

    return frechet_distance
    






