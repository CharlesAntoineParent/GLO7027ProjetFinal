"""Module defining metrics."""


## Import
import torch


## Metrics definition
def accuracy_metric(input_batch, target_batch):

    return float((input_batch.argmax(-1) == target_batch).sum()/len(target_batch))