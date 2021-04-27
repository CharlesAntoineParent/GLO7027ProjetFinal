"""Module defining training process."""


## Import
import torch

from lib import loading
from lib import initializing
from lib import learning


## Training process definition
def training_process(loading_process, initializing_process, dataset, split, transformation, initialization, 
                        optimizer, criterion, metrics, hyperparameters, device):
    dataset_train, dataset_val, dataset_test = loading.loading_process(dataset, loading_process, split, transformation)

    model = initializing.initializing_process(dataset, initializing_process, initialization)

    learning_results = learning.learning_process(model, optimizer, criterion, metrics, dataset_train, dataset_val, dataset_test, device, 
                                hyperparameters)

    print(learning_results)