"""Module that load data."""


## Import
from lib import loading
from param import datasets
from param import splits
from param import transformations


## Load data function 
def load_data(dataset, process, split, transformation):
    return process(dataset['train_path'], dataset['test_path'], split, transformation)


## Separate data
if __name__ == "__main__":
    load_data(datasets.data_CUB200, loading.load_CUB200, splits.average_val, transformations.normalize_with_CUB200)