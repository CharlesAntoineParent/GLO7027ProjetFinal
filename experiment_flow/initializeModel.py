"""Module that separate data."""


## Import
from lib import separating
from param import datasets


## Separate data function 
def separate_data(dataset, process):
    separating.separating_process(dataset, process)


## Separate data
if __name__ == "__main__":
    separate_data(datasets.data_CUB200, separating.separate_CUB200_15first)