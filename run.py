# pylint:disable=undefined-variable
"""Module running experiments."""


## Import
import experiments


## List of experiments to run
experiments_list = ["experiment0"]


## Running experiments
for experiment_name in experiments_list:
    experiment = experiments.extract_experiment(experiment_name)
    experiment = experiments.parse_experiment(experiment)
    experiments.execute_experiment(experiment)