"""Module running experiments."""


## Import
import experiments


## List of experiments to run
experiments_list = ["experiment1", "experiment2", "experiment3", "experiment4"]


## Running experiments
for experiment_name in experiments_list:
    experiment = experiments.extract_experiment(experiment_name)
    experiment = experiments.parse_experiment(experiment)
    experiments.execute_experiment(experiment)