# pylint:disable=undefined-variable
"""Module executing experiments."""


## Import
import json

from lib.lib_experiments import *
from lib.lib_flows import *
from lib.lib_models import *
from lib.lib_parameters import *
from lib.lib_processes import *


## Extracting and parsing experiment function
def extract_experiment(experiment_name):
    with open(experiments.experiments_path + experiment_name + ".json") as file:
        experiment = json.load(file)

    return experiment

def parse_experiment(experiment):
    for key, value in experiment.items():
        if type(value) == str:
            try:
                experiment[key] = eval(value)
            except:
                pass
        elif type(value) == list:
            try:
                experiment[key] = [eval(element) for element in value]
            except:
                for element in value:
                    parse_experiment(element)
        elif type(value) == dict:
            parse_experiment(value)

    return experiment


## List of experiments to execute
experiments_list = ["experiment0"]


## Executing experiments
for experiment_name in experiments_list:
    experiment = extract_experiment(experiment_name)
    experiment = parse_experiment(experiment)
    executeExperiment.execute_experiment(experiment)