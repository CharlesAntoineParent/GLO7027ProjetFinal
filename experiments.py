# pylint:disable=undefined-variable
"""Module that define experiments process."""


## Import
import json

from lib.lib_flows import *
from lib.lib_models import *
from lib.lib_parameters import *
from lib.lib_processes import *


## Experiments process definition
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

def execute_experiment(experiment):
    results = []
    for flow in experiment['flows']:
        if flow['type'] == "SEPARATE_DATA":
            flow['flow'](experiment['dataset'])

        if flow['type'] == "LOAD_DATA":
            data = flow['flow'](experiment['dataset'], flow['transformation'], flow['split'])

        if flow['type'] == "INITIALIZE_MODEL":
            model - flow['flow'](flow['initialization'], experiment['dataset'], flow['network'])

        if flow['type'] == "TRAIN_MODEL":
            results.append(flow['flow'](flow['device'], flow['hyperparameters'], 
                                        model, flow['optimizer'], flow['criterion'], flow['metrics'], 
                                        *data))

        if flow['type'] == "ANALYZE_RESULTS":
            print("analyzing BLA BLA BLA")

    print(results[0][1])
    print("Experiment done !")
