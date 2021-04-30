# pylint:disable=undefined-variable
"""Module that define experiments process."""


## Import
import json
import time

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
    if type(experiment) == dict:
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
    print(f"========== Running {experiment['label']} ========== \n \n")
    experiment_start = time.time()

    results = {}

    results['labels'] = []
    results['evaluations'] = []
    results['histories'] = []
    results['models'] = []

    results['graphics'] = {}
    results['statistics'] = {}

    for flow in experiment['flows']:
        if flow['type'] == "SEPARATE_DATA":
            print("===== Running separateData flow =====")

            start = time.time()
            execute_separateData_flow(flow, experiment)
            end = time.time()

            print(f"===== separateData flow done in {end - start} sec ===== \n")

        if flow['type'] == "LOAD_DATA":
            print("===== Running loadData flow =====")

            start = time.time()
            data = execute_loadData_flow(flow, experiment)
            end = time.time()

            print(f"===== loadData flow done in {end - start} sec ===== \n")

        if flow['type'] == "INITIALIZE_MODEL":
            print("===== Running initializeModel flow =====")

            start = time.time()
            model = execute_initializeModel_flow(flow, experiment)
            end = time.time()

            print(f"===== initializeModel flow done in {end - start} sec ===== \n")

        if flow['type'] == "TRAIN_MODEL":
            print("===== Running trainModel flow =====")

            start = time.time()
            results = execute_trainModel_flow(flow, experiment, data, model, results)
            end = time.time()

            print(f"===== trainModel flow done in {end - start} sec ===== \n")
            
        if flow['type'] == "CONSTRUCT_GRAPHICS":
            print("===== Running constructGraphics flow =====")

            start = time.time()
            results = execute_constructGraphics_flow(flow, experiment, results)
            end = time.time()

            print(f"===== constructGraphics flow done in {end - start} sec ===== \n")

        if flow['type'] == "SAVE_RESULTS":
            print("===== Running saveResults flow =====")

            start = time.time()
            execute_saveResults_flow(flow, experiment, results)
            end = time.time()

            print(f"===== saveResults flow done in {end - start} sec ===== \n")

    experiment_end = time.time()
    print(f"========== {experiment['label']} done in {experiment_end - experiment_start} sec ========== \n \n")


## Utils function
def execute_separateData_flow(flow, experiment):
    flow['flow'](experiment['dataset'])

def execute_loadData_flow(flow, experiment):
    data = flow['flow'](experiment['dataset'], flow['transformation'], flow['split'])

    return data

def execute_initializeModel_flow(flow, experiment):
    model = flow['flow'](flow['initialization'], experiment['dataset'], flow['network'])

    return model

def execute_trainModel_flow(flow, experiment, data, model, results):
    result = flow['flow'](flow['device'], flow['hyperparameters'], 
                model, flow['optimizer'], flow['criterion'], flow['metrics'], 
                *data)

    results['labels'].append(flow['labels'])
    results['models'].append(result['model'])
    results['histories'].append(result['histories'])
    results['evaluations'].append(result['evaluations'])

    return results

def execute_constructGraphics_flow(flow, experiment, results):
    names, graphics = flow['flow'](results, flow['arguments'])
    
    for name, graphic in zip(names, graphics):
        results['graphics'][name] = graphic

    return results

def execute_saveResults_flow(flow, experiment, results):
    flow['flow'](experiment['label'], flow['path'], results)