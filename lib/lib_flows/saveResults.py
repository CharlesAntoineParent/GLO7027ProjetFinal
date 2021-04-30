"""Module that define results saving flows."""


## Import
import json
import torch


## Results saving flows definition
def save_results_flow(experiment, paths, results):
    for label, evaluation, history, model in zip(
        results['labels'], results['evaluations'], results['histories'], results['models']):

        name = label['model'] + "_" + label['optimizer'] + "_" + label['criterion'] + "_" + label['hyperparameters']
        path = paths['results_path'] + experiment + "\\"

        save_model(model, path + paths['models_path'], name)
        save_history(history, path + paths['histories_path'], name)
        save_evaluation(evaluation, path + paths['evaluations_path'], name)

    for name, graphic in results['graphics'].items():
        save_graphic(graphic, path + paths['graphics_path'], name)

    for name, statistic in results['statistics'].items():
        save_statistic(statistic, path + paths['statistics_path'], name)


## Utils function
def save_evaluation(evaluation, path, name):
    with open(path + name + ".json", 'w') as file:
        json.dump(evaluation, file, indent=1)

def save_model(model, path, name):
    torch.save(model.state_dict(), path + name + ".pth")

def save_history(history, path, name):
    with open(path + name + ".json", 'w') as file:
        json.dump(history, file, indent=1)

def save_graphic(graphic, path, name):
    graphic.savefig(path + name + ".pdf")

def save_statistic(statistic, path, name):
    pass


