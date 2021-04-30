"""Module that define graphics construction from results flows."""


## Import
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator


## Graphics construction flows definition
def construct_loss_graphics_flow(results, arguments):
    names = []
    graphics = []

    for argument in arguments:
        name = argument + "_loss_graphic"

        figure = plt.figure()
        plt.title("Loss evolution during training")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        datasets = argument.split("_")

        if len(datasets) > 1:
            for dataset in datasets:
                if dataset == "train":
                    loss_curves(results, "train_losses_history", dataset)
                elif dataset == "val":
                    loss_curves(results, "val_losses_history", dataset)
                elif dataset == "test":
                    loss_curves(results, "test_losses_history", dataset)
        else:
            dataset = datasets[0]
            if dataset == "train":
                loss_curves(results, "train_losses_history")
            elif dataset == "val":
                loss_curves(results, "val_losses_history")
            elif dataset == "test":
                loss_curves(results, "test_losses_history")

        figure.legend()

        names.append(name)
        graphics.append(figure)

    return names, graphics

def construct_metrics_graphics_flow(results, arguments):
    names = []
    graphics = []

    for argument in arguments:
        name = argument + "_metrics_graphic"

        figure = plt.figure()
        plt.title("Metrics evolution during training")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")

        datasets = argument.split("_")

        if len(datasets) > 1:
            for dataset in datasets:
                if dataset == "train":
                    metrics_curves(results, "train_metrics_history", dataset)
                elif dataset == "val":
                    metrics_curves(results, "val_metrics_history", dataset)
                elif dataset == "test":
                    metrics_curves(results, "test_metrics_history", dataset)
        else:
            dataset = datasets[0]
            if dataset == "train":
                metrics_curves(results, "train_metrics_history")
            elif dataset == "val":
                metrics_curves(results, "val_metrics_history")
            elif dataset == "test":
                metrics_curves(results, "test_metrics_history")

        plt.legend()

        names.append(name)
        graphics.append(figure)

    return names, graphics

def construct_loss_metrics_graphics_flow(results, arguments):
    names = []
    graphics = []

    for argument in arguments:
        name = argument + "_loss_metrics_graphic"

        figure = plt.figure()
        plt.title("Loss and metrics evolution during training")
        plt.xlabel("Epochs")
        plt.ylabel("Loss and metrics")

        datasets = argument.split("_")

        if len(datasets) > 1:
            for dataset in datasets:
                if dataset == "train":
                    loss_curves(results, "train_losses_history", dataset)
                    metrics_curves(results, "train_metrics_history", dataset)
                elif dataset == "val":
                    loss_curves(results, "val_losses_history", dataset)
                    metrics_curves(results, "val_metrics_history", dataset)
                elif dataset == "test":
                    loss_curves(results, "test_losses_history", dataset)
                    metrics_curves(results, "test_metrics_history", dataset)
        else:
            dataset = datasets[0]
            if dataset == "train":
                loss_curves(results, "train_losses_history")
                metrics_curves(results, "train_metrics_history")
            elif dataset == "val":
                loss_curves(results, "val_losses_history")
                metrics_curves(results, "val_metrics_history")
            elif dataset == "test":
                loss_curves(results, "test_losses_history")
                metrics_curves(results, "test_metrics_history")

        plt.legend()

        names.append(name)
        graphics.append(figure)

    return names, graphics


## Utils function
def loss_curves(results, history, dataset = ""):
    non_unique_labels = []

    model_labels = []
    optimizer_labels = []
    hyperparameters_labels = []

    for labels in results['labels']:
        model_labels.append('model')
        optimizer_labels.append('optimizer')
        hyperparameters_labels.append('hyperparameters')

    if len(set(model_labels)) > 1:
        non_unique_labels.append("model")
    if len(set(optimizer_labels)) > 1:
        non_unique_labels.append("optimizer")
    if len(set(hyperparameters_labels)) > 1:
        non_unique_labels.append("hyperparameters")

    for labels, result in zip(results['labels'], results['histories']):
        epochs = result['epoch_values']
        values = result[history]

        label = labels['criterion']
        for non_unique_label in non_unique_labels:
            label += "_" + labels[non_unique_label]

        if dataset != "":
            dataset = "_" + dataset

        plt.plot(epochs, values, label= label + dataset)

def metrics_curves(results, history, dataset = ""):
    non_unique_labels = []

    model_labels = []
    optimizer_labels = []
    hyperparameters_labels = []

    for labels in results['labels']:
        model_labels.append('model')
        optimizer_labels.append('optimizer')
        hyperparameters_labels.append('hyperparameters')

    if len(set(model_labels)) > 1:
        non_unique_labels.append("model")
    if len(set(optimizer_labels)) > 1:
        non_unique_labels.append("optimizer")
    if len(set(hyperparameters_labels)) > 1:
        non_unique_labels.append("hyperparameters")

    for labels, result in zip(results['labels'], results['histories']):
        epochs = result['epoch_values']
        
        label_end = ""
        for non_unique_label in non_unique_labels:
            label_end += "_" + labels[non_unique_label]

        if type(labels['metrics']) == list:
            for label, values in zip(labels['metrics'], result[history]):

                if dataset != "":
                    dataset = "_" + dataset

                plt.plot(epochs, values, label=label + label_end + dataset)
        else:
            values = result[history]
            label = labels['metrics']

            if dataset != "":
                    dataset = "_" + dataset

            plt.plot(epochs, values, label=label + label_end + dataset)