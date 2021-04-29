"""Module that define model training flows."""


## Import
import torch

from tqdm import tqdm


## Model training flows definition
def train_model_flow(device, hyperparameters, model, optimizer, criterion, metric, train_dataset, test_dataset, val_dataset = None):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)
    else:
        val_loader = None

    learn_results = learn(device, hyperparameters, model, optimizer, criterion, metric, train_loader, val_loader)
    final_train_evaluation = evaluate(learn_results[0], criterion, metric, train_loader, device)
    final_test_evaluation = evaluate(learn_results[0], criterion, metric, test_loader, device)

    if val_dataset is not None:
        final_val_evaluation = evaluate(learn_results[0], criterion, metric, val_loader, device)
        training_process_results = learn_results + [final_train_evaluation, final_val_evaluation, final_test_evaluation]
    else:
        training_process_results = learn_results + [final_train_evaluation, final_test_evaluation]

    return training_process_results


## Utils function
def learn(device, hyperparameters, model, optimizer, criterion, metric, train_loader, val_loader = None):
    instantiated_optimizer = optimizer(model.parameters(), hyperparameters['hyperparameters_optimizer'])

    model.to(device)
    
    epoch_values = []
    train_evaluations = []

    if val_loader is not None:
        val_evaluations = []

    for epoch in tqdm(range(hyperparameters['epochs'])):
        for input_batch, target_batch in train_loader:
            model.train()
            instantiated_optimizer.zero_grad()

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            output_batch = model(input_batch)
            running_loss = criterion(output_batch, target_batch)

            running_loss.backward()
            instantiated_optimizer.step()

        train_evaluation = evaluate(model, criterion, metric, train_loader, device)
        
        epoch_values.append(epoch + 1)
        train_evaluations.append(train_evaluation)

        if val_loader is not None:
            val_evaluation = evaluate(model, criterion, metric, val_loader, device)
            val_evaluations.append(val_evaluation)

        print(f"\nepoch {epoch + 1}/{hyperparameters['epochs']}, " + 
                f"train_loss : {train_evaluation[0]}, train_metric : {train_evaluation[1]}") 
                
        if val_loader is not None:
            print(f", val_loss : {val_evaluation[0]}, val_metric : {val_evaluation[1]}")

    if val_loader is not None:
        learn_results = [model, train_evaluations, val_evaluations]
    else:
        learn_results = [model, train_evaluations]

    return learn_results

def evaluate(model, criterion, metric, loader, device):
    model.to(device)

    model.eval()

    loss_average = 0

    if type(metric) == list:
        metric_average = [0]*len(metric)

        with torch.no_grad():
            for input_batch, target_batch in loader:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                output_batch = model(input_batch)

                evaluation_loss = float(criterion(output_batch, target_batch))
                loss_average += evaluation_loss*(len(input_batch)/len(loader.dataset))

                evaluation_metric = [metric(output_batch, target_batch) for metric in metric]
                metric_average = [metric_average + evaluation_metric*(len(input_batch)/len(loader.dataset)) 
                                    for metric_average, evaluation_metric in zip(metric_average, evaluation_metric)]
    else:
        metric_average = 0

        with torch.no_grad():
            for input_batch, target_batch in loader:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                output_batch = model(input_batch)

                evaluation_loss = float(criterion(output_batch, target_batch))
                loss_average += evaluation_loss*(len(input_batch)/len(loader.dataset))

                evaluation_metric = metric(output_batch, target_batch)
                metric_average += evaluation_metric*(len(input_batch)/len(loader.dataset))
    
    return [loss_average, metric_average]