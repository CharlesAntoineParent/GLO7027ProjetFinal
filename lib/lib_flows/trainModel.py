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
    final_train_evaluation = evaluate(learn_results['model'], criterion, metric, train_loader, device)
    final_test_evaluation = evaluate(learn_results['model'], criterion, metric, test_loader, device)

    if val_dataset is not None:
        final_val_evaluation = evaluate(learn_results['model'], criterion, metric, val_loader, device)
        training_process_results = learn_results
        training_process_results['evaluations'] =   {
                                                    "train_loss": final_train_evaluation[0], 
                                                    "train_metrics": final_train_evaluation[1], 
                                                    "val_loss": final_val_evaluation[0], 
                                                    "val_metrics": final_val_evaluation[1], 
                                                    "test_loss": final_test_evaluation[0],
                                                    "test_metrics": final_test_evaluation[1]
                                                    }
    else:
        training_process_results = learn_results 
        training_process_results['evaluations'] =   {
                                                    "train_loss": final_train_evaluation[0], 
                                                    "train_metrics": final_train_evaluation[1], 
                                                    "test_loss": final_test_evaluation[0],
                                                    "test_metrics": final_test_evaluation[1]
                                                    }

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
            running_loss = criterion(model, output_batch, target_batch)

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
        learn_results = {
                        "model": model, 
                        "histories": {
                                     "epoch_values": epoch_values, 
                                     "train_losses_history": [train_evaluation[0] for train_evaluation in train_evaluations], 
                                     "train_metrics_history": [train_evaluation[1] for train_evaluation in train_evaluations], 
                                     "val_losses_history": [val_evaluation[0] for val_evaluation in val_evaluations], 
                                     "val_metrics_history": [val_evaluation[1] for val_evaluation in val_evaluations]
                                     }
                        }
    else:
        learn_results = {"model": model, 
                        "histories": {
                                     "epoch_values": epoch_values, 
                                     "train_losses_history": [train_evaluation[0] for train_evaluation in train_evaluations], 
                                     "train_metrics_history": [train_evaluation[1] for train_evaluation in train_evaluations]
                                     }
                        }

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

                evaluation_loss = float(criterion(model, output_batch, target_batch))
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

                evaluation_loss = float(criterion(model, output_batch, target_batch))
                loss_average += evaluation_loss*(len(input_batch)/len(loader.dataset))

                evaluation_metric = metric(output_batch, target_batch)
                metric_average += evaluation_metric*(len(input_batch)/len(loader.dataset))
    
    return [loss_average, metric_average]