"""Module defining learning processes."""


## Import
import torch

from tqdm import tqdm


## Learning processes definition
def learning_process(model, optimizer, criterion, metrics, train_dataset, val_dataset, test_dataset, device, hyperparameters):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

    learn_results = learn(model, optimizer, criterion, metrics, train_loader, val_loader, device, hyperparameters)

    final_train_evaluation = evaluate(learn_results[0], criterion, metrics, train_loader, device)
    final_val_evaluation = evaluate(learn_results[0], criterion, metrics, val_loader, device)
    final_test_evaluation = evaluate(learn_results[0], criterion, metrics, test_loader, device)

    training_process_results = learn_results + [final_train_evaluation, final_val_evaluation, final_test_evaluation]

    return training_process_results

def learn(model, optimizer, criterion, metrics, train_loader, val_loader, device, hyperparameters):
    instantiated_optimizer = optimizer(model.parameters(), hyperparameters['hyperparameters_optimizer'])

    model.to(device)
    
    epoch_values = []
    train_evaluations = []
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

        train_evaluation = evaluate(model, criterion, metrics, train_loader, device)
        val_evaluation = evaluate(model, criterion, metrics, val_loader, device)
        
        epoch_values.append(epoch + 1)
        train_evaluations.append(train_evaluation)
        val_evaluations.append(val_evaluation)

        print(f"\nepoch {epoch + 1}/{hyperparameters['epochs']}, " + 
                f"train_loss : {train_evaluation[0]}, train_metric : {train_evaluation[1]}, " +
                f"val_loss : {val_evaluations[0]}, val_metric : {val_evaluations[1]}")

    return [model, train_evaluations, val_evaluations]

def evaluate(model, criterion, metrics, loader, device):
    model.to(device)

    model.eval()

    loss_average = 0
    metric_averages = [0]*len(metrics)

    with torch.no_grad():
        for input_batch, target_batch in loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            output_batch = model(input_batch)
            evaluation_loss = float(criterion(output_batch, target_batch))

            evaluation_metrics = [None]*len(metrics)
            for index, metric in enumerate(metrics):
                evaluation_metrics[index] = metric(output_batch, target_batch)

            loss_average += evaluation_loss*(len(input_batch)/len(loader.dataset))
            metric_averages += [evaluation_metric*(len(input_batch)/len(loader.dataset)) for evaluation_metric in evaluation_metrics]
    
    return [loss_average, metric_averages]