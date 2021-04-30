"""Module defining losses."""


## Import
import torch


## Losses definition
def CrossEntropy_criterion(model, input_batch, target_batch):
    loss = torch.nn.CrossEntropyLoss()

    return loss(input_batch, target_batch)

def MSE_criterion(model, input_batch, target_batch):
  loss = torch.nn.BCELoss(size_average = False)

  return loss

def contractive_criterion(model, y_pred, y_true):
    mse = K.mean(K.square(y_true - y_pred), axis=1)

    W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
    W = K.transpose(W)  # N_hidden x N
    h = model.get_layer('encoded').output
    dh = h * (1 - h)  # N_batch x N_hidden

    # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
    contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

    return mse + contractive