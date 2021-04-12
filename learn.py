"""Module for training models."""
import torch
import torch.nn as nn
import torch.optim as optim
from livelossplot import PlotLosses
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets import SeverstalDataset
from load import load
from losses import DiceBCELoss
from metrics import DiceCoefficient, JaccardIndex
from models import UNet


def learn(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, epochs: int = 100, batch_size: int = 16) -> None:
    """Train a model.

    Args:
        model (nn.Module): The custom model to train.
        optimizer (optim.Optimizer): The optimizer to use when training.
        criterion (nn.Module): The loss used for backpropagation.
        epochs (int, optional): The number of epochs. Defaults to 100.
        batch_size (int, optional): The minibatch size. Defaults to 16.
    """
    train_defects, val_defects = load('./../data/train.csv')

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = SeverstalDataset(train_defects, transform)
    val_dataset = SeverstalDataset(val_defects, transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plotlosses = PlotLosses()
    model.to(device)

    for epoch in tqdm(range(epochs)):
        metrics = {metric: 0.0 for metric in ['loss', 'dice', 'jaccard', 'val_loss', 'val_dice', 'val_jaccard']}

        model.train()

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch)
            loss = criterion(output, y_batch)

            loss.backward()
            optimizer.step()

            metrics['loss'] += float(loss)
            metrics['dice'] += float(DiceCoefficient.forward(output, y_batch))
            metrics['jaccard'] += float(JaccardIndex.forward(output, y_batch))

        metrics['loss'] /= len(train_loader)
        metrics['dice'] /= len(train_loader)
        metrics['jaccard'] /= len(train_loader)

        model.eval()

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(x_batch)
                loss = criterion(output, y_batch)

                metrics['val_loss'] += float(loss)
                metrics['val_dice'] += float(DiceCoefficient.forward(output, y_batch))
                metrics['val_jaccard'] += float(JaccardIndex.forward(output, y_batch))

        metrics['val_loss'] /= len(val_loader)
        metrics['val_dice'] /= len(val_loader)
        metrics['val_jaccard'] /= len(val_loader)

        plotlosses.update(metrics)
        plotlosses.send()

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict,
                    **metrics}, f'{model.__class__.__name__}.tar')

    torch.save(model.state_dict(), f'{model.__class__.__name__}.pt')

if __name__ == '__main__':
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = DiceBCELoss()

    learn(model, optimizer, criterion)
