"""Module for predicting defects using trained models."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import SeverstalDataset
from load import load
from models import UNet


def predict(model: nn.Module, batch_size: int = 16) -> None:
    """Predict using a trained model.

    Args:
        model (nn.Module): The custom model to predict with.
        batch_size (int, optional): The minibatch size. Defaults to 16.
    """
    test_defects = load('./../data/test.csv')  # TODO The file 'test.csv' doesn't exist, some processing needs to be done.

    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = SeverstalDataset(test_defects, transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(f'{model.__class__.__name__}.pt'))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)

            output = model(x_batch)

            # TODO Encode predictions into RLE and save into csv.

if __name__ == '__main__':
    predict(UNet())
