"""Module for custom metrics."""
import torch
import torch.nn as nn


class DiceCoefficient(nn.Module):
    """Dice coefficient metric."""
    @staticmethod
    def forward(inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Compute the Dice coefficient metric.

        Args:
            inputs (torch.Tensor): Predictions of shape [Nx4x256x1600].
            targets (torch.Tensor): Targets of shape [Nx4x256x1600].
            smooth (float, optional): Smooth factor for the Dice coefficient. Defaults to 1.0.

        Returns:
            torch.Tensor: The Dice coefficient.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = torch.sum(inputs * targets)

        return (2.0 * intersection + smooth) / (torch.sum(inputs) + torch.sum(targets) + smooth)

class JaccardIndex(nn.Module):
    """Jaccard index (intersection over union) metric."""
    @staticmethod
    def forward(inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Compute the Jaccard index metric.

        Args:
            inputs (torch.Tensor): Predictions of shape [Nx4x256x1600].
            targets (torch.Tensor): Targets of shape [Nx4x256x1600].
            smooth (float, optional): Smooth factor for the Jaccard index. Defaults to 1e-6.

        Returns:
            torch.Tensor: The Jaccard index.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = torch.sum(inputs * targets)
        total = torch.sum(inputs + targets)
        union = total - intersection

        return (intersection + smooth) / (union + smooth)
