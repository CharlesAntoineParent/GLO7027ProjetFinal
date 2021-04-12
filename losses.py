"""Module for custom losses."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import DiceCoefficient, JaccardIndex


class DiceLoss(nn.Module):
    """Dice coefficient loss."""
    @staticmethod
    def forward(inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Compute the Dice coefficient loss.

        Args:
            inputs (torch.Tensor): Predictions of shape [Nx4x256x1600].
            targets (torch.Tensor): Targets of shape [Nx4x256x1600].
            smooth (float, optional): Smooth factor for the Dice coefficient. Defaults to 1.0.

        Returns:
            torch.Tensor: The Dice loss.
        """
        return 1.0 - DiceCoefficient.forward(inputs, targets, smooth)

class JaccardLoss(nn.Module):
    """Jaccard index (intersection over union) loss."""
    @staticmethod
    def forward(inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Compute the Jaccard index loss.

        Args:
            inputs (torch.Tensor): Predictions of shape [Nx4x256x1600].
            targets (torch.Tensor): Targets of shape [Nx4x256x1600].
            smooth (float, optional): Smooth factor for the Jaccard index. Defaults to 1e-6.

        Returns:
            torch.Tensor: The Jaccard loss.
        """
        return 1.0 - JaccardIndex.forward(inputs, targets, smooth)

class FocalLoss(nn.Module):
    """The focal loss for dense object detection."""
    @staticmethod
    def forward(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.8, gamma: float = 2.0) -> torch.Tensor:
        """Compute the focal loss.

        Args:
            inputs (torch.Tensor): Predictions of shape [Nx4x256x1600].
            targets (torch.Tensor): Targets of shape [Nx4x256x1600].
            alpha (float, optional): Weights for class imbalance. Defaults to 0.8.
            gamma (float, optional): Scaling factor. Defaults to 2.0.

        Returns:
            torch.Tensor: The focal loss.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return alpha * (1.0 - torch.exp(-bce_loss)) ** gamma * bce_loss

class DiceBCELoss(nn.Module):
    """Combined Dice loss with the standard binary cross-entropy loss."""
    @staticmethod
    def forward(inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0, weights: Tuple[float, float] = (1.0, 1.0)) -> torch.Tensor:
        """Compute the combined Dice loss with the standard binary cross-entropy loss.

        Args:
            inputs (torch.Tensor): Predictions of shape [Nx4x256x1600].
            targets (torch.Tensor): Targets of shape [Nx4x256x1600].
            smooth (float, optional): Smooth factor for the Dice coefficient. Defaults to 1.0.
            weights (Tuple[float, float], optional): Weights to apply on the Dice and BCE losses. Defaults to (1.0, 1.0).

        Returns:
            torch.Tensor: The combined loss.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        dice_loss = DiceLoss.forward(inputs, targets, smooth)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return weights[0] * dice_loss + weights[1] * bce_loss
