"""Module defining models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


## Models definition
class UNet(nn.Module):
    """Convolutional networks for image segmentation."""
    def __init__(self) -> None:
        super().__init__()

        self.c11 = nn.Conv2d(1, 8, 3, padding=1)
        self.c12 = nn.Conv2d(8, 8, 3, padding=1)
        self.p1 = nn.MaxPool2d(2)

        self.c21 = nn.Conv2d(8, 16, 3, padding=1)
        self.c22 = nn.Conv2d(16, 16, 3, padding=1)
        self.p2 = nn.MaxPool2d(2)

        self.c31 = nn.Conv2d(16, 32, 3, padding=1)
        self.c32 = nn.Conv2d(32, 32, 3, padding=1)
        self.p3 = nn.MaxPool2d(2)

        self.c41 = nn.Conv2d(32, 64, 3, padding=1)
        self.c42 = nn.Conv2d(64, 64, 3, padding=1)
        self.p4 = nn.MaxPool2d(2)

        self.c51 = nn.Conv2d(64, 64, 3, padding=1)
        self.c52 = nn.Conv2d(64, 64, 3, padding=1)
        self.p5 = nn.MaxPool2d(2)

        self.c551 = nn.Conv2d(64, 128, 3, padding=1)
        self.c552 = nn.Conv2d(128, 128, 3, padding=1)

        self.c61 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c62 = nn.Conv2d(128, 64, 3, padding=1)
        self.c63 = nn.Conv2d(64, 64, 3, padding=1)

        self.c71 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c72 = nn.Conv2d(96, 32, 3, padding=1)
        self.c73 = nn.Conv2d(32, 32, 3, padding=1)

        self.c81 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.c82 = nn.Conv2d(64, 32, 3, padding=1)
        self.c83 = nn.Conv2d(32, 32, 3, padding=1)

        self.c91 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.c92 = nn.Conv2d(32, 16, 3, padding=1)
        self.c93 = nn.Conv2d(16, 16, 3, padding=1)

        self.c101 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.c102 = nn.Conv2d(16, 8, 3, padding=1)
        self.c103 = nn.Conv2d(8, 8, 3, padding=1)

        self.c111 = nn.Conv2d(8, 4, 1)
        self.s1 = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Segment images using the U-Net architecture.

        Args:
            inputs (torch.Tensor): Masks of shape [Nx1x256x1600].

        Returns:
            torch.Tensor: Predictions of shape [Nx4x256x1600].
        """
        c1 = F.elu(self.c11(inputs))
        c1 = F.elu(self.c12(c1))
        p1 = self.p1(c1)

        c2 = F.elu(self.c21(p1))
        c2 = F.elu(self.c22(c2))
        p2 = self.p2(c2)

        c3 = F.elu(self.c31(p2))
        c3 = F.elu(self.c32(c3))
        p3 = self.p3(c3)

        c4 = F.elu(self.c41(p3))
        c4 = F.elu(self.c42(c4))
        p4 = self.p4(c4)

        c5 = F.elu(self.c51(p4))
        c5 = F.elu(self.c52(c5))
        p5 = self.p5(c5)

        c55 = F.elu(self.c551(p5))
        c55 = F.elu(self.c552(c55))

        c6 = self.c61(c55)
        c6 = torch.cat([c6, c5], dim=1)
        c6 = F.elu(self.c62(c6))
        c6 = F.elu(self.c63(c6))

        c7 = self.c71(c6)
        c7 = torch.cat([c7, c4], dim=1)
        c7 = F.elu(self.c72(c7))
        c7 = F.elu(self.c73(c7))

        c8 = self.c81(c7)
        c8 = torch.cat([c8, c3], dim=1)
        c8 = F.elu(self.c82(c8))
        c8 = F.elu(self.c83(c8))

        c9 = self.c91(c8)
        c9 = torch.cat([c9, c2], dim=1)
        c9 = F.elu(self.c92(c9))
        c9 = F.elu(self.c93(c9))

        c10 = self.c101(c9)
        c10 = torch.cat([c10, c1], dim=1)
        c10 = F.elu(self.c102(c10))
        c10 = F.elu(self.c103(c10))

        outputs = self.s1(self.c111(c10))

        return outputs
