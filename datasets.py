"""Module for custom datasets."""
import os
from typing import Optional, Tuple

import cv2
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from transform import create_masks


class SeverstalDataset(Dataset):
    """Severstal steel defect detection dataset."""
    def __init__(self, defects: pd.DataFrame, transform: Optional[transforms.Compose] = None) -> None:
        super().__init__()

        self.shape = (4, 256, 1600)
        self.filenames = defects['ImageId'].values
        self.labels = defects['ClassId'].values
        self.masks = create_masks(defects['EncodedPixels'].values, self.labels, self.shape)
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(os.path.join('./../data/train_images/', self.filenames[index]), cv2.IMREAD_GRAYSCALE)
        mask = torch.tensor(self.masks[index], dtype=torch.float)

        if self.transform is not None:
            image = self.transform(image)

        return image, mask

    def __len__(self) -> int:
        return len(self.filenames)

    def plot_image(self, index: int) -> None:
        """Plot images with the groundtruth defects marked in blue.

        Args:
            index (int): Index of the image to plot.
        """
        label = self.labels[index]
        image = cv2.imread(os.path.join('./../data/train_images/', self.filenames[index]))
        image[self.masks[index, label - 1] == 1] = [255, 0, 0]

        cv2.imshow(f'Defects - {self.filenames[index]} | Class Id: {label}', image)
        cv2.waitKey()
        cv2.destroyAllWindows()
