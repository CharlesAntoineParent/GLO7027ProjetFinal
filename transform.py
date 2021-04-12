"""Module for preprocessing datasets."""
from typing import Tuple

import numpy as np


def create_masks(encoded_sequences: np.ndarray, labels: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """Create masks of defects out of RLE-encoded sequences.

    Args:
        encoded_sequences (np.ndarray): The RLE-encoded sequences.
        shape (Tuple[int, int, int]): Channels, height and width of the images.

    Returns:
        np.ndarray: Masks of defects of shape [Nx4x256x1600].
    """
    batch = len(encoded_sequences)
    channels, height, width = shape[0], shape[1], shape[2]
    masks = np.zeros((batch, channels, height * width), dtype=np.uint8)

    for index, (encoded_sequence, label) in enumerate(zip(encoded_sequences, labels)):
        encoded_pixels = encoded_sequence.split()

        starts, lengths = np.asarray(encoded_pixels[::2], dtype=int), np.asarray(encoded_pixels[1::2], dtype=int)
        # starts -= 1  # TODO Should starts be inclusive or exclusive?
        ends = starts + lengths

        for start, end in zip(starts, ends):
            masks[index, label - 1, start:end] = 1

    masks = np.transpose(masks.reshape((batch, channels, width, height)), axes=(0, 1, 3, 2))

    return masks
