"""Module for extracting datasets."""
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a file containing defects and seperate them between training and validation dataset.

    Args:
        path (str): The path to the file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and validation datasets.
    """
    defects = pd.read_csv(path)
    unique_defects = defects.drop_duplicates(subset=['ImageId'], keep='first')
    duplicate_defects = defects[defects.duplicated(['ImageId'], keep='first')]

    train_defects, val_defects = train_test_split(unique_defects, test_size=0.2, stratify=unique_defects['ClassId'], random_state=0)

    train_defects = train_defects.append(duplicate_defects[duplicate_defects['ImageId'].isin(train_defects['ImageId'])])
    val_defects = val_defects.append(duplicate_defects[duplicate_defects['ImageId'].isin(val_defects['ImageId'])])

    return train_defects, val_defects
