"""Simulate artifical patterns"""

import numpy as np
import pandas as pd
import pkg_resources
from .utils import array_to_dataframe

SPATIAL_PATTERNS = [
    "human_DLPFC_151507",
    "human_DLPFC_151508",
    "human_DLPFC_151509",
    "human_DLPFC_151510",
    "human_DLPFC_151669",
    "human_DLPFC_151670",
    "human_DLPFC_151671",
    "human_DLPFC_151672",
    "human_DLPFC_151673",
    "human_DLPFC_151674",
    "human_DLPFC_151675",
    "human_DLPFC_151676",
    "mouse_cerebellum",
    "mouse_coronal_slices",
    "breast_tumor",
]

def discrete(size: int = 50, x_gap: int=3, y_gap:int=3) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    size : int, optional
        _description_, by default 50
    x_gap : int, optional
        _description_, by default 3
    y_gap : int, optional
        _description_, by default 3

    Returns
    -------
    pd.DataFrame
        _description_
    """

    pattern = np.zeros([size, size])
    for i in range(size):
        if i %x_gap == 1:
            for j in range(size):
                if j%y_gap == 1:
                    pattern[i, j] = 1

    pattern = array_to_dataframe(pattern)
    return pattern


def corners(size: int = 50) -> pd.DataFrame:
    """
    Generate corners pattern

    Parameters
    ----------
    size : int, optional
        Input size, by default 50

    Returns
    -------
    pd.DataFrame
        Generated spatial pattern
    """
    b = np.zeros([size, size])
    for i in range(50):
        b[i, i:] = 1
    a = np.flip(b, axis=1)
    ab = np.hstack((a, b))
    cd = np.flip(ab, axis=0)

    pattern = np.vstack((ab, cd))
    pattern = array_to_dataframe(pattern)

    return pattern


def scotland(size: int = 50, width: int = 4) -> pd.DataFrame:
    """
    Generate scotland pattern

    Parameters
    ----------
    size : int, optional
        dimensions, by default 50
    width: int, optional
        width, by default 4
    Returns
    -------
    pd.DataFrame
        Generated spatial pattern
    """

    pattern = np.eye(size)

    for i in range(width):
        for j in range(size - i):
            pattern[j, j + i] = 1
            pattern[j + i, j] = 1

    pattern = pattern + np.fliplr(pattern)
    pattern[pattern > 1] = 1

    pattern = array_to_dataframe(pattern)

    return pattern

def get_all_patterns() -> list():
    """
    Get all available spatial patterns
    """

    return SPATIAL_PATTERNS


def read_pattern(spatial_pattern: str) -> pd.DataFrame:
    """
    Read spatial pattern

    Parameters
    ----------
    spatial_pattern : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    assert spatial_pattern in SPATIAL_PATTERNS, f"Cannot find {spatial_pattern}"

    filename = pkg_resources.resource_stream(
        __name__, f"patterns/{spatial_pattern}.csv"
    )

    return pd.read_csv(filename, index_col=0)

