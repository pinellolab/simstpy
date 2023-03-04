"""Helper functions"""

import pandas as pd
import numpy as np


def array_to_dataframe(pattern: np.array) -> pd.DataFrame:
    """
    Convert spatial pattern from array to dataframe

    Parameters
    ----------
    pattern : np.array
        input pattern

    Returns
    -------
    pd.DataFrame
        _description_
    """

    nrow, ncol = pattern.shape
    x, y, value = list(), list(), list()
    for i in range(nrow):
        for j in range(ncol):
            x.append(i)
            y.append(j)
            value.append(pattern[i, j])

    return pd.DataFrame(data={"x": x, "y": y, "spatial_cluster": value})


def dataframe_to_array(pattern: pd.DataFrame) -> np.array:
    """
    Convert saptial pattern from array to dataframe

    Parameters
    ----------
    pattern : pd.DataFrame
        Input data

    Returns
    -------
    np.array
        _description_
    """

    x_axis = pattern.x.values
    y_axis = pattern.y.values
    spatial_cluster = pattern.spatial_cluster.values

    nrow = np.max(x_axis)
    ncol = np.max(y_axis)
    res = np.zeros(shape=(nrow + 1, ncol + 1))

    for i in range(len(x_axis)):
        res[x_axis[i], y_axis[i]] = spatial_cluster[i]

    return res
