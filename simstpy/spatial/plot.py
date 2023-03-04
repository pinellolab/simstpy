""" Plot functions"""

import pandas as pd
import matplotlib.pyplot as plt

from .utils import dataframe_to_array

def plot_pattern(pattern:pd.DataFrame):
    """
    Plot spatial pattern

    Parameters
    ----------
    pattern : pd.DataFrame
        Spatial pattern for plotting
    """

    # convert dataframe to array
    pattern = dataframe_to_array(pattern)

    plt.imshow(pattern)
