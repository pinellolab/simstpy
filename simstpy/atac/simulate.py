"""Functions for simulating single cell ATAC-seq"""

import pandas as pd
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix


def sim_data(df_spatial: pd.DataFrame=None, 
             group_name: str = None,
             n_unique_fragments: int=10000,
             n_da_peaks: int=1000,
             n_non_da_peaks: int=10000):
    """
    Generate simulation ATAC-seq data

    Parameters
    ----------
    df_spatial : pd.DataFrame, optional
        _description_, by default None
    """

    df_spatial[group_name] = df_spatial[group_name].astype(str)
    n_groups = len(df_spatial[group_name].unique())

    gamma_dist = sp.stats.gamma((0.6227306182997188, 0, 0.24285327175872837))
    da_peaks = gamma_dist.rvs(n_da_peaks)
    non_da_peaks = gamma_dist.rvs(n_non_da_peaks)

    # for SVGs, we split the genes into serveral groups
    da_peaks_list = np.array_split(range(n_da_peaks), n_groups)
    all_cell_ids, accessibility = [], np.empty((0, n_da_peaks + n_non_da_peaks))

    for i, cell_group in enumerate(df_spatial[group_name].unique()):
        cell_ids = df_spatial.index.values[df_spatial[group_name].values == cell_group]
        all_cell_ids += list(cell_ids)

        # randomly select a number of DE genes
        da_peaks_idx = da_peaks_list[i]

        # generate DE factor from log-normal distribution
        _da_peaks = da_peaks.copy()
        _da_peaks[da_peaks_idx] = _da_peaks[da_peaks_idx] * 4

        peak_acc = np.concatenate((_da_peaks, non_da_peaks))

        peak_acc = peak_acc / np.sum(peak_acc)
        peak_acc = np.expand_dims(peak_acc, axis=0)

        _lib_size = np.array([n_unique_fragments]*len(cell_ids))
        _lib_size = np.expand_dims(_lib_size, axis=1)
        mat = np.matmul(_lib_size, peak_acc)
        accessibility = np.concatenate((accessibility, mat), axis=0)

    # we then generate non SVGs
    counts = np.random.binomial(accessibility)
