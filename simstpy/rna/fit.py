"""Functions for data fitting"""

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix


def fit_library_size(data: csr_matrix):
    """
    Fitting observed library size using log-normal distribution.

    Parameters
    ----------
    data : csr_matrix
        Raw cell by gene count matrix. 
    """

    library_size = np.array(data.sum(axis=1)).flatten()

    # Fit a log-normal distribution to the data using maximum likelihood estimation
    params = sp.stats.lognorm.fit(library_size, floc=0)

    return params


def fit_gene_expression(data: csr_matrix, quantile: float = 0.99):
    """
    Fit mean gene expression using Gamma distribution

    Parameters
    ----------
    data : csr_matrix
        Normalized cell by gene expression matrix. 
    quantile : float, optional
        quantile, by default True
    """
    mean = np.array(data.mean(axis=0)).flatten()
    if quantile is not None:
        threshold = np.quantile(mean, quantile)
        # update the mean value
        mean[mean > threshold] = threshold

    # Fit a gamma distribution to the data using maximum likelihood estimation
    params = sp.stats.gamma.fit(mean, floc=0)

    return params
