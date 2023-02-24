"""Functions for fitting data distribution"""
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

def fit_library_size(data: csr_matrix):
    """
    Fitting observed library size using log-normal distribution.

    Parameters
    ----------
    data : _type_
        Raw cell by gene count matrix. 
    plot : bool, optional
        Plot the data and fitted distribution, by default True
    """

    library_size = np.array(data.sum(axis=1)).flatten()

    # Fit a log-normal distribution to the data using maximum likelihood estimation
    params = sp.stats.lognorm.fit(library_size, floc=0)

    return params


def fit_mean_expression_gamma(data: csr_matrix, q: float=0.99):
    """
    Fit mean expression of genes using Gamma distribution

    Parameters
    ----------
    data : _type_
        Normalized cell by gene expression matrix. 
    q : float, optional
        _description_, by default True
    """
    mean = np.array(data.mean(axis=0)).flatten()
    if q is not None:
        threshold = np.quantile(mean, q)
        # update the mean value
        mean[mean > threshold] = threshold

    # Fit a gamma distribution to the data using maximum likelihood estimation
    params = sp.stats.gamma.fit(mean, floc=0)

    return params
