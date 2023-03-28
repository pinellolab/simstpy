"""Functions for simulating spatially variable genes"""

import numpy as np
import pandas as pd
import scipy as sp
from anndata import AnnData
from sklearn.preprocessing import normalize
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared
from sklearn.datasets import make_spd_matrix
from itertools import product
from numpy.random import default_rng

from .utils import check_pos_semidefinite


def sim_svgs(
    height: int = 50,
    width: int = 50,
    n_svgs: int = 50,
    n_non_svgs: int = 100,
    n_kernels: int = 5,
    alpha: float = 0.0,
    sigma: float = 1.0,
    library_size: int = 1e4,
    random_state: int = 42,
) -> AnnData:
    """
    Generate spatially variable genes

    Parameters
    ----------
    height : int, optional
        Height of spatial space, by default 50
    width : int, optional
        Width of spatial space, by default 50
    n_rbf_svgs : int, optional
        Number of SVGs based on RBF kernel, by default 50
    n_period_svgs : int, optional
        Number of SVGs based on periodic kernel, by default 100
    n_rand_svgs : int, optional
        Number of SVGs based on random kernel, by default 20
    n_non_svgs : int, optional
        Number of non-SVGs, by default 100
    alpha : float, optional
        Proportion of noise for SVGs, by default 0.1
    sigma : float, optional
        Amplitude of covariance for SVGs, by default 1.0
    library_size : int, optional
        Library size, by default 1e4
    random_state : int, optional
        Random seed, by default 32

    Returns
    -------
    AnnData
        _description_
    """

    rng = default_rng(random_state)

    n_locations = height * width
    x, y = np.meshgrid(np.arange(height), np.arange(width))
    coords = np.column_stack((np.ndarray.flatten(x), np.ndarray.flatten(y)))
    
    # generate SVGs using RBF kernel
    length_scales = np.linspace(start=1, stop=10, num=n_kernels)
    cov = np.zeros((n_kernels, n_locations, n_locations))
    for i, length_scale in enumerate(length_scales):
        cov[i] = get_cov_from_rbf(coords=coords, length_scale=length_scale)

    # get random proportion
    svg_exp = np.zeros((n_locations, n_svgs))
    for i in range(n_svgs):
        proportion = rng.dirichlet(alpha=1.0 / np.ones(n_kernels))

        _cov = np.zeros((n_locations, n_locations))
        for j in range(n_kernels):
            _cov += np.multiply(cov[j], proportion[j])

        svg_exp[:, i] = rng.multivariate_normal(mean=np.zeros(n_locations), cov=_cov)
        svg_exp[:, i] = np.multiply(svg_exp[:, i], sigma)

    # add noise to simualted SVGs
    noise = np.multiply(rng.standard_normal(size=(n_locations, n_svgs)))
    svg_exp = np.multiply(svg_exp, 1 - alpha) + np.multiply(noise, alpha)

    # generate non-SVGs using white noise kernel
    non_svgs_exp = np.zeros((n_locations, n_non_svgs))
    for i in range(n_non_svgs):
        non_svgs_exp[:, i] = np.multiply(rng.standard_normal(n_locations), sigma)

    ## combine SVGs and non-SVGs, and convert the data to counts
    exp = np.concatenate((svg_exp, non_svgs_exp), axis=1)
    exp = np.exp(exp)
    exp = normalize(exp, axis=1, norm="l1")
    counts = rng.poisson(library_size * exp)

    is_de_genes = [False] * (n_svgs + n_non_svgs)
    for i in range(n_svgs):
        is_de_genes[i] = True

    var_ids = [f"gene_{i}" for i in range(n_svgs + n_non_svgs)]
    obs_ids = [f"loc_{i}" for i in range(height * width)]

    var = pd.DataFrame(data={"spatially_variable": is_de_genes}, index=var_ids)
    obs = pd.DataFrame(data={"obs_ids": obs_ids}, index=obs_ids)

    counts = sp.sparse.csr_matrix(counts)
    adata = AnnData(
        counts, obs=obs, var=var, obsm={"spatial": coords}, dtype=np.float32
    )

    return adata


def svgs_rand_covariance(
    coords: np.array, n_svgs: int = 20, sigma: float = 1
) -> np.array:
    """
    Generate spatially variable genes using random covariane matrix

    Parameters
    ----------
    coords : np.array
        Spatial coordinates
    n_svgs : int, optional
        Number of genes to generate, by default 10

    Returns
    -------
    np.array
        _description_
    """
    n_locations = coords.shape[0]
    exp = np.zeros((n_locations, n_svgs))

    for i in range(n_svgs):
        cov = make_spd_matrix(n_dim=n_locations, random_state=i)
        cov = np.multiply(cov, sigma)
        exp[:, i] = sp.stats.multivariate_normal.rvs(
            mean=np.zeros(n_locations), cov=cov
        )

    return exp


def get_cov_from_rbf(coords: np.array, length_scale: float = 1.0) -> np.array:
    """
    Generate covariance matrix using RBF kernel

    Parameters
    ----------
    coords : np.array
        Spatial coordinates
    length_scale : float, optional
        Length scale, by default 1.0

    Returns
    -------
    np.array
        Covariance matrix
    """

    kernel = RBF(length_scale=length_scale)
    cov = kernel(coords)

    return cov


def svgs_rbf_kernel(
    coords: np.array, length_scales: list = None, sigma: float = 1
) -> np.array:
    """
    Generate SVGs using RBF kernel

    Parameters
    ----------
    coords : np.array
        Spatial coordinates
    length_scales : list, optional
        Length scales, by default None
    """
    n_locations = coords.shape[0]
    exp = np.zeros((n_locations, len(length_scales)))

    for i, length_scale in enumerate(length_scales):
        kernel = RBF(length_scale=length_scale)
        cov = kernel(coords)
        cov = check_pos_semidefinite(cov=cov)
        cov = np.multiply(cov, sigma)
        exp[:, i] = sp.stats.multivariate_normal.rvs(
            mean=np.zeros(n_locations), cov=cov
        )

    return exp


def svgs_period_kernel(
    coords: np.array,
    length_scales: list = None,
    periodicities: list = None,
    sigma: float = 1,
) -> np.array:
    """
    Generate spatially variable genes using period kernel

    Parameters
    ----------
    coords : np.array
        _description_
    length_scales : list, optional
        _description_, by default None
    periodicities : list, optional
        _description_, by default None
    sigma : float, optional
        _description_, by default 1

    Returns
    -------
    np.array
        _description_
    """
    n_locations = coords.shape[0]

    length_scale_periods = list(product(length_scales, periodicities))
    exp = np.zeros((n_locations, len(length_scale_periods)))

    for i, length_scale_period in enumerate(length_scale_periods):
        kernel = ExpSineSquared(
            length_scale=length_scale_period[0], periodicity=length_scale_period[1]
        )
        cov = kernel(coords)
        cov = check_pos_semidefinite(cov=cov)
        cov = np.multiply(cov, sigma)
        exp[:, i] = sp.stats.multivariate_normal.rvs(
            mean=np.zeros(n_locations), cov=cov
        )

    return exp
