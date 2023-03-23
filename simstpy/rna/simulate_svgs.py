"""Functions for simulating spatially variable genes"""

import numpy as np
import pandas as pd
import scipy as sp
from anndata import AnnData
from sklearn.preprocessing import normalize
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared
from itertools import product


def sim_svgs(
    height: int = 50,
    width: int = 50,
    n_non_svgs=100,
    sigma: float = 1.0,
    library_size: int = 1e4,
    random_state: int = 32,
) -> AnnData:

    x, y = np.meshgrid(np.arange(height), np.arange(width))
    coords = np.column_stack((np.ndarray.flatten(x), np.ndarray.flatten(y)))

    # set random seed and amplitude of the covariance
    np.random.seed(random_state)
    sigma = sigma**2
    n_svgs = 0

    # generate SVGs using RBF kernel
    length_scales = np.linspace(start=1, stop=50, num=50)
    rbf_svgs_exp = svgs_rbf_kernel(
        coords=coords, length_scales=length_scales, sigma=sigma
    )

    # generate SVGs using periodic kernel
    length_scales = np.linspace(start=1, stop=10, num=10)
    periodicities = np.linspace(start=1, stop=10, num=10)
    period_svgs_exp = svgs_period_kernel(
        coords=coords,
        length_scales=length_scales,
        periodicities=periodicities,
        sigma=sigma,
    )

    n_svgs = rbf_svgs_exp.shape[1] + period_svgs_exp[1]

    # generate non-SVGs using White noise kernel
    non_svgs_exp = np.zeros((height * width, n_non_svgs))
    for i in range(n_non_svgs):
        non_svgs_exp[:, i] = np.random.standard_normal(height * width) + sigma

    ## combine SVGs and non-SVGs
    exp = np.concatenate((rbf_svgs_exp, non_svgs_exp), axis=1)
    exp = np.exp(exp)
    exp = normalize(exp, axis=1, norm="l1")
    counts = np.random.poisson(library_size * exp)

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
        exp[:, i] = sp.stats.multivariate_normal.rvs(
            mean=np.zeros(n_locations), cov=np.multiply(cov, sigma)
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
        exp[:, i] = sp.stats.multivariate_normal.rvs(
            mean=np.zeros(n_locations), cov=np.multiply(cov, sigma)
        )

    return exp
