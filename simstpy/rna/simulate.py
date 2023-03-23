"""Functions for simulating RNA-seq data"""

import numpy as np
import pandas as pd
import scipy as sp
from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared
from itertools import product

def sim_library_size(params: tuple, n_cells: int):
    """
    Simulate cell-specific library size

    Parameters
    ----------
    params : tuple
        Estimated parameters for library size
    n_cells : int
        Number of simulated cells
    """

    lognormal_dist = sp.stats.lognorm(*params)
    library_size_samples = lognormal_dist.rvs(n_cells)

    return library_size_samples


def sim_mean_expression(params: tuple, n_genes: int):
    """
    Simulate mean expression of genes using gamma distribution

    Parameters
    ----------
    params : tuple
        Estimated parameters for mean gene expression
    n_genes : int
        Number of simulated genes
    """

    gamma_dist = sp.stats.gamma(*params)
    gene_mean_samples = gamma_dist.rvs(n_genes)

    return gene_mean_samples


def sim_single_group(
    library_size_params: tuple,
    mean_expression_params: tuple,
    n_cells: int,
    n_genes: int,
    cell_ids: list = None,
) -> AnnData:
    """
    Simulate a count matrix including one group.

    Parameters
    ----------
    library_size_params : tuple
        Estimated library size parameters
    mean_expression_params : tuple
        Estimated mean expression parameters
    n_cells : int
        Number of simulated cells
    n_genes : int
        Number of simulated genes
    cell_ids : list
        List of cell names

    Returns
    -------
    AnnData
        An AnnData object containing simulated count matrix
    """

    library_size = sim_library_size(library_size_params, n_cells)
    mean_expression = sim_mean_expression(mean_expression_params, n_genes)

    # get a cellxgene matrix where the value represent average expression
    gene_mean = mean_expression / np.sum(mean_expression)

    library_size = np.expand_dims(library_size, axis=1)
    gene_mean = np.expand_dims(gene_mean, axis=0)
    mat = np.matmul(library_size, gene_mean)
    counts = csr_matrix(np.random.poisson(mat))

    if cell_ids is None:
        cell_ids = [f"cell_{i}" for i in range(n_cells)]
        obs = pd.DataFrame(index=cell_ids)

    gene_ids = [f"gene_{i}" for i in range(n_genes)]
    var = pd.DataFrame(index=gene_ids)

    adata = AnnData(counts, obs=obs, var=var, dtype=np.int32)

    return adata


def sim_multi_group(
    lib_size_params: tuple = (0.51313746, 0.0, 2300.6353),
    gene_exp_params: tuple = (0.6227306182997188, 0, 0.24285327175872837),
    n_non_svgs: int = None,
    n_svgs: int = None,
    df_spatial: pd.DataFrame = None,
    group_name: str = None,
    library_id: str = "spatial",
    fold_change: str = "lognormal",
    log_fc: float = 1,
    mean: float = 2,
    sigma: float = 0.5,
    min_fc: float = 2,
    max_fc: float = 4,
) -> AnnData:
    """
    Simulate data with multiple groups

    Parameters
    ----------
    library_size_params : tuple
        Estimated library size parameters
    mean_expression_params : tuple
        Estimated mean expression parameters
    n_genes : int
        Number of simulated genes
    n_marker_genes: int
        Number of marker genes per group
    cell_groups : list
        A list including group information for cells
    cell_ids: list
        A list of cell names. It must be consistent with cell_groups
    mean: float
        Mean of log-normal distribution for sampling differential expression factor
    sigma: float
        Sigma of log-normal distribution for sampling differential expression factor

    Returns
    -------
    AnnData
        An AnnData object containing simulated count matrix
    """

    lib_size = sim_library_size(lib_size_params, n_cells=len(df_spatial))

    df_spatial[group_name] = df_spatial[group_name].astype(str)
    n_groups = len(df_spatial[group_name].unique())

    # we sampled a number of SVGs and non-SVGs
    non_svgs_exp = sim_mean_expression(gene_exp_params, n_non_svgs)
    svgs_exp = sim_mean_expression(gene_exp_params, n_svgs)

    # for SVGs, we split the genes into serveral groups
    svgs_idx_list = np.array_split(range(n_svgs), n_groups)
    all_cell_ids, expression = [], np.empty((0, n_svgs + n_non_svgs))

    for i, cell_group in enumerate(df_spatial[group_name].unique()):
        cell_ids = df_spatial.index.values[df_spatial[group_name].values == cell_group]
        all_cell_ids += list(cell_ids)

        # randomly select a number of DE genes
        svgs_idx = svgs_idx_list[i]

        # generate DE factor from log-normal distribution
        if fold_change == "lognormal":
            de_ratio = np.random.lognormal(
                mean=mean, sigma=sigma, size=len(svgs_idx))
            de_ratio[de_ratio < 1] = 1 / de_ratio[de_ratio < 1]
        elif fold_change == "fixed":
            de_ratio = 2**log_fc
        elif fold_change == "step":
            de_ratio = np.linspace(
                start=min_fc, stop=max_fc, num=len(svgs_idx))

        _svgs_exp = svgs_exp.copy()
        _svgs_exp[svgs_idx] = _svgs_exp[svgs_idx] * de_ratio

        exp = np.concatenate((_svgs_exp, non_svgs_exp))

        exp = exp / np.sum(exp)
        exp = np.expand_dims(exp, axis=0)

        _lib_size = lib_size[df_spatial[group_name].values ==
                             cell_group].copy()
        _lib_size = np.expand_dims(_lib_size, axis=1)

        mat = np.matmul(_lib_size, exp)
        expression = np.concatenate((expression, mat), axis=0)

    # we then generate non SVGs
    counts = np.random.poisson(expression)

    is_de_genes = [False] * (n_svgs + n_non_svgs)
    for i in range(n_svgs):
        is_de_genes[i] = True

    gene_ids = [f"gene_{i}" for i in range(n_svgs + n_non_svgs)]
    var = pd.DataFrame(
        data={"spatially_variable": is_de_genes}, index=gene_ids)

    df_spatial = df_spatial.loc[all_cell_ids]
    df_spatial['obs_ids'] = all_cell_ids
    counts = sp.sparse.csr_matrix(counts)
    adata = AnnData(counts, obs=df_spatial, dtype=np.float32, var=var)

    adata.uns["spatial"] = {library_id: {}}
    adata.obsm["spatial"] = adata.obs[["x", "y"]].values
    adata.obs.drop(columns=["x", "y"], inplace=True)

    return adata

def svgs_rbf_kernel(coords:np.array, 
                    length_scales: list=None, 
                    sigma:float=1) -> np.array:
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
    sigma = sigma**2
    for i, length_scale in enumerate(length_scales):
        kernel = RBF(length_scale=length_scale)
        cov = kernel(coords)
        exp[:, i] = sp.stats.multivariate_normal.rvs(
            mean=np.zeros(n_locations), 
            cov=np.multiply(cov, sigma))
        
    return exp

def svgs_period_kernel(coords:np.array, 
                        length_scales: list=None, 
                        periodicities: list=None,
                        sigma:float=1) -> np.array:
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

    length_scale_periods = list(product(length_scales, periods))


def sim_svgs(height: int = 50,
             width: int = 50,
             n_non_svgs=100,
             library_size=1e4,
             random_state=42,
             sigma=1,
             rbf_length_scales: list = None) -> AnnData:
    """
    Simulate gene expression with spatial correlation

    Parameters
    ----------
    n_svgs : int, optional
        Number of SVGs, by default 10
    n_non_svgs : int, optional
        Number of non-SVGs, by default 2
    library_size : _type_, optional
        Library size, by default 1e4
    random_state : int, optional
        Seed to generate random numbers, by default 42

    Returns
    -------
    Anndata
        An anndata object inclduing simulated data
    """
    x, y = np.meshgrid(np.arange(height), np.arange(width))

    coords = np.column_stack((np.ndarray.flatten(x), np.ndarray.flatten(y)))

    # generate data for SVGs
    np.random.seed(random_state)
    sigma = sigma**2

    # using RBF kernel with different length scale
    rbf_len_scales = np.linspace(start=1, stop=50, num=50)
    rbf_svgs_exp = np.zeros((height * width, len(rbf_length_scales)))
    for length_scale in rbf_len_scales:
        cov = get_cov_from_rbf_kernel(coords, length_scale)
        rbf_svgs_exp[:, i] = sp.stats.multivariate_normal.rvs(
            mean=np.zeros(height*width),
            cov=sigma*cov)

    # using Periodic kernel
    len_scales = np.linspace(start=1, stop=10, num=10)
    periods = np.linspace(start=1, stop=10, num=10)
    len_periods = list(product(len_scales, periods))

    period_svgs_exp = np.zeros((height * width, len(len_periods))

    # generate non-SVGs
    non_svgs_exp=np.zeros((height * width, n_non_svgs))
    for i in range(n_non_svgs):
        non_svgs_exp[:, i]=np.random.standard_normal(height*width) + sigma

    exp=np.concatenate((svgs_exp, non_svgs_exp), axis=1)
    exp=np.exp(exp)
    exp=normalize(exp, axis=1, norm="l1")
    exp=np.random.poisson(library_size * exp)

    counts=np.concatenate((svgs_counts, non_svgs_counts), axis=1)
    counts=np.exp(counts)
    counts=normalize(counts, axis=1, norm="l1")
    counts=np.random.poisson(library_size * counts)

    _, n_svgs=svgs_exp.shape
    is_de_genes=[False] * (n_svgs + n_non_svgs)
    for i in range(n_svgs):
        is_de_genes[i]=True

    var_ids=[f"gene_{i}" for i in range(n_svgs + n_non_svgs)]
    obs_ids=[f"loc_{i}" for i in range(50 * 50)]

    var=pd.DataFrame(data={"spatially_variable": is_de_genes}, index=var_ids)
    obs=pd.DataFrame(data={"obs_ids": obs_ids}, index=obs_ids)

    counts=sp.sparse.csr_matrix(counts)
    adata=AnnData(
        counts, obs=obs, var=var, obsm={"spatial": coords}, dtype=np.float32
    )

    return adata

