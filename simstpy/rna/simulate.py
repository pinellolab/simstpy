"""Functions for simulating RNA-seq data"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse import csr_matrix
from anndata import AnnData


def simulate_library_size(params: tuple, n_cells: int):
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


def simulate_mean_expression(params: tuple, n_genes: int):
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


def simuate_single_group(
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

    sim_library_size = simulate_library_size(library_size_params, n_cells)
    sim_mean_expression = simulate_mean_expression(mean_expression_params, n_genes)

    # get a cellxgene matrix where the value represent average expression
    gene_mean = sim_mean_expression / np.sum(sim_mean_expression)

    sim_library_size = np.expand_dims(sim_library_size, axis=1)
    gene_mean = np.expand_dims(gene_mean, axis=0)
    mat = np.matmul(sim_library_size, gene_mean)
    counts = csr_matrix(np.random.poisson(mat))

    if cell_ids is None:
        cell_ids = [f"cell_{i}" for i in range(n_cells)]
        obs = pd.DataFrame(index=cell_ids)

    gene_ids = [f"gene_{i}" for i in range(n_genes)]
    var = pd.DataFrame(index=gene_ids)

    adata = AnnData(counts, obs=obs, var=var, dtype=np.int32)

    return adata


def simulate_multi_group(
    lib_size_params: tuple = (0.51313746, 0.0, 2300.6353),
    gene_exp_params: tuple = (0.6227306182997188, 0, 0.24285327175872837),
    n_non_svgs: int = None,
    n_svgs: int = None,
    df_spatial: pd.DataFrame = None,
    group_name: str = None,
    fold_change: str = 'lognormal',
    mean: float = 2,
    sigma: float = 0.5
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

    lib_size = simulate_library_size(lib_size_params, n_cells=len(df_spatial))

    df_spatial[group_name] = df_spatial[group_name].astype(str)
    n_groups = len(df_spatial[group_name].unique())

    # we first generate SVGs for each group
    svgs_exp = simulate_mean_expression(gene_exp_params, n_svgs)
    all_cell_ids, svg_counts = [], np.empty((0, n_svgs))
    svgs_idx_list = np.array_split(range(n_svgs), n_groups)

    for i, cell_group in enumerate(df_spatial[group_name].unique()):
        cell_ids = df_spatial.index.values[df_spatial[group_name].values == cell_group]
        all_cell_ids += list(cell_ids)

        # randomly select a number of DE genes
        svgs_idx = svgs_idx_list[i]

        # generate DE factor from log-normal distribution
        if fold_change == "lognormal":
            de_ratio = np.random.lognormal(mean=mean, sigma=sigma, size=len(svgs_idx))
            de_ratio[de_ratio < 1] = 1 / de_ratio[de_ratio < 1]
        elif fold_change == "fixed":
            de_ratio = mean

        _svgs_exp = svgs_exp
        _svgs_exp[svgs_idx] = _svgs_exp[svgs_idx] * de_ratio

        _svgs_exp = _svgs_exp / np.sum(_svgs_exp)
        _svgs_exp = np.expand_dims(_svgs_exp, axis=0)

        _lib_size = lib_size[df_spatial[group_name].values == cell_group].copy()
        _lib_size = np.expand_dims(_lib_size, axis=1)

        mat = np.matmul(_lib_size, _svgs_exp)
        mat = np.random.poisson(mat)
        svg_counts = np.concatenate((svg_counts, mat), axis=0)

    # we then generate non SVGs
    non_svgs_exp = simulate_mean_expression(gene_exp_params, n_non_svgs)
    non_svgs_exp = non_svgs_exp / np.sum(non_svgs_exp)

    lib_size = np.expand_dims(lib_size, axis=1)
    non_svgs_exp = np.expand_dims(non_svgs_exp, axis=0)
    mat = np.matmul(lib_size, non_svgs_exp)
    non_svg_counts = np.random.poisson(mat)

    counts = np.concatenate((svg_counts, non_svg_counts), axis=1)

    # # select the top 90% genes
    # gene_idx = np.argwhere(
    #     sim_mean_expression > np.quantile(
    #         sim_mean_expression, marker_gene_vmin)
    # ).flatten()

    # df_spatial[group_name] = df_spatial[group_name].astype(str)

    # cell_ids, all_de_genes, counts = [], [], np.empty((0, n_genes))

    # for cell_group in df_spatial[group_name].unique():
    #     # get library size for cells in
    #     library_size = sim_library_size[df_spatial[group_name].values == cell_group]

    #     # get cell ids
    #     cell_ids += list(
    #         df_spatial.index.values[df_spatial[group_name].values == cell_group]
    #     )

    #     # randomly select a number of DE genes
    #     de_genes = np.random.choice(gene_idx, size=n_marker_genes)
    #     all_de_genes += list(de_genes)

    #     # generate DE factor from log-normal distribution
    #     de_ratio = np.random.lognormal(
    #         mean=mean, sigma=sigma, size=n_marker_genes)
    #     de_ratio[de_ratio < 1] = 1 / de_ratio[de_ratio < 1]

    #     # multiply the DE factor to mean gene expression
    #     mean_expression = sim_mean_expression.copy()
    #     mean_expression[de_genes] = mean_expression[de_genes] * de_ratio
    #     mean_expression = mean_expression / np.sum(mean_expression)

    #     library_size = np.expand_dims(library_size, axis=1)
    #     mean_expression = np.expand_dims(mean_expression, axis=0)

    #     mat = np.matmul(library_size, mean_expression)
    #     mat = np.random.poisson(mat)
    #     counts = np.concatenate((counts, mat), axis=0)

    is_de_genes = [False] * (n_svgs + n_non_svgs)
    for i in range(n_svgs):
        is_de_genes[i] = True

    gene_ids = [f"gene_{i}" for i in range(n_svgs + n_non_svgs)]
    var = pd.DataFrame(data={"spatially_variable": is_de_genes}, index=gene_ids)

    counts = sp.sparse.csr_matrix(counts)
    adata = AnnData(counts, obs=df_spatial.loc[all_cell_ids], dtype=np.int16, var=var)

    return adata
