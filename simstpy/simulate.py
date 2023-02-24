""" Functions for data simulation."""

import numpy as np
import scipy as sp
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix


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


def simuate_single_group(library_size_params: tuple,
                         mean_expression_params: tuple,
                         n_cells: int, n_genes: int, cell_ids: list = None) -> AnnData:
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
    csr_matrix
        A simulated count matrix
    """

    sim_library_size = simulate_library_size(library_size_params, n_cells)
    sim_mean_expression = simulate_mean_expression(
        mean_expression_params, n_genes)

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
