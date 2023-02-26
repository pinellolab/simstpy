"""Plotting functions"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.sparse import csr_matrix


def plot_library_size_distribution(data: csr_matrix, params: tuple):
    """
    Plot fitted distribution for library size

    Parameters
    ----------
    data : csr_matrix
        Raw cell by gene count matrix. 
    params : tuple
        Estimated parameters
    """
    library_size = np.array(data.sum(axis=1)).flatten()

    x = np.linspace(0, np.max(library_size), 1000)
    pdf = sp.stats.lognorm.pdf(x, *params)

    # Plot the histogram of the data and the fitted distribution
    plt.hist(library_size, bins=100, density=True, alpha=0.5, label="Data")
    plt.plot(x, pdf, "r-", label="Log-Norm PDF")

    # Calculate the Kolmogorov-Smirnov test statistic and p-value
    ks_stat, p_val = sp.stats.kstest(library_size, "lognorm", args=params)

    # Add the test statistic and p-value to the plot
    plt.title(f"KS stat: {ks_stat:.3f}; p-value: {p_val:.3f}")

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Probability density")

    # Show the plot
    plt.show()


def plot_gene_mean_distribution(data: np.array, params: tuple):
    """
    Plot fitted distribution for mean gene expression

    Parameters
    ----------
    data : np.array
        Mean gene expression
    params : tuple
        Estimated parameters
    """
    mean = np.array(data.mean(axis=0)).flatten()

    # Get the PDF values of the fitted distribution
    x = np.linspace(0, np.max(mean), 1000)
    pdf = sp.stats.gamma.pdf(x, *params)

    # Plot the histogram of the data and the fitted distribution
    plt.hist(mean, bins=100, density=True, alpha=0.5, label="Data")
    plt.plot(x, pdf, "r-", label="Gamma PDF")

    # Calculate the Kolmogorov-Smirnov test statistic and p-value
    ks_stat, p_val = sp.stats.kstest(mean, "gamma", args=params)

    # Add the test statistic and p-value to the plot
    plt.title(f"KS stat: {ks_stat:.3f}; p-value: {p_val:.3f}")

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Probability density")

    # Show the plot
    plt.show()


def compare_library_size(obs_data: csr_matrix, sim_data: csr_matrix):
    """
    Compare observed and simualted library size

    Parameters
    ----------
    obs_data : csr_matrix
        Observed count matrix
    sim_data : csr_matrix
        Simulated count matrix
    """
    obs_library_size = np.sum(obs_data.toarray(), axis=1)
    sim_library_size = np.sum(sim_data.toarray(), axis=1)

    ks_stat, p_val = sp.stats.kstest(obs_library_size, sim_library_size)

    group = ["observed"] * len(obs_library_size) + ["simulated"] * len(sim_library_size)
    library_size = np.log(np.concatenate((obs_library_size, sim_library_size)))

    data = {"library_size": library_size, "group": group}
    data = pd.DataFrame(data=data)

    sns.boxplot(data=data, x="group", y="library_size", showfliers=False)

    # Add the test statistic and p-value to the plot
    plt.title(f"KS stat: {ks_stat:.3f}; p-value: {p_val:.3f}")

    # Show the plot
    plt.show()


def compare_mean_expression(obs_data: csr_matrix, sim_data: csr_matrix):
    """
    Compare observed and simualted mean gene expression

    Parameters
    ----------
    obs_data : csr_matrix
        Observed normalized matrix
    sim_data : csr_matrix
        Simulated normalized matrix
    """
    obs_mean = np.array(obs_data.mean(axis=0)).flatten()
    sim_mean = np.array(sim_data.mean(axis=0)).flatten()

    ks_stat, p_val = sp.stats.kstest(obs_mean, sim_mean)
    group = ["observed"] * len(obs_mean) + ["simulated"] * len(sim_mean)
    mean = np.concatenate((obs_mean, sim_mean))

    data = {"mean": mean, "group": group}
    df = pd.DataFrame(data=data)

    sns.boxplot(data=df, x="group", y="mean", showfliers=False)

    # Add the test statistic and p-value to the plot
    plt.title(f"KS stat: {ks_stat:.3f}; p-value: {p_val:.3f}")

    # Show the plot
    plt.show()


def compare_gene_variance(obs_data: csr_matrix, sim_data: csr_matrix):
    """
    Compare observed and simualted variance of gene expression

    Parameters
    ----------
    obs_data : csr_matrix
        Observed normalized matrix
    sim_data : csr_matrix
        Simulated normalized matrix
    """

    obs_var = np.var(obs_data.toarray(), axis=0)
    sim_var = np.var(sim_data.toarray(), axis=0)

    ks_stat, p_val = sp.stats.kstest(obs_var, sim_var)

    group = ["observed"] * len(obs_var) + ["simulated"] * len(sim_var)
    variance = np.concatenate((obs_var, sim_var))

    data = {"variance": variance, "group": group}
    df = pd.DataFrame(data=data)
    sns.boxplot(data=df, x="group", y="variance", showfliers=False)

    # Add the test statistic and p-value to the plot
    plt.title(f"KS stat: {ks_stat:.3f}; p-value: {p_val:.3f}")

    # Show the plot
    plt.show()
