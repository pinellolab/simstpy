"""Utils for ATAC-seq simulation"""
import pandas as pd
import pkg_resources

def get_chrom_sizes(genome: str="hg38"):
    """
    Get chromosome size for input genome

    Parameters
    ----------
    genome : str, optional
        which genome to use, by default "hg38"
    """

    filename = pkg_resources.resource_stream(
        __name__, f"genomes/{genome}.chrom.sizes"
    )

    df_chrom_size = pd.read_csv(filename, index_col=0)

    chrom_size = dict()
    with open(filename) as f:
        f.readline()

    df_spatial = pd.read_csv(filename, index_col=0)
