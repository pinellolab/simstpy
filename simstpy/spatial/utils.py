"""Helper functions"""

import pandas as pd
import pkg_resources

SPATIAL_PATTERNS = [
    "human_DLPFC_151507",
    "human_DLPFC_151508",
    "human_DLPFC_151509",
    "human_DLPFC_151510",
    "human_DLPFC_151669",
    "human_DLPFC_151670",
    "human_DLPFC_151671",
    "human_DLPFC_151672",
    "human_DLPFC_151673",
    "human_DLPFC_151674",
    "human_DLPFC_151675",
    "human_DLPFC_151676",
    "mouse_cerebellum",
    "mouse_coronal_slices",
    "breast_tumor"
]


def get_all_patterns() -> list():
    """
    Get all available spatial patterns
    """

    return SPATIAL_PATTERNS


def read_pattern(spatial_pattern: str) -> pd.DataFrame:
    """
    Read spatial pattern

    Parameters
    ----------
    spatial_pattern : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    assert spatial_pattern in SPATIAL_PATTERNS, f"Cannot find {spatial_pattern}"

    filename = pkg_resources.resource_stream(
        __name__, f"patterns/{spatial_pattern}.csv")

    return pd.read_csv(filename, index_col=0)
