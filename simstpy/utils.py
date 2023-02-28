"""Helper functions"""

import json
from pathlib import Path
import pandas as pd
from anndata import AnnData
import pkg_resources
import matplotlib.pyplot as plt

from squidpy.read._utils import _load_image
from squidpy._constants._pkg_constants import Key


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
]


def add_image_file_10x(adata: AnnData, library_id: str, image_path: str) -> AnnData:
    """
    Add image files to AnnData

    Parameters
    ----------
    adata : AnnData
        Input anndata object
    library_id : str
        A string of library
    image_path : str
        Path of image data

    Returns
    -------
    AnnData
        An anndata object
    """

    path = Path(image_path)

    # load count matrix
    adata.uns[Key.uns.spatial] = {library_id: {}}

    # load image
    adata.uns[Key.uns.spatial][library_id][Key.uns.image_key] = {
        res: _load_image(path / f"tissue_{res}_image.png")
        for res in ["hires", "lowres"]
    }
    adata.uns[Key.uns.spatial][library_id]["scalefactors"] = json.loads(
        (path / "scalefactors_json.json").read_bytes()
    )

    # load coordinates
    coords = pd.read_csv(path / "tissue_positions_list.txt", header=None, index_col=0)
    coords.columns = [
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    ]

    adata.obs = pd.merge(
        adata.obs, coords, how="left", left_index=True, right_index=True
    )
    adata.obsm[Key.obsm.spatial] = adata.obs[
        ["pxl_row_in_fullres", "pxl_col_in_fullres"]
    ].values
    adata.obs.drop(columns=["pxl_row_in_fullres", "pxl_col_in_fullres"], inplace=True)

    return adata


def add_spatial_assay(
    adata: AnnData, df_spatial: pd.DataFrame, library_id: str
) -> AnnData:
    """
    Add spatial assay to anndata

    Parameters
    ----------
    adata : AnnData
        Input anndata object
    df_spatial : pd.DataFrame
        A dataframe including spatial coordinates information
    library_id: str
        Library Id

    Returns
    -------
    AnnData
        Output anndata object
    """

    # load count matrix
    adata.uns[Key.obsm.spatial] = {library_id: {}}
    adata.obsm[Key.obsm.spatial] = adata.obs[["x", "y"]].values

    adata.obs.drop(columns=["x", "y"], inplace=True)

    return adata


def read_spatial_pattern(spatial_pattern: str) -> pd.DataFrame:
    """
    Read spsatial pattern

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

    filename = pkg_resources.resource_stream(__name__, f"data/{spatial_pattern}.csv")

    return pd.read_csv(filename, index_col=0)
