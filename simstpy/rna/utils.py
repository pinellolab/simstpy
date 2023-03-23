"""Helper functions"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from anndata import AnnData
from squidpy.read._utils import _load_image
from squidpy._constants._pkg_constants import Key


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
    coords = pd.read_csv(path / "tissue_positions_list.txt",
                         header=None, index_col=0)
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
    adata.obs.drop(columns=["pxl_row_in_fullres",
                   "pxl_col_in_fullres"], inplace=True)

    return adata


def check_pos_semidefinite(cov: np.array) -> np.array:
    """
    Check if the generated covariance is positive and semidefinite

    Ref: https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning
    
    Parameters
    ----------
    cov : np.array
        Input covariance matrix

    Returns
    -------
    np.array
        Returned matrix
    """
    
    min_eig = np.min(np.real(np.linalg.eigvals(cov)))
    
    if min_eig < 0:
        cov -= 10*min_eig * np.eye(*cov.shape)
        
    return cov