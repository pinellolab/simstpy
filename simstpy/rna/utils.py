"""Helper functions"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from anndata import AnnData
from sklearn.gaussian_process.kernels import RBF
from squidpy.read._utils import _load_image
from squidpy._constants._pkg_constants import Key

def get_cov_from_rbf_kernel(coords: np.array, length_scale: float=1.0) -> np.array:
    """
    Generate covariance matrix from RBF kernel

    Parameters
    ----------
    coords : np.array
        Spatial coordinates
    length_scale : float, optional
        Length scale for RBF, by default 1.0

    Returns
    -------
    np.array
        Covariance matrix
    """

    rbf = RBF(length_scale=length_scale)

    cov = rbf(coords)

    return cov


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
