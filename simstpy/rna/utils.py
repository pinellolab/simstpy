"""Utils functions"""
import pandas as pd
from anndata import AnnData
from pathlib import Path
import json

from squidpy.read._utils import _load_image
from squidpy._constants._pkg_constants import Key


def add_spatial_assay_10x(adata: AnnData, library_id: str, image_path: str) -> AnnData:
    """
    Add spatial assay to AnnData

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
        _description_
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
