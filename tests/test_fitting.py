"""Test data fitting functions"""

import squidpy as sq
import numpy as np
import simstpy as sim

def test_fit_library_size():
    """
    Test fit library size
    """

    # load the pre-processed dataset
    adata = sq.datasets.visium_hne_adata()
    adata.layers['counts'] = adata.X.copy()

    params = sim.rna.fit_library_size(adata.layers['counts'])

    assert np.linalg.norm(np.asarray(params) -
                          [0.13651185, 0.0, 6631.854]) < 1e-5
