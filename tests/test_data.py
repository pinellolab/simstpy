"""Tests for `simstpy data` module."""

import simstpy as sim

def test_read_spatial_pattern():
    """
    Test reading spatial pattern
    """

    sp_patterns = sim.spatial.get_all_patterns()

    for sp_pattern in sp_patterns:
        sim.spatial.read_pattern(sp_pattern)
