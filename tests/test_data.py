"""Tests for `simstpy data` module."""

import simstpy as sim

def test_read_spatial_pattern():
    """
    Test reading spatial pattern
    """

    sp_patterns = sim.get_all_spatial_patterns()

    for sp_pattern in sp_patterns:
        sim.read_spatial_pattern(sp_pattern)
