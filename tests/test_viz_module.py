"""Tests for md_exafs.viz (ADR 0007)."""

import numpy as np

from md_exafs.paths import PathResult
from md_exafs.viz import calculate_path_vectors, group_path_results


def test_group_path_results():
    p1 = PathResult(
        frame_idx=0,
        site_idx=0,
        r_eff=2.51,
        nlegs=2,
        degeneracy=12.0,
        scatterer="Cu",
        cw_ratio=100.0,
        k=np.linspace(0, 20, 10),
        feff_data=np.ones((10, 6)),
    )
    p2 = PathResult(
        frame_idx=1,
        site_idx=0,
        r_eff=2.53,
        nlegs=2,
        degeneracy=12.0,
        scatterer="Cu",
        cw_ratio=98.0,
        k=np.linspace(0, 20, 10),
        feff_data=np.ones((10, 6)),
    )
    grouped = group_path_results([p1, p2], r_bin_width=0.1)
    assert len(grouped) == 1
    assert grouped[0]["path_key"] == "Cu_2_2.50"
    assert grouped[0]["count"] == 2
    assert np.isclose(grouped[0]["r_eff"], 2.52)
    assert np.isclose(grouped[0]["cw_ratio"], 99.0)


def test_calculate_path_vectors():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )
    # Path: 0 -> 1 -> 2 -> 0
    segments = calculate_path_vectors(positions, [0, 1, 2, 0])
    assert len(segments) == 3
    assert np.isclose(segments[0]["length"], 2.0)
    assert np.isclose(segments[1]["length"], 2.0)
    assert np.isclose(segments[2]["length"], np.sqrt(8.0))
