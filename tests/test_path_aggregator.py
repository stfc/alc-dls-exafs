"""Tests for :class:`larch_cli_wrapper.exafs_data.PathAggregator`.

``PathAggregator`` pools raw per-frame FEFF path samples into averaged
path populations. It replaced a fixed-width R-bin hash (``make_path_key``)
with proper clustering to fix two correctness bugs:

* the same physical path landing in different "bins" purely because its
  ``r_eff`` straddled a bin edge in different frames, and
* two geometrically distinct 3-body paths (different angle) at a
  coincidentally similar ``r_eff`` being silently averaged together.
"""

from __future__ import annotations

import numpy as np

from larch_cli_wrapper.exafs_data import PathAggregator, _has_angle

_FOURIER_PARAMS = {
    "kmin": 0.0,
    "kmax": 10.0,
    "kweight": 2,
    "dk": 1.0,
    "window": "hanning",
}


def _sample(r_eff, nlegs, scatterer, *, angle=None, cw_ratio=1.0, chi_scale=1.0):
    k = np.linspace(0.0, 10.0, 50)
    chi = chi_scale * np.sin(2 * k)
    return {
        "k": k,
        "chi": chi,
        "r_eff": r_eff,
        "nlegs": nlegs,
        "degeneracy": 1.0,
        "scatterer": scatterer,
        "cw_ratio": cw_ratio,
        "angle": angle,
    }


class TestScattererCanonicalization:
    def test_reversed_scatterer_order_pools_together(self):
        """'N-C' and 'C-N' from different frames must pool as one path."""
        agg = PathAggregator()
        agg.add({"a": _sample(3.50, 3, "N-C", angle=160.0)})
        agg.add({"b": _sample(3.52, 3, "C-N", angle=158.0)})

        result = agg.finalize(_FOURIER_PARAMS)

        assert len(result) == 1
        pc = next(iter(result.values()))
        assert pc.n_samples == 2


class TestDistanceClustering:
    def test_reff_near_bin_edge_still_pools_together(self):
        """Samples straddling the old fixed-bin edge must not be split.

        With the old ``round(r_eff / 0.15) * 0.15`` binning, 2.849 and 2.851
        Å land in different bins purely from floating-point placement near
        the boundary; proper clustering must keep them together.
        """
        agg = PathAggregator()
        agg.add({"a": _sample(2.849, 2, "Cl")})
        agg.add({"b": _sample(2.851, 2, "Cl")})

        result = agg.finalize(_FOURIER_PARAMS)

        assert len(result) == 1
        pc = next(iter(result.values()))
        assert pc.n_samples == 2

    def test_distant_reff_values_split_into_separate_paths(self):
        agg = PathAggregator()
        agg.add({"a": _sample(2.85, 2, "Cl")})
        agg.add({"b": _sample(4.50, 2, "Cl")})

        result = agg.finalize(_FOURIER_PARAMS)

        assert len(result) == 2
        n_samples = sorted(pc.n_samples for pc in result.values())
        assert n_samples == [1, 1]

    def test_different_scatterer_elements_never_pool(self):
        agg = PathAggregator()
        agg.add({"a": _sample(2.85, 2, "Cl")})
        agg.add({"b": _sample(2.85, 2, "K")})

        result = agg.finalize(_FOURIER_PARAMS)

        assert len(result) == 2


class TestAngleAwareClustering:
    def test_same_reff_different_angle_paths_stay_separate(self):
        """Two geometrically distinct 3-body paths must not be averaged
        together just because their Reff coincides."""
        agg = PathAggregator()
        agg.add({"a": _sample(3.50, 3, "N-C", angle=90.0, chi_scale=1.0)})
        agg.add({"b": _sample(3.50, 3, "N-C", angle=90.0, chi_scale=1.02)})
        agg.add({"c": _sample(3.51, 3, "N-C", angle=175.0, chi_scale=0.3)})

        result = agg.finalize(_FOURIER_PARAMS)

        assert len(result) == 2
        n_samples = sorted(pc.n_samples for pc in result.values())
        assert n_samples == [1, 2]

    def test_averaged_angle_is_stored_on_path_contribution(self):
        agg = PathAggregator()
        agg.add({"a": _sample(3.50, 3, "N-C", angle=160.0)})
        agg.add({"b": _sample(3.52, 3, "N-C", angle=158.0)})

        result = agg.finalize(_FOURIER_PARAMS)

        pc = next(iter(result.values()))
        assert pc.angle == 159.0

    def test_2body_paths_have_no_angle(self):
        agg = PathAggregator()
        agg.add({"a": _sample(2.85, 2, "Cl")})

        result = agg.finalize(_FOURIER_PARAMS)

        pc = next(iter(result.values()))
        assert pc.angle is None

    def test_missing_angle_falls_back_to_distance_only_bucket(self):
        """3-body samples without an angle (e.g. older data) still pool by
        distance alone, and don't crash or get silently dropped."""
        agg = PathAggregator()
        agg.add({"a": _sample(3.50, 3, "N-C", angle=None)})
        agg.add({"b": _sample(3.51, 3, "N-C", angle=None)})

        result = agg.finalize(_FOURIER_PARAMS)

        assert len(result) == 1
        pc = next(iter(result.values()))
        assert pc.n_samples == 2
        assert pc.angle is None

    def test_angled_and_angleless_samples_kept_in_separate_populations(self):
        agg = PathAggregator()
        agg.add({"a": _sample(3.50, 3, "N-C", angle=160.0)})
        agg.add({"b": _sample(3.51, 3, "N-C", angle=None)})

        result = agg.finalize(_FOURIER_PARAMS)

        assert len(result) == 2

    def test_nan_angle_treated_same_as_none(self):
        """A stray NaN (the on-disk HDF5 sentinel) must not be treated as a
        real angle value if it ever reaches PathAggregator directly, bypassing
        the None-conversion normally done by iter_path_contributions."""
        agg = PathAggregator()
        agg.add({"a": _sample(3.50, 3, "N-C", angle=float("nan"))})
        agg.add({"b": _sample(3.51, 3, "N-C", angle=None)})

        result = agg.finalize(_FOURIER_PARAMS)

        # Both samples fall into the same distance-only ("no angle") bucket.
        assert len(result) == 1
        pc = next(iter(result.values()))
        assert pc.n_samples == 2
        assert pc.angle is None


class TestHasAngle:
    def test_none_is_not_an_angle(self):
        assert _has_angle({"angle": None}) is False

    def test_missing_key_is_not_an_angle(self):
        assert _has_angle({}) is False

    def test_nan_is_not_an_angle(self):
        assert _has_angle({"angle": float("nan")}) is False

    def test_finite_value_is_an_angle(self):
        assert _has_angle({"angle": 160.0}) is True

    def test_zero_is_an_angle(self):
        assert _has_angle({"angle": 0.0}) is True
