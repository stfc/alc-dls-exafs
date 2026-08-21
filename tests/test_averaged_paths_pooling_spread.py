"""Tests that pooling-spread diagnostics (r_eff_std / angle_std) survive the
full FEFF path pipeline: PathAggregator -> PathContribution ->
AveragedPathsStore -> HDF5 attrs.

These fields answer "how tightly did the per-frame FEFF geometries that got
pooled into this path actually agree" — the FEFF-side analogue of the DW
side's sigma2/angle_var pooling diagnostics.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from larch import Group

from larch_cli_wrapper.exafs_data import PathAggregator
from larch_cli_wrapper.hdf5_store import AveragedPathsStore

_FOURIER_PARAMS = {
    "kmin": 0.0,
    "kmax": 10.0,
    "kweight": 2,
    "dk": 1.0,
    "window": "hanning",
}


def _sample(r_eff, nlegs, scatterer, *, angle=None, chi_scale=1.0):
    k = np.linspace(0.0, 10.0, 50)
    chi = chi_scale * np.sin(2 * k)
    return {
        "k": k,
        "chi": chi,
        "r_eff": r_eff,
        "nlegs": nlegs,
        "degeneracy": 1.0,
        "scatterer": scatterer,
        "cw_ratio": 1.0,
        "angle": angle,
    }


class TestPathContributionSpread:
    def test_r_eff_std_reflects_pooled_spread(self):
        agg = PathAggregator()
        agg.add({"a": _sample(2.80, 2, "Cl")})
        agg.add({"b": _sample(2.90, 2, "Cl")})
        result = agg.finalize(_FOURIER_PARAMS)

        pc = next(iter(result.values()))
        assert pc.n_samples == 2
        # std of [2.80, 2.90] with ddof=0.
        assert abs(pc.r_eff_std - 0.05) < 1e-9

    def test_single_sample_has_zero_std(self):
        agg = PathAggregator()
        agg.add({"a": _sample(2.85, 2, "Cl")})
        result = agg.finalize(_FOURIER_PARAMS)

        pc = next(iter(result.values()))
        assert pc.r_eff_std == 0.0

    def test_angle_std_reflects_pooled_spread(self):
        agg = PathAggregator()
        agg.add({"a": _sample(3.50, 3, "N-C", angle=155.0)})
        agg.add({"b": _sample(3.52, 3, "N-C", angle=165.0)})
        result = agg.finalize(_FOURIER_PARAMS)

        pc = next(iter(result.values()))
        assert abs(pc.angle_std - 5.0) < 1e-9

    def test_2body_angle_std_is_none(self):
        agg = PathAggregator()
        agg.add({"a": _sample(2.85, 2, "Cl")})
        result = agg.finalize(_FOURIER_PARAMS)

        pc = next(iter(result.values()))
        assert pc.angle_std is None


class TestHDF5Persistence:
    def test_r_eff_std_and_angle_std_persisted_as_attrs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "averaged_paths.h5"
            agg = PathAggregator()
            agg.add({"a": _sample(3.50, 3, "N-C", angle=150.0)})
            agg.add({"b": _sample(3.60, 3, "N-C", angle=160.0)})
            contribs = agg.finalize(_FOURIER_PARAMS)

            store = AveragedPathsStore(hdf5_path)
            total = Group(k=np.linspace(0, 10, 50), chi=np.zeros(50))
            store.write_average("overall_average", total, contribs, n_total=2)
            store.close()

            import h5py

            with h5py.File(hdf5_path, "r") as fh:
                paths_grp = fh["overall_average"]["paths"]
                path_key = next(iter(paths_grp.keys()))
                attrs = dict(paths_grp[path_key].attrs)

        assert "r_eff_std" in attrs
        assert abs(float(attrs["r_eff_std"]) - 0.05) < 1e-9
        assert "angle_std" in attrs
        assert abs(float(attrs["angle_std"]) - 5.0) < 1e-9

    def test_r_eff_std_persisted_even_when_zero(self):
        """A single-sample (zero-spread) path still writes r_eff_std=0.0,
        distinguishing "no spread" from "spread unknown" (older data)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "averaged_paths.h5"
            agg = PathAggregator()
            agg.add({"a": _sample(2.85, 2, "Cl")})
            contribs = agg.finalize(_FOURIER_PARAMS)

            store = AveragedPathsStore(hdf5_path)
            total = Group(k=np.linspace(0, 10, 50), chi=np.zeros(50))
            store.write_average("overall_average", total, contribs, n_total=1)
            store.close()

            import h5py

            with h5py.File(hdf5_path, "r") as fh:
                paths_grp = fh["overall_average"]["paths"]
                path_key = next(iter(paths_grp.keys()))
                attrs = dict(paths_grp[path_key].attrs)

        assert float(attrs["r_eff_std"]) == 0.0
        assert "angle_std" not in attrs  # 2-body path has no angle at all
