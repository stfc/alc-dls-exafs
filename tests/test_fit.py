"""Tests for md_exafs.fit.fit_ensemble.

The archive's overall chi(k) and the synthetic "experiments" both come from
``paths.path_chi`` (the exact EXAFS equation, with Larch's delta-E0 sign), the
experiments with known S0^2 and delta-E0, so recovery also guards the sign
convention of ``scaled_chi_arrays``.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from md_exafs import EnsembleWriter
from md_exafs.execution import DEFAULT_K_GRID
from md_exafs.fit import fit_ensemble
from md_exafs.paths import path_chi

FT = {"kmin": 3.0, "kmax": 13.0, "kweight": 2, "dk": 1.0, "rmax": 5.5}
SHELLS = ((2.5, 6.0, 0.004), (3.5, 4.0, 0.006))  # r_eff, degeneracy, sigma2
K_NATIVE = np.linspace(0.0, 20.0, 41)
RRANGE = (1.2, 3.8)


def _feff_data() -> np.ndarray:
    data = np.zeros((len(K_NATIVE), 6))
    data[:, 0] = 0.3  # real_phc
    data[:, 1] = 0.8 * np.exp(-0.5 * ((K_NATIVE - 6.0) / 6.0) ** 2) + 0.1  # mag_feff
    data[:, 2] = -0.2 * K_NATIVE  # pha_feff
    data[:, 3] = 0.9  # red_fact
    data[:, 4] = 8.0 + 0.2 * K_NATIVE  # lam
    data[:, 5] = K_NATIVE  # rep
    return data


def _chi(k: np.ndarray, s02: float = 1.0, e0_shift: float = 0.0) -> np.ndarray:
    return sum(
        path_chi(
            K_NATIVE, _feff_data(), r, n, k, sigma2=sig2, s02=s02, e0_shift=e0_shift
        )
        for r, n, sig2 in SHELLS
    )


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    path = tmp_path / "ens.h5"
    with EnsembleWriter(
        path, k_grid=DEFAULT_K_GRID, fourier_params=FT, radius=6.0
    ) as w:
        w.set_overall_average(DEFAULT_K_GRID, _chi(DEFAULT_K_GRID))
    return path


def test_fit_ensemble_recovers_s02_and_e0_shift(archive: Path):
    exp_k = np.arange(0.0, 16.0, 0.05)
    n_idp = 1.0 + 2.0 * (RRANGE[1] - RRANGE[0]) * (FT["kmax"] - FT["kmin"]) / np.pi
    for s02_true, e0_true, space in ((0.82, 3.5, "r"), (0.90, -2.5, "k")):
        experiment = {"k": exp_k, "chi": _chi(exp_k, s02_true, e0_true)}
        res = fit_ensemble(
            archive, experiment, kweights=(1, 2, 3), rrange=RRANGE, fit_space=space
        )

        assert json.loads(json.dumps(res, allow_nan=False)) == res
        assert res["success"] is True
        assert res["fit_space"] == space
        assert res["nvarys"] == 2
        assert res["n_idp"] == pytest.approx(n_idp)
        assert res["rmax_valid"] == pytest.approx(5.5)
        assert res["params"]["s02"]["value"] == pytest.approx(s02_true, abs=8e-3)
        assert res["params"]["e0_shift"]["value"] == pytest.approx(e0_true, abs=0.03)
        for name in ("s02", "e0_shift"):
            assert 0.0 < res["params"][name]["stderr"] < np.inf
        assert set(res["correlations"]) == {"s02", "e0_shift"}
        for key in ("rfactor", "rfactor_r", "rfactor_k"):
            assert 0.0 <= res[key] < 1e-3

        # Both curves sit on the archive grid and agree where the data overlap.
        for curve in ("experiment", "theory"):
            np.testing.assert_allclose(res[curve]["k"], DEFAULT_K_GRID)
        window = (DEFAULT_K_GRID > FT["kmin"]) & (DEFAULT_K_GRID < FT["kmax"])
        exp_chi, theory_chi = (
            np.array(res[c]["chi"], dtype=float)[window]
            for c in ("experiment", "theory")
        )
        np.testing.assert_allclose(theory_chi, exp_chi, atol=2e-3)
