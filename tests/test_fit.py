"""Tests for md_exafs.fit.fit_ensemble and fit_site_group_weights.

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
from md_exafs.attribution import group_sites
from md_exafs.execution import DEFAULT_K_GRID
from md_exafs.fit import fit_ensemble, fit_site_group_weights
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


def _shell_chi(
    r: float, k: np.ndarray, s02: float = 1.0, e0_shift: float = 0.0
) -> np.ndarray:
    return path_chi(
        K_NATIVE, _feff_data(), r, 6.0, k, sigma2=0.004, s02=s02, e0_shift=e0_shift
    )


def test_fit_site_group_weights_recovers_simplex_weights(tmp_path: Path):
    radii = (2.2, 2.7, 3.3)
    path = tmp_path / "sites.h5"
    site_chis = [_shell_chi(r, DEFAULT_K_GRID) for r in radii for _ in range(2)]
    with EnsembleWriter(
        path, k_grid=DEFAULT_K_GRID, fourier_params=FT, radius=6.0
    ) as w:
        w.set_overall_average(DEFAULT_K_GRID, np.mean(site_chis, axis=0))
        for i, chi in enumerate(site_chis):
            w.add_site_average(i, DEFAULT_K_GRID, chi)
    groups = group_sites(path, rrange=(1.0, 4.0), n_groups=3)
    assert [g["n_sites"] for g in groups["groups"]] == [2, 2, 2]

    s02_true, e0_true = 0.85, 2.0
    w_true = {2.2: 0.60, 2.7: 0.25, 3.3: 0.15}
    exp_k = np.arange(0.0, 16.0, 0.05)
    experiment = {
        "k": exp_k,
        "chi": sum(
            w * _shell_chi(r, exp_k, s02_true, e0_true) for r, w in w_true.items()
        ),
    }
    # Each group is one shell; identify its planted weight by its site indices.
    planted = {
        tuple(g["site_indices"]): w_true[radii[g["site_indices"][0] // 2]]
        for g in groups["groups"]
    }

    res = fit_site_group_weights(
        path,
        experiment,
        groups,
        rrange=(1.0, 4.0),
        kweights=(1, 2, 3),
        s02=s02_true,
        e0_shift=e0_true,
    )
    assert json.loads(json.dumps(res, allow_nan=False)) == res
    assert res["success"] is True
    assert res["e0_shift_held_fixed"] is True
    fitted = [x["fitted_weight"] for x in res["weights"]]
    assert min(fitted) >= 0.0
    assert sum(fitted) == pytest.approx(1.0)
    for x in res["weights"]:
        assert x["md_weight"] == pytest.approx(1 / 3)
        assert x["fitted_weight"] == pytest.approx(
            planted[tuple(x["site_indices"])], abs=1e-2
        )
    assert res["rfactor_fit"] < 1e-4
    assert res["rfactor_fit"] < res["rfactor_md"]

    # With s02 and e0_shift omitted, Fit A supplies them.
    auto = fit_site_group_weights(path, experiment, groups, rrange=(1.0, 4.0))
    assert auto["success"] is True
    assert auto["rfactor_fit"] < auto["rfactor_md"]
