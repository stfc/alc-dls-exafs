"""Fit B building blocks: exact path pooling and the Larch adapter (no FEFF)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from larch import Group
from larch.xafs.feffit import FeffitDataSet, TransformGroup, feffit
from lmfit import Parameters  # type: ignore[import-untyped]

from md_exafs.fit_static import pool_paths, to_feffit_path
from md_exafs.paths import PathResult, path_chi

K_NATIVE = np.arange(0.0, 20.0001, 0.1)
K_OUT = np.arange(0.0, 15.05, 0.05)


def _feff_data(shift: float = 0.0) -> np.ndarray:
    """Smooth, FEFF-like columns (real_phc, mag, pha, red, lam, rep)."""
    k = K_NATIVE
    return np.column_stack(
        [
            -0.2 + 0.01 * k + shift,
            0.8 * np.exp(-0.05 * k) + 0.1 * shift,
            1.5 + 0.12 * k - 0.002 * k**2 + 2.0 * shift,
            np.full_like(k, 0.9),
            6.0 + 0.4 * k,
            0.5 + 0.98 * k,
        ]
    )


def _path(
    site: int,
    r_eff: float,
    nlegs: int = 2,
    degeneracy: float = 2.0,
    scatterer: str = "C",
    angle: float | None = None,
    shift: float = 0.0,
) -> PathResult:
    return PathResult(
        frame_idx=0,
        site_idx=site,
        r_eff=r_eff,
        nlegs=nlegs,
        degeneracy=degeneracy,
        scatterer=scatterer,
        cw_ratio=50.0,
        k=K_NATIVE,
        feff_data=_feff_data(shift),
        angle=angle,
    )


def _fp_chi(pooled: list, **kw: float) -> np.ndarray:
    total = np.zeros_like(K_OUT)
    for p in pooled:
        fp = to_feffit_path(p, **kw)
        fp._calc_chi(k=K_OUT)
        total += fp.chi
    return total


def test_pool_paths_exact_complex_sum_and_parameter_avg_bug():
    n_abs = 4
    cluster = [
        _path(0, 2.500, degeneracy=2.0),
        _path(1, 2.512, degeneracy=2.0),
        _path(2, 2.518, degeneracy=1.0),
        _path(3, 2.506, degeneracy=2.0),
    ]
    partial = [_path(0, 3.400, scatterer="O", degeneracy=1.0, shift=0.3)]
    partial.append(_path(1, 3.412, scatterer="O", degeneracy=1.0, shift=0.3))
    # Same non-isosceles triangle listed in both orientations (swapped labels).
    triangle = [
        _path(0, 3.800, 3, 2.0, "C-O", angle=70.0, shift=-0.2),
        _path(1, 3.812, 3, 2.0, "O-C", angle=70.0, shift=-0.2),
    ]
    members = cluster + partial + triangle

    pooled = pool_paths(members, n_abs, r_tol=0.03)
    assert len(pooled) == 3
    assert len({p.label for p in pooled}) == 3
    by_nlegs = {p.nlegs: p for p in pooled if p.scatterer != "O"}
    tri = by_nlegs[3]
    assert tri.count == 2
    assert tri.degeneracy == sum(p.degeneracy for p in triangle) / n_abs  # no 2x
    part = next(p for p in pooled if p.scatterer == "O")
    assert part.degeneracy == sum(p.degeneracy for p in partial) / n_abs
    assert by_nlegs[2].degeneracy == sum(p.degeneracy for p in cluster) / n_abs

    def exact(**kw: float) -> np.ndarray:
        return (
            sum(
                path_chi(p.k, p.feff_data, p.r_eff, p.degeneracy, K_OUT, **kw)
                for p in members
            )
            / n_abs
        )

    # Static limit: only the spline resampling of the pooled columns differs.
    tight = 1e-6
    assert np.allclose(_fp_chi(pooled), exact(), atol=tight, rtol=0)

    # With dE0, sigma2 and dR the members' different R matter at second order.
    loose = 5e-4
    pars = {"sigma2": 0.005, "e0": 2.5, "deltar": 0.02}
    assert np.allclose(
        _fp_chi(pooled, **pars),
        exact(sigma2=0.005, e0_shift=2.5, deltar=0.02),
        atol=loose,
        rtol=0,
    )

    # Averaging the parameters instead of summing amplitudes loses the dephasing.
    averaged = np.zeros_like(K_OUT)
    for group in (cluster, partial, triangle):
        deg = sum(p.degeneracy for p in group)
        r_mean = float(np.mean([p.r_eff for p in group]))
        data = np.mean([p.feff_data for p in group], axis=0)
        averaged += path_chi(K_NATIVE, data, r_mean, deg / n_abs, K_OUT)
    assert not np.allclose(averaged, exact(), atol=tight, rtol=0)


def test_to_feffit_path_parity_and_larch_feffit_roundtrip():
    member = _path(0, 2.55, degeneracy=6.0)
    (pooled,) = pool_paths([member], 1)
    fp = to_feffit_path(pooled)
    assert "pooled:" in repr(fp)
    assert copy.deepcopy(fp)._feffdat.reff == member.r_eff

    for kw, ref in (
        ({}, {}),
        (
            {"s02": 0.8, "e0": 3.0, "sigma2": 0.006, "deltar": -0.03},
            {"s02": 0.8, "e0_shift": 3.0, "sigma2": 0.006, "deltar": -0.03},
        ),
    ):
        fp = to_feffit_path(pooled, **kw)
        fp._calc_chi(k=K_OUT)
        expected = path_chi(
            member.k, member.feff_data, member.r_eff, member.degeneracy, K_OUT, **ref
        )
        assert np.allclose(fp.chi, expected, atol=1e-6, rtol=0)
        assert np.allclose(copy.deepcopy(fp)._feffdat.amp, fp._feffdat.amp)

    # Real feffit on two pooled paths with a known s02, e0 and isotropic alpha.
    paths = [_path(0, 2.5, degeneracy=4.0), _path(0, 3.5, degeneracy=6.0, shift=0.2)]
    sigma2s = [0.004, 0.006]
    s02, e0, alpha = 0.85, 2.0, 0.01
    k_data = 0.05 * np.arange(401)
    chi = sum(
        path_chi(
            p.k, p.feff_data, p.r_eff, p.degeneracy, k_data,
            sigma2=s2, s02=s02, e0_shift=e0, deltar=alpha * p.r_eff,
        )
        for p, s2 in zip(paths, sigma2s, strict=True)
    )  # fmt: skip
    feff_paths = [
        to_feffit_path(
            pool_paths([p], 1)[0],
            s02="s02", e0="e0", deltar="alpha*reff", sigma2=s2,
        )
        for p, s2 in zip(paths, sigma2s, strict=True)
    ]  # fmt: skip
    params = Parameters()
    params.add("s02", value=1.0, vary=True)
    params.add("e0", value=0.0, vary=True)
    params.add("alpha", value=0.0, vary=True)
    trans = TransformGroup(
        kmin=3, kmax=14, dk=1, window="kaiser", kweight=[1, 2, 3],
        rmin=1.0, rmax=4.0, fitspace="r",
    )  # fmt: skip
    dataset = FeffitDataSet(
        data=Group(k=k_data, chi=chi),
        paths=feff_paths,
        transform=trans,
        epsilon_k=1e-3,
    )
    result = feffit(params, dataset)
    assert result.success
    assert abs(result.params["s02"].value - s02) < 1e-3
    assert abs(result.params["e0"].value - e0) < 0.05
    assert abs(result.params["alpha"].value - alpha) < 1e-3


def test_md_sigma2_collinear_3leg_snap_and_4leg_classification():
    from md_exafs.fit_static import _md_sigma2

    # 3-leg collinear paths at 5.11 A (static 0.1/179.8 vs MD folded 2.3/175.3 deg)
    p_a000 = _path(0, 5.11, nlegs=3, degeneracy=12.0, scatterer="Cu-Cu", angle=0.10)
    p_a180 = _path(0, 5.11, nlegs=3, degeneracy=24.0, scatterer="Cu-Cu", angle=179.77)

    # 4-leg paths at 5.11 A with explicit FEFF geometry: rattle, focus, thru_abs, bent
    rlegs = (2.555, 2.555, 2.555, 2.555)
    elems = ("Cu", "Cu", "Cu", "Cu")
    p_rattle = PathResult(
        frame_idx=0,
        site_idx=0,
        r_eff=5.11,
        nlegs=4,
        degeneracy=12.0,
        scatterer="Cu-Cu-Cu",
        cw_ratio=4.0,
        k=K_NATIVE,
        feff_data=_feff_data(),
        rlegs=rlegs,
        betas=(180.0, 180.0, 180.0, 180.0),
        leg_elements=elems,
    )
    p_focus = PathResult(
        frame_idx=0,
        site_idx=0,
        r_eff=5.11,
        nlegs=4,
        degeneracy=12.0,
        scatterer="Cu-Cu-Cu",
        cw_ratio=33.0,
        k=K_NATIVE,
        feff_data=_feff_data(0.1),
        rlegs=rlegs,
        betas=(180.0, 0.2, 180.0, 0.2),
        leg_elements=elems,
    )
    p_thru = PathResult(
        frame_idx=0,
        site_idx=0,
        r_eff=5.11,
        nlegs=4,
        degeneracy=12.0,
        scatterer="Cu-Cu-Cu",
        cw_ratio=19.0,
        k=K_NATIVE,
        feff_data=_feff_data(0.2),
        rlegs=rlegs,
        betas=(0.2, 180.0, 0.2, 180.0),
        leg_elements=elems,
    )
    p_bent = PathResult(
        frame_idx=0,
        site_idx=0,
        r_eff=5.11,
        nlegs=4,
        degeneracy=60.0,
        scatterer="Cu-Cu-Cu",
        cw_ratio=0.8,
        k=K_NATIVE,
        feff_data=_feff_data(0.3),
        rlegs=rlegs,
        betas=(60.0, 180.0, 60.0, 180.0),
        leg_elements=elems,
    )

    pooled = pool_paths(
        [p_a000, p_a180, p_rattle, p_focus, p_thru, p_bent], n_abs=1, absorber="Cu"
    )
    by_kind = {
        (p.nlegs, p.path_kind, round(p.angle or -1)): (i, p)
        for i, p in enumerate(pooled)
    }
    assert len(pooled) == 6

    res_2b = [
        {"type": "Cu-Cu", "reff": 2.555, "count": 120, "sigma2_thermal_A2": 0.0073},
        {"type": "Cu-Cu", "reff": 5.110, "count": 120, "sigma2_thermal_A2": 0.0094},
    ]
    res_3b = [
        {
            "type": "Cu-Cu-Cu",
            "reff": 5.112,
            "count": 60,
            "angle": 2.3,
            "scatterer_angles_deg": (2.3, 2.3),
            "sigma2_thermal_A2": 0.0098,
        },
        {
            "type": "Cu-Cu-Cu",
            "reff": 5.113,
            "count": 120,
            "angle": 175.3,
            "scatterer_angles_deg": (2.4, 175.3),
            "sigma2_thermal_A2": 0.0093,
        },
    ]

    matched, _warns = _md_sigma2(pooled, res_2b, res_3b, match_r_tol=0.15)
    idx_a000 = by_kind[(3, None, 0)][0]
    idx_a180 = by_kind[(3, None, 180)][0]
    idx_rattle = by_kind[(4, "rattle", -1)][0]
    idx_focus = by_kind[(4, "collinear_focus", -1)][0]
    idx_thru = by_kind[(4, "collinear_thru_abs", -1)][0]
    idx_bent = by_kind[(4, "bent", -1)][0]

    assert matched[idx_a000]["sigma2"] == pytest.approx(0.0098)
    assert matched[idx_a180]["sigma2"] == pytest.approx(0.0093)
    assert matched[idx_rattle]["sigma2"] == pytest.approx(4.0 * 0.0073)
    assert matched[idx_focus]["sigma2"] == pytest.approx(0.0094)
    assert matched[idx_thru]["sigma2"] == pytest.approx(0.0098)
    assert idx_bent not in matched
