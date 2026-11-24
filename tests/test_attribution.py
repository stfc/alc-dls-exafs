"""Tests for md_exafs.attribution on synthetic ensemble archives."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from md_exafs import EnsembleWriter
from md_exafs.attribution import group_sites, peak_attribution

K = np.arange(0.05, 16.0, 0.05)
R0 = 2.5
FT = {"kmin": 3.0, "kmax": 12.0, "kweight": 2, "dk": 1.0, "rmax": 6.0}


def _base(r: float, phase: float = 0.0) -> np.ndarray:
    return np.sin(2 * K * r + phase) / (K * r**2)


def _archive(
    path: Path,
    site_chis: list[np.ndarray],
    counts: list[int],
    top_paths: list[dict] | None = None,
    version: int | None = None,
) -> Path:
    chi = np.array(site_chis)
    n = np.array([np.full(K.shape, c, dtype=np.int32) for c in counts])
    overall = (n * chi).sum(axis=0) / n.sum(axis=0)
    with EnsembleWriter(path, k_grid=K, fourier_params=FT, radius=7.0) as w:
        w.set_overall_average(K, overall, n_contributors=n.sum(axis=0))
        for s, (c, nn) in enumerate(zip(chi, n, strict=True)):
            w.add_site_average(s, K, c, n_contributors=nn)
        if top_paths is not None:
            w.add_top_paths(top_paths, n_tasks=int(sum(counts)))
    if version is not None:
        with h5py.File(path, "r+") as f:
            f["meta"].attrs["format_version"] = version
    return path


def _path(key: str, chi: np.ndarray) -> dict:
    return {"path_key": key, "scatterer": "Cu", "nlegs": 2, "r_eff": R0, "chi": chi}


def test_peak_attribution_signed_projection_and_v4_guard(tmp_path: Path):
    p_in, p_out = 1.3 * _base(R0), -0.3 * _base(R0)
    total = p_in + p_out + 0.15 * _base(R0, 0.8)
    paths = [_path("p_in", p_in), _path("p_out", p_out)]
    archive = _archive(tmp_path / "v4.h5", [total], [3], top_paths=paths)

    res = peak_attribution(archive, by="paths", rrange=(1.0, 3.5))
    assert json.loads(json.dumps(res, allow_nan=False)) == res
    peak = min(res["peaks"], key=lambda p: abs(p["r_peak"] - R0))
    assert abs(peak["r_peak"] - R0) < 0.3
    shares = {s["id"]: s for s in peak["shares"]}
    assert sum(s["projection"] for s in peak["shares"]) + peak["unassigned"][
        "projection"
    ] == pytest.approx(1.0, abs=1e-6)
    assert peak["unassigned"]["projection"] != pytest.approx(0.0, abs=1e-3)
    assert shares["p_in"]["projection"] > 0
    assert shares["p_in"]["loo_delta"] > 0
    assert shares["p_out"]["projection"] < 0
    assert shares["p_out"]["loo_delta"] < 0

    sites = [_base(2.4), 0.8 * _base(2.5), 1.1 * _base(2.6)]
    multi = _archive(tmp_path / "sites.h5", sites, [2, 3, 4])
    by_sites = peak_attribution(multi, by="sites", rrange=(1.0, 3.5))
    assert by_sites["unassigned"]["norm_fraction"] < 1e-6
    assert by_sites["peaks"]
    for p in by_sites["peaks"]:
        total_share = (
            sum(s["projection"] for s in p["shares"]) + p["unassigned"]["projection"]
        )
        assert total_share == pytest.approx(1.0, abs=1e-6)

    v3 = _archive(tmp_path / "v3.h5", [total], [3], top_paths=paths, version=3)
    with pytest.raises(ValueError, match="version"):
        peak_attribution(v3, by="paths", rrange=(1.0, 3.5))


def test_group_sites_recovers_planted_populations(tmp_path: Path):
    rng = np.random.default_rng(0)
    chis = []
    for s in range(6):
        r, scale = (2.3, 1.0) if s % 2 == 0 else (2.8, 0.6)
        chis.append(scale * (1 + 0.02 * rng.standard_normal()) * _base(r))
    archive = _archive(tmp_path / "groups.h5", chis, [3] * 6)

    res = group_sites(archive, rrange=(1.0, 4.0), tol=0.2)
    assert json.loads(json.dumps(res, allow_nan=False)) == res
    assert len(res["groups"]) == 2
    assert {frozenset(g["site_indices"]) for g in res["groups"]} == {
        frozenset({0, 2, 4}),
        frozenset({1, 3, 5}),
    }
    assert sum(g["weight"] for g in res["groups"]) == pytest.approx(1.0)

    attr = peak_attribution(archive, by="site_groups", groups=res, rrange=(1.0, 4.0))
    assert attr["unassigned"]["norm_fraction"] < 1e-6
    assert attr["peaks"]
    for p in attr["peaks"]:
        total_share = (
            sum(s["projection"] for s in p["shares"]) + p["unassigned"]["projection"]
        )
        assert total_share == pytest.approx(1.0, abs=1e-6)
