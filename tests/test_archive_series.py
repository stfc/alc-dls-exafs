"""Tests for md_exafs.archive_series on synthetic ensemble archives."""

import json
from pathlib import Path

import numpy as np
import pytest

from md_exafs import ArchiveReader, EnsembleWriter
from md_exafs.archive_series import (
    ViewUnavailableError,
    archive_series,
    series_to_vega_lite,
)
from md_exafs.spectra import average_chi_arrays, xftf_arrays

K = np.arange(0.05, 16.0, 0.05)
FT = {"kmin": 3.0, "kmax": 12.0, "kweight": 2, "dk": 1.0, "rmax": 6.0}


def _chi(frame: int, site: int) -> np.ndarray:
    amp = 0.5 + 0.05 * frame + 0.01 * site
    return amp * np.sin(2 * 2.5 * K + 0.3 * site) * np.exp(-0.02 * K**2)


def _write(
    path: Path,
    n_frames: int = 3,
    n_sites: int = 2,
    *,
    overall_scale: float = 1.0,
    ragged_last_frame: bool = False,
) -> Path:
    """Archive whose aggregates are built the way the pipeline builds them.

    With ``ragged_last_frame`` every spectrum of the last frame stops at k=10.
    """
    k_of, chi_of = {}, {}
    for f in range(n_frames):
        for s in range(n_sites):
            keep = K <= 10.0 if ragged_last_frame and f == n_frames - 1 else K > 0
            k_of[f, s], chi_of[f, s] = K[keep], _chi(f, s)[keep]

    def avg(keys):
        return average_chi_arrays(
            [k_of[x] for x in keys], [chi_of[x] for x in keys], k_grid=K
        )

    overall = avg(list(k_of))
    with EnsembleWriter(path, k_grid=K, fourier_params=FT) as w:
        w.set_overall_average(
            overall.k,
            overall.mean * overall_scale,
            chi_std=overall.std,
            n_contributors=overall.n_contributors,
        )
        for f in range(n_frames):
            a = avg([(f, s) for s in range(n_sites)])
            w.add_frame_average(f, a.k, a.mean, n_contributors=a.n_contributors)
        for s in range(n_sites):
            a = avg([(f, s) for f in range(n_frames)])
            w.add_site_average(s, a.k, a.mean, n_contributors=a.n_contributors)
    return path


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    return _write(tmp_path / "ens.h5")


def test_series_is_versioned_and_json_safe(archive: Path):
    series = archive_series(archive)
    assert series["schema_version"] == 1
    text = json.dumps(series, allow_nan=False)
    assert json.loads(text) == series
    assert series["ft_params"]["kweight"] == 2
    assert series["units"]["k"] == "A^-1"
    assert series["units"]["chir"] == "A^-3"
    assert series["source"] == {"n_sites": 2, "n_frames": 3}
    assert series["warnings"] == []


def test_overall_matches_stored_arrays(archive: Path):
    series = archive_series(archive)
    reader = ArchiveReader(archive)
    o = series["overall"]
    np.testing.assert_allclose(o["k"], reader.k)
    np.testing.assert_allclose(o["chi"], reader.chi)
    np.testing.assert_allclose(o["r"], reader.r)
    np.testing.assert_allclose(o["chir_mag"], reader.chir_mag)
    assert o["n_contributors"][0] == 6
    assert o["chi_std"][0] is not None


def test_chir_is_transform_of_kweighted_chi(archive: Path):
    series = archive_series(archive)
    ft = xftf_arrays(
        np.array(series["overall"]["k"]), np.array(series["overall"]["chi"]), FT
    )
    np.testing.assert_allclose(series["overall"]["chir_mag"], ft["chir_mag"])


def test_healthy_archive_check_ok(archive: Path):
    check = archive_series(archive)["checks"]["stored_vs_recomputed"]
    assert check["status"] == "ok"
    assert check["max_rel_deviation"] < 1e-9
    assert check["weighting"] == "n_contributors"


def test_stale_overall_triggers_warning(tmp_path: Path):
    path = _write(tmp_path / "stale.h5", overall_scale=1.1)
    series = archive_series(path)
    check = series["checks"]["stored_vs_recomputed"]
    assert check["status"] == "mismatch"
    assert check["max_rel_deviation"] == pytest.approx(0.1 / 1.1, rel=1e-2)
    assert any("stale" in w for w in series["warnings"])
    # the stored (stale) average is the one plotted
    stored = ArchiveReader(path).chi
    np.testing.assert_allclose(series["overall"]["chi"], stored)


def test_rel_tol_is_a_parameter(tmp_path: Path):
    path = _write(tmp_path / "stale.h5", overall_scale=1.001)
    assert archive_series(path, rel_tol=1e-2)["warnings"] == []
    assert archive_series(path, rel_tol=1e-5)["warnings"]


def test_ragged_ensemble_does_not_trigger_warning(tmp_path: Path):
    path = _write(tmp_path / "ragged.h5", ragged_last_frame=True)
    series = archive_series(path)
    check = series["checks"]["stored_vs_recomputed"]
    assert check["status"] == "ok", series["warnings"]
    assert series["overall"]["n_contributors"][0] == 6
    assert series["overall"]["n_contributors"][-1] == 4
    assert series["overall"]["chi"][-1] is not None


def test_nan_becomes_null(tmp_path: Path):
    path = tmp_path / "nan.h5"
    chi = np.sin(K)
    chi_nan = chi.copy()
    chi_nan[-5:] = np.nan
    with EnsembleWriter(path, k_grid=K, fourier_params=FT) as w:
        w.set_overall_average(
            K, chi_nan, chi_std=np.where(np.isnan(chi_nan), np.nan, 0.1)
        )
    series = archive_series(path)
    assert series["overall"]["chi"][-1] is None
    assert series["overall"]["chi_std"][-1] is None
    assert series["overall"]["chi"][0] is not None
    json.dumps(series, allow_nan=False)
    assert any("No frame averages" in w for w in series["warnings"])
    assert series["checks"]["stored_vs_recomputed"]["status"] == "skipped"


def test_frame_cap_is_even_and_recorded(tmp_path: Path):
    path = _write(tmp_path / "many.h5", n_frames=45, n_sites=1)
    series = archive_series(path, max_frames=20)
    shown = series["frames"]["shown_indices"]
    assert series["frames"]["total"] == 45
    assert len(shown) == 20
    assert shown[0] == 0 and shown[-1] == 44
    assert shown == sorted(set(shown))
    gaps = np.diff(shown)
    assert gaps.max() - gaps.min() <= 1
    assert [s["frame_idx"] for s in series["frames"]["spectra"]] == shown
    assert archive_series(path, max_frames=0)["frames"]["spectra"] == []


def test_few_frames_all_shown(archive: Path):
    series = archive_series(archive)
    assert series["frames"]["shown_indices"] == [0, 1, 2]
    assert series["sites"]["shown_indices"] == [0, 1]


def test_max_points_caps_and_keeps_peaks(archive: Path):
    full = archive_series(archive)
    thin = archive_series(archive, max_points=64)
    for key in ("k", "chi", "chi_std", "n_contributors"):
        assert len(thin["overall"][key]) <= 64
    assert len(thin["overall"]["r"]) <= 64
    assert len(thin["overall"]["chir_mag"]) == len(thin["overall"]["r"])
    assert thin["overall"]["k"][0] == full["overall"]["k"][0]
    assert thin["overall"]["k"][-1] == full["overall"]["k"][-1]
    assert max(thin["overall"]["chi"]) == max(full["overall"]["chi"])
    assert min(thin["overall"]["chi"]) == min(full["overall"]["chi"])
    assert max(thin["overall"]["chir_mag"]) == max(full["overall"]["chir_mag"])
    assert len(thin["frames"]["spectra"][0]["chi"]) <= 64
    assert thin["downsampling"]["max_points"] == 64


def test_bad_parameters(archive: Path, tmp_path: Path):
    with pytest.raises(ValueError):
        archive_series(archive, max_points=4)
    with pytest.raises(ValueError):
        archive_series(archive, rel_tol=0)
    with pytest.raises(FileNotFoundError):
        archive_series(tmp_path / "missing.h5")


def test_non_ensemble_archive_rejected(tmp_path: Path):
    import h5py

    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        f.create_group("meta")
    with pytest.raises(ValueError):
        archive_series(path)


def _datasets(spec: dict) -> list[list[dict]]:
    out = []
    if "data" in spec and "values" in spec["data"]:
        out.append(spec["data"]["values"])
    out.extend(spec.get("datasets", {}).values())
    return out


def _encodings(spec: dict) -> set[tuple[str, str]]:
    found = set()
    if "encoding" in spec:
        for channel, enc in spec["encoding"].items():
            if isinstance(enc, dict) and "field" in enc:
                found.add((channel, enc["field"]))
    for key in ("layer", "vconcat"):
        for sub in spec.get(key, []):
            found |= _encodings(sub)
    return found


@pytest.mark.parametrize("view", ["chi_k", "chi_r", "sites", "frames"])
def test_vega_lite_views(archive: Path, view: str):
    series = archive_series(archive)
    spec = series_to_vega_lite(series, view=view)
    json.dumps(spec, allow_nan=False)
    assert "vega-lite" in spec["$schema"]
    datasets = _datasets(spec)
    assert datasets and all(len(rows) > 0 for rows in datasets)
    fields = _encodings(spec)
    if view != "chi_r":
        assert ("x", "k") in fields
    if view == "chi_r":
        assert ("x", "r") in fields and ("y", "chir_mag") in fields
        labels = {row["path"] for rows in datasets for row in rows}
        assert labels == {"|χ(R)|", "Re χ(R)", "Im χ(R)"}
    if view == "chi_k":
        assert ("y", "chi_weighted") in fields
        assert ("y2", "hi") in fields
        assert ("y", "n") in fields
    if view == "frames":
        assert ("y", "chir_mag") in fields
        labels = {row["path"] for rows in datasets for row in rows}
        assert labels == {"frame 0", "frame 1", "frame 2"}


def test_vega_lite_nan_is_null(tmp_path: Path):
    path = tmp_path / "nan.h5"
    chi = np.sin(K)
    chi[-5:] = np.nan
    with EnsembleWriter(path, k_grid=K, fourier_params=FT) as w:
        w.set_overall_average(K, chi, chi_std=np.where(np.isnan(chi), np.nan, 0.1))
    spec = series_to_vega_lite(archive_series(path), view="chi_k")
    json.dumps(spec, allow_nan=False)


def test_vega_lite_errors(archive: Path, tmp_path: Path):
    series = archive_series(archive)
    with pytest.raises(ValueError):
        series_to_vega_lite(series, view="nope")
    series["frames"]["spectra"] = []
    with pytest.raises(ViewUnavailableError):
        series_to_vega_lite(series, view="frames")


def test_panels_have_independent_zoom(archive: Path):
    spec = series_to_vega_lite(archive_series(archive), view="frames")
    names = [p["name"] for p in spec["params"]]
    assert sorted(names) == ["zoom_k", "zoom_r"]


def test_vega_lite_carries_series_warnings_in_usermeta(tmp_path: Path):
    stale = _write(tmp_path / "stale.h5", overall_scale=1.2)
    series = archive_series(stale)
    spec = series_to_vega_lite(series, view="chi_r")
    meta = spec["usermeta"]
    assert meta["warnings"] == series["warnings"] and meta["warnings"]
    assert meta["ft_params"]["kweight"] == 2
    assert meta["checks"]["stored_vs_recomputed"]["status"] == "mismatch"
    assert meta["shown"]["frames"] == [0, 1, 2]
    assert meta["view"] == "chi_r"
