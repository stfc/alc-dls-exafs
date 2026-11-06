"""Tests for md_exafs.archive_series and series_to_vega_lite.

Synthetic ensemble archives written with EnsembleWriter; chi(k) is a 1/k
decaying sine with a known shell at R0, so the transform has a checkable peak.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from md_exafs import ArchiveReader, EnsembleWriter
from md_exafs.archive_series import (
    KWEIGHTS,
    RMAX_MAX,
    InvalidFtParamsError,
    archive_series,
)
from md_exafs.spectra import average_chi_arrays, xftf_arrays
from md_exafs.viz import ViewUnavailableError, series_to_vega_lite

K = np.arange(0.05, 16.0, 0.05)
R0 = 2.5
FT = {"kmin": 3.0, "kmax": 12.0, "kweight": 2, "dk": 1.0, "rmax": 6.0}
STORED = {**FT, "window": "kaiser"}
ALL = ("sites", "frames")


def _chi(frame: int, site: int) -> np.ndarray:
    scale = 1.0 + 0.05 * frame + 0.01 * site
    return scale * np.sin(2 * K * R0) / (K * R0**2)


def _archive(
    path: Path,
    *,
    keep: Callable[[np.ndarray], np.ndarray] = lambda k: k > 0,
    n_frames: int = 3,
    n_sites: int = 2,
) -> Path:
    """Archive whose spectra cover the k-points selected by ``keep``."""
    k_of, chi_of = {}, {}
    for f in range(n_frames):
        for s in range(n_sites):
            m = keep(K)
            k_of[f, s], chi_of[f, s] = K[m], _chi(f, s)[m]

    def avg(keys):
        return average_chi_arrays(
            [k_of[x] for x in keys], [chi_of[x] for x in keys], k_grid=K
        )

    overall = avg(list(k_of))
    with EnsembleWriter(path, k_grid=K, fourier_params=FT) as w:
        w.set_overall_average(
            overall.k,
            overall.mean,
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
    return _archive(tmp_path / "ens.h5")


# --- reading ---------------------------------------------------------------


def test_series_is_versioned_and_json_safe(archive: Path):
    series = archive_series(archive, include=ALL)
    assert series["schema_version"] == 1
    assert json.loads(json.dumps(series, allow_nan=False)) == series
    assert series["units"]["k"] == "A^-1"
    assert series["units"]["chir"] == "A^-3"
    assert series["source"] == {"n_sites": 2, "n_frames": 3}
    assert series["warnings"] == []


def test_overall_matches_stored_arrays(archive: Path):
    o = archive_series(archive)["overall"]
    reader = ArchiveReader(archive)
    np.testing.assert_allclose(o["k"], reader.k)
    np.testing.assert_allclose(o["chi"], reader.chi)
    np.testing.assert_allclose(o["r"], reader.r)
    np.testing.assert_allclose(o["chir_mag"], reader.chir_mag)
    assert o["n_contributors"][0] == 6
    assert o["chi_std"][0] is not None


def test_breakdowns_are_read_only_when_included(archive: Path):
    bare = archive_series(archive)
    assert bare["sites"] == {"total": 2, "shown_indices": [], "spectra": []}
    assert bare["frames"]["total"] == 3 and bare["frames"]["spectra"] == []

    sites_only = archive_series(archive, include=("sites",))
    assert len(sites_only["sites"]["spectra"]) == 2
    assert sites_only["frames"]["spectra"] == []

    with pytest.raises(ValueError, match="Unknown section"):
        archive_series(archive, include=("sites", "bogus"))


def test_frame_cap_is_even_and_recorded(tmp_path: Path):
    path = _archive(tmp_path / "many.h5", n_frames=45, n_sites=1)
    frames = archive_series(path, max_frames=20, include=("frames",))["frames"]
    shown = frames["shown_indices"]
    assert frames["total"] == 45
    assert len(shown) == 20
    assert shown[0] == 0 and shown[-1] == 44
    assert shown == sorted(set(shown))
    assert [s["frame_idx"] for s in frames["spectra"]] == shown


def test_nan_becomes_null(tmp_path: Path):
    path = tmp_path / "nan.h5"
    chi = np.sin(K)
    chi[-5:] = np.nan
    with EnsembleWriter(path, k_grid=K, fourier_params=FT) as w:
        w.set_overall_average(K, chi, chi_std=np.where(np.isnan(chi), np.nan, 0.1))
    series = archive_series(path)
    assert series["overall"]["chi"][-1] is None
    assert series["overall"]["chi_std"][-1] is None
    assert series["overall"]["chi"][0] is not None
    json.dumps(series, allow_nan=False)
    json.dumps(series_to_vega_lite(series, view="chi_k"), allow_nan=False)


def test_bad_arguments_and_non_ensemble_archives(archive: Path, tmp_path: Path):
    import h5py

    with pytest.raises(ValueError):
        archive_series(archive, max_frames=-1)
    with pytest.raises(FileNotFoundError):
        archive_series(tmp_path / "missing.h5")

    bare = tmp_path / "bare.h5"
    with h5py.File(bare, "w") as f:
        f.create_dataset("unrelated", data=np.zeros(3))
    reader = ArchiveReader(bare)
    assert not reader.is_ensemble and not reader.is_shard
    assert reader.fourier_params == {}
    with pytest.raises(ValueError, match="not an ensemble archive"):
        archive_series(bare)


def test_aggregates_only_archive_without_meta_group(archive: Path):
    import h5py

    with h5py.File(archive, "r+") as f:
        del f["meta"]
    reader = ArchiveReader(archive)
    assert reader.is_ensemble and not reader.is_shard
    assert reader.fourier_params == {}

    series = archive_series(archive)
    assert series["ft_params_source"] == "archive"
    assert any("records no Fourier-transform" in w for w in series["warnings"])
    assert series["overall"]["chi"][0] is not None
    recomputed = archive_series(archive, ft_params={"kweight": 1})
    assert recomputed["ft_params"]["kweight"] == 1


# --- Fourier-transform parameters -------------------------------------------


def test_kweight_scales_chi_and_moves_chir_consistently(archive: Path):
    base = archive_series(archive)["overall"]
    k = np.array(base["k"])
    for n in KWEIGHTS:
        series = archive_series(archive, ft_params={"kweight": n})
        o = series["overall"]
        np.testing.assert_allclose(o["chi"], base["chi"])
        np.testing.assert_allclose(o["chi_weighted"], k**n * np.array(o["chi"]))
        assert series["units"]["chir"] == f"A^-{n + 1}"
        assert abs(np.array(o["r"])[np.argmax(o["chir_mag"])] - R0) < 0.1


def test_chi_weighted_is_applied_to_std_sites_and_frames(archive: Path):
    series = archive_series(archive, ft_params={"kweight": 3}, include=ALL)
    o = series["overall"]
    k = np.array(o["k"])
    np.testing.assert_allclose(o["chi_std_weighted"], k**3 * np.array(o["chi_std"]))
    for spectrum in (series["sites"]["spectra"][0], series["frames"]["spectra"][-1]):
        np.testing.assert_allclose(
            spectrum["chi_weighted"],
            np.array(spectrum["k"]) ** 3 * np.array(spectrum["chi"]),
        )


@pytest.mark.parametrize(
    "request_params",
    [
        {
            "kmin": 4.0,
            "kmax": 11.0,
            "dk": 2.0,
            "window": "hanning",
            "kweight": 1,
            "rmax": 5.0,
        },
        {"window": "parzen", "rmax": 4.0},
    ],
)
def test_override_matches_direct_xftf(archive: Path, request_params: dict):
    series = archive_series(archive, ft_params=request_params, include=("sites",))
    reader = ArchiveReader(archive)
    resolved = {**STORED, **request_params}
    o = series["overall"]
    direct = xftf_arrays(reader.k, reader.chi, resolved)
    for name in ("r", "chir_mag", "chir_re", "chir_im"):
        np.testing.assert_allclose(o[name], direct[name], rtol=1e-12, atol=1e-14)

    site = series["sites"]["spectra"][1]
    direct_site = xftf_arrays(
        np.array(site["k"]), np.array(reader.get_site_average(1)["chi"]), resolved
    )
    np.testing.assert_allclose(site["chir_mag"], direct_site["chir_mag"])


def test_override_equal_to_stored_reproduces_stored_chir(archive: Path):
    stored = archive_series(archive, include=("frames",))
    same = archive_series(archive, ft_params=dict(STORED), include=("frames",))
    np.testing.assert_allclose(
        same["overall"]["chir_mag"], stored["overall"]["chir_mag"]
    )
    np.testing.assert_allclose(
        same["frames"]["spectra"][0]["chir_mag"],
        stored["frames"]["spectra"][0]["chir_mag"],
    )


def test_source_and_partial_override_resolution(archive: Path):
    stored = archive_series(archive)
    assert stored["ft_params_source"] == "archive"
    assert stored["ft_params"] == stored["ft_defaults"] == STORED

    partial = archive_series(archive, ft_params={"kweight": 1})
    assert partial["ft_params_source"] == "request"
    assert partial["ft_params"] == {**STORED, "kweight": 1}
    assert partial["ft_defaults"] == STORED


@pytest.mark.parametrize(
    ("request_params", "fragment"),
    [
        ({"foo": 1}, "unknown"),
        ({"window": "bogus"}, "window"),
        ({"kmin": float("nan")}, "kmin"),
        ({"kmin": 12.0}, "below kmax"),
        ({"rmax": RMAX_MAX + 1}, "rmax"),
        ({"kweight": 5}, "kweight"),
        ({"kweight": True}, "kweight"),
    ],
)
def test_invalid_ft_params_raise_with_message(
    archive: Path, request_params: dict, fragment: str
):
    with pytest.raises(InvalidFtParamsError, match=fragment) as info:
        archive_series(archive, ft_params=request_params)
    assert len(info.value.problems) == 1


def test_invalid_ft_params_report_every_problem(archive: Path):
    with pytest.raises(InvalidFtParamsError) as info:
        archive_series(archive, ft_params={"kmin": -1.0, "bogus": 1, "kweight": 9})
    assert len(info.value.problems) == 3


def test_ft_limits_describe_the_archive(archive: Path):
    limits = archive_series(archive)["ft_limits"]
    assert limits["kmin_data"] == pytest.approx(0.05)
    assert limits["kmax_data"] == pytest.approx(15.95)
    assert limits["k_grid_max"] == pytest.approx(15.95)


def test_window_beyond_data_warns_but_returns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    path = _archive(tmp_path / "ragged.h5", keep=lambda k: k <= 10.0)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        series = archive_series(path, ft_params={"kmax": 12.0}, include=ALL)
    assert not [r for r in caplog.records if r.name.startswith("md_exafs.spectra")]
    assert [(w["code"], w["section"]) for w in series["ft_warnings"]] == [
        ("window_beyond_data", "overall"),
        ("window_beyond_data", "sites"),
        ("window_beyond_data", "frames"),
    ]
    assert series["ft_warnings"][0]["data_limit"] == pytest.approx(10.0, abs=0.05)
    assert any("kmax=12" in w for w in series["warnings"])
    assert series["overall"]["chi_weighted"][-1] is None
    json.dumps(series, allow_nan=False)

    assert archive_series(path, ft_params={"kmax": 9.0})["ft_warnings"] == []


def test_kmin_below_data_warns(tmp_path: Path):
    path = _archive(tmp_path / "high.h5", keep=lambda k: k >= 2.0)
    series = archive_series(path, ft_params={"kmin": 1.0})
    assert [w["param"] for w in series["ft_warnings"]] == ["kmin"]


# --- Vega-Lite --------------------------------------------------------------


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _titles(spec: dict) -> set[str]:
    return {
        enc["title"]
        for node in _walk(spec)
        for enc in (node.get("encoding") or {}).values()
        if isinstance(enc, dict) and "title" in enc
    }


def _y_fields(spec: dict) -> set[str]:
    return {
        node["encoding"]["y"]["field"]
        for node in _walk(spec)
        if isinstance(node.get("encoding"), dict)
        and isinstance(node["encoding"].get("y"), dict)
        and "field" in node["encoding"]["y"]
    }


def _rule_datums(spec: dict) -> list[float]:
    return [
        node["encoding"]["x"]["datum"]
        for node in _walk(spec)
        if (node.get("mark") or {}).get("type") == "rule"
    ]


@pytest.mark.parametrize("view", ["chi_k", "chi_r", "sites", "frames"])
def test_vega_lite_views(archive: Path, view: str):
    spec = series_to_vega_lite(archive_series(archive, include=ALL), view=view)
    json.dumps(spec, allow_nan=False)
    assert "vega-lite" in spec["$schema"]
    datasets = [*spec.get("datasets", {}).values()]
    assert datasets and all(len(rows) > 0 for rows in datasets)
    if view == "frames":
        labels = {row["path"] for rows in datasets for row in rows}
        assert labels == {"frame 0", "frame 1", "frame 2"}


@pytest.mark.parametrize("kweight", [0, 2])
def test_chi_k_title_and_rules_follow_kweight(archive: Path, kweight: int):
    series = archive_series(archive, ft_params={"kweight": kweight})
    spec = series_to_vega_lite(series, view="chi_k")
    assert {0: "χ(k)", 2: "k²χ(k) (Å⁻²)"}[kweight] in _titles(spec)
    assert "chi_weighted" in _y_fields(spec)
    assert _rule_datums(spec) == [3.0, 12.0]


def test_chi_r_view_uses_a_neutral_axis_for_mag_re_and_im(archive: Path):
    series = archive_series(archive, ft_params={"kweight": 1, "rmax": 5.5})
    spec = series_to_vega_lite(series, view="chi_r")
    assert "χ(R) (Å⁻²)" in _titles(spec)
    assert not any("|χ(R)| (" in t for t in _titles(spec))
    assert _y_fields(spec) == {"chir"}
    labels = {row["path"] for rows in spec["datasets"].values() for row in rows}
    assert labels == {"|χ(R)|", "Re χ(R)", "Im χ(R)"}
    assert _rule_datums(spec) == [5.5]


def test_breakdown_views_use_weighted_chi_and_independent_zoom(archive: Path):
    series = archive_series(archive, ft_params={"kweight": 1}, include=ALL)
    for view in ("sites", "frames"):
        spec = series_to_vega_lite(series, view=view)
        assert "kχ(k) (Å⁻¹)" in _titles(spec)
        assert _y_fields(spec) == {"chi_weighted", "chir_mag"}
        assert sorted(p["name"] for p in spec["params"]) == ["zoom_k", "zoom_r"]


def test_vega_lite_errors(archive: Path):
    series = archive_series(archive)
    with pytest.raises(ValueError, match="Unknown view"):
        series_to_vega_lite(series, view="nope")
    with pytest.raises(ViewUnavailableError, match="include"):
        series_to_vega_lite(series, view="frames")


def test_vega_lite_usermeta_carries_series_context(archive: Path):
    series = archive_series(archive, ft_params={"kmax": 11.0}, include=("frames",))
    meta = series_to_vega_lite(series, view="chi_k")["usermeta"]
    assert meta["view"] == "chi_k"
    assert meta["warnings"] == series["warnings"]
    assert meta["ft_params"]["kmax"] == 11.0
    assert meta["ft_params_source"] == "request"
    assert meta["ft_defaults"] == STORED
    assert meta["ft_warnings"] == []
    assert meta["shown"]["frames"] == [0, 1, 2]
