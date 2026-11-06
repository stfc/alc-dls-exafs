"""Caller-supplied Fourier-transform parameters for archive_series.

Synthetic ensemble archives written with EnsembleWriter; chi(k) is a 1/k
decaying sine with a known shell at R0, so the transform has a checkable peak.
"""

import json
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
    series_to_vega_lite,
)
from md_exafs.spectra import FT_WINDOWS, average_chi_arrays, xftf_arrays

K = np.arange(0.05, 16.0, 0.05)
R0 = 2.5
FT = {"kmin": 3.0, "kmax": 12.0, "kweight": 2, "dk": 1.0, "rmax": 6.0}
STORED = {
    "kmin": 3.0,
    "kmax": 12.0,
    "dk": 1.0,
    "window": "kaiser",
    "kweight": 2,
    "rmax": 6.0,
}


def _chi(frame: int, site: int) -> np.ndarray:
    scale = 1.0 + 0.05 * frame + 0.01 * site
    return scale * np.sin(2 * K * R0) / (K * R0**2)


def _archive(
    path: Path,
    *,
    keep: Callable[[np.ndarray], np.ndarray] = lambda k: k > 0,
    n_frames: int = 3,
    n_sites: int = 2,
    overall_scale: float = 1.0,
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
    return _archive(tmp_path / "ft.h5")


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _titles(spec: dict) -> set[str]:
    out = set()
    for node in _walk(spec):
        for enc in (node.get("encoding") or {}).values():
            if isinstance(enc, dict) and "title" in enc:
                out.add(enc["title"])
    return out


def _rule_datums(spec: dict) -> list[float]:
    out = []
    for node in _walk(spec):
        mark = node.get("mark")
        kind = mark.get("type") if isinstance(mark, dict) else mark
        if kind == "rule":
            out.append(node["encoding"]["x"]["datum"])
    return out


def test_kweight_scales_chi_k_and_moves_chir_consistently(archive: Path):
    k = np.array(archive_series(archive)["overall"]["k"])
    peaks, amps = {}, {}
    for n in (0, 1, 2, 3):
        series = archive_series(archive, ft_params={"kweight": n})
        o = series["overall"]
        np.testing.assert_allclose(o["chi"], archive_series(archive)["overall"]["chi"])
        np.testing.assert_allclose(o["chi_weighted"], k**n * np.array(o["chi"]))
        assert series["units"]["chi_weighted"] == (f"A^-{n}" if n else "dimensionless")
        assert series["units"]["chir"] == f"A^-{n + 1}"
        r, mag = np.array(o["r"]), np.array(o["chir_mag"])
        peaks[n] = r[np.argmax(mag)]
        amps[n] = mag.max()
    for n in (0, 1, 2, 3):
        assert abs(peaks[n] - R0) < 0.1, peaks
    assert amps[1] < amps[2] < amps[3]


def test_chi_weighted_is_applied_to_sites_frames_and_std(archive: Path):
    series = archive_series(archive, ft_params={"kweight": 3})
    k = np.array(series["overall"]["k"])
    np.testing.assert_allclose(
        series["overall"]["chi_std_weighted"],
        k**3 * np.array(series["overall"]["chi_std"]),
    )
    site = series["sites"]["spectra"][0]
    np.testing.assert_allclose(
        site["chi_weighted"], np.array(site["k"]) ** 3 * np.array(site["chi"])
    )
    frame = series["frames"]["spectra"][-1]
    np.testing.assert_allclose(
        frame["chi_weighted"], np.array(frame["k"]) ** 3 * np.array(frame["chi"])
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
        {
            "kmin": 2.0,
            "kmax": 9.0,
            "dk": 0.5,
            "window": "gaussian",
            "kweight": 3,
            "rmax": 7.0,
        },
        {"window": "parzen", "rmax": 4.0},
    ],
)
def test_override_matches_direct_xftf(archive: Path, request_params: dict):
    series = archive_series(archive, ft_params=request_params)
    reader = ArchiveReader(archive)
    resolved = {**STORED, **request_params}
    o = series["overall"]
    direct = xftf_arrays(reader.k, reader.chi, resolved)
    np.testing.assert_allclose(
        o["chir_mag"], direct["chir_mag"], rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(o["chir_re"], direct["chir_re"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(o["chir_im"], direct["chir_im"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(o["r"], direct["r"])

    site = series["sites"]["spectra"][1]
    direct_site = xftf_arrays(
        np.array(site["k"]), np.array(reader.get_site_average(1)["chi"]), resolved
    )
    np.testing.assert_allclose(site["chir_mag"], direct_site["chir_mag"])


def test_override_changes_chir_from_stored(archive: Path):
    base = archive_series(archive)
    other = archive_series(archive, ft_params={"kmax": 9.0, "window": "parzen"})
    assert not np.allclose(base["overall"]["chir_mag"], other["overall"]["chir_mag"])


def test_override_equal_to_stored_reproduces_stored_chir(archive: Path):
    stored = archive_series(archive)
    same = archive_series(archive, ft_params=dict(STORED))
    np.testing.assert_allclose(
        same["overall"]["chir_mag"], stored["overall"]["chir_mag"]
    )
    np.testing.assert_allclose(same["overall"]["chir_re"], stored["overall"]["chir_re"])
    np.testing.assert_allclose(
        same["frames"]["spectra"][0]["chir_mag"],
        stored["frames"]["spectra"][0]["chir_mag"],
    )


def test_source_and_partial_override_resolution(archive: Path):
    stored = archive_series(archive)
    assert stored["ft_params_source"] == "archive"
    assert stored["ft_params"] == STORED
    assert stored["ft_defaults"] == STORED

    partial = archive_series(archive, ft_params={"kweight": 1})
    assert partial["ft_params_source"] == "request"
    assert partial["ft_params"] == {**STORED, "kweight": 1}
    assert partial["ft_defaults"] == STORED
    assert partial["units"]["chir"] == "A^-2"


def test_override_does_not_touch_k_space_check_or_add_warnings(tmp_path: Path):
    stale = _archive(tmp_path / "stale.h5", overall_scale=1.1)
    base = archive_series(stale)
    other = archive_series(stale, ft_params={"kweight": 1, "rmax": 5.0})
    assert other["checks"] == base["checks"]
    assert other["checks"]["stored_vs_recomputed"]["status"] == "mismatch"
    assert other["warnings"] == base["warnings"]
    assert other["ft_warnings"] == []


@pytest.mark.parametrize(
    ("request_params", "fragment"),
    [
        ({"foo": 1}, "unknown"),
        ({"window": "bogus"}, "window"),
        ({"window": 3}, "window"),
        ({"kmin": -1.0}, "kmin"),
        ({"kmin": float("nan")}, "kmin"),
        ({"kmin": 12.0}, "below kmax"),
        ({"kmax": 2.0}, "below kmax"),
        ({"kmax": float("inf")}, "kmax"),
        ({"dk": -0.5}, "dk"),
        ({"rmax": 0.0}, "rmax"),
        ({"rmax": RMAX_MAX + 1}, "rmax"),
        ({"kweight": 5}, "kweight"),
        ({"kweight": -1}, "kweight"),
        ({"kweight": 2.5}, "kweight"),
        ({"kweight": True}, "kweight"),
    ],
)
def test_invalid_ft_params_raise_with_message(
    archive: Path, request_params: dict, fragment: str
):
    with pytest.raises(InvalidFtParamsError) as info:
        archive_series(archive, ft_params=request_params)
    assert isinstance(info.value, ValueError)
    assert len(info.value.problems) == 1
    assert fragment in str(info.value)


def test_invalid_ft_params_report_every_problem(archive: Path):
    with pytest.raises(InvalidFtParamsError) as info:
        archive_series(
            archive, ft_params={"kmin": -1.0, "rmax": 0.0, "bogus": 1, "kweight": 9}
        )
    assert len(info.value.problems) == 4
    assert "bogus" in str(info.value)


def test_supported_windows_are_one_constant():
    assert "kaiser" in FT_WINDOWS and "hanning" in FT_WINDOWS
    assert KWEIGHTS == (0, 1, 2, 3)


def test_kmax_beyond_data_warns_but_returns(tmp_path: Path):
    path = _archive(tmp_path / "ragged.h5", keep=lambda k: k <= 10.0)
    series = archive_series(path, ft_params={"kmax": 12.0})
    assert series["ft_params"]["kmax"] == 12.0
    assert len(series["ft_warnings"]) == 1
    warning = series["ft_warnings"][0]
    assert warning["code"] == "window_beyond_data"
    assert warning["param"] == "kmax"
    assert warning["data_limit"] == pytest.approx(10.0, abs=0.05)
    assert any("kmax=12" in w for w in series["warnings"])
    assert series["overall"]["chir_mag"][0] is not None
    json.dumps(series, allow_nan=False)

    within = archive_series(path, ft_params={"kmax": 9.0})
    assert within["ft_warnings"] == []
    assert within["ft_limits"]["kmax_data"] == pytest.approx(10.0, abs=0.05)


def test_kmin_below_data_warns(tmp_path: Path):
    path = _archive(tmp_path / "high.h5", keep=lambda k: k >= 2.0)
    series = archive_series(path, ft_params={"kmin": 1.0})
    assert [w["param"] for w in series["ft_warnings"]] == ["kmin"]
    assert series["ft_limits"]["kmin_data"] == pytest.approx(2.0, abs=0.05)


def test_ragged_series_is_json_safe_under_override(tmp_path: Path):
    path = _archive(tmp_path / "ragged.h5", keep=lambda k: k <= 10.0)
    series = archive_series(path, ft_params={"kweight": 3, "kmax": 12.0})
    json.dumps(series, allow_nan=False)
    assert series["overall"]["chi_weighted"][-1] is None


def test_ft_limits_describe_the_archive(archive: Path):
    series = archive_series(archive)
    limits = series["ft_limits"]
    assert limits["kmin_data"] == pytest.approx(0.05)
    assert limits["kmax_data"] == pytest.approx(15.95)
    assert limits["k_grid_max"] == pytest.approx(15.95)
    assert limits["kweights"] == [0, 1, 2, 3]
    assert limits["windows"] == list(FT_WINDOWS)
    assert limits["rmax_max"] == RMAX_MAX
    assert series["ft_warnings"] == []


def test_include_limits_transformed_sections(archive: Path):
    only_overall = archive_series(archive, include=("overall",))
    assert only_overall["sites"]["total"] == 2
    assert only_overall["sites"]["spectra"] == []
    assert only_overall["frames"]["shown_indices"] == []
    assert only_overall["overall"]["chir_mag"] is not None

    only_sites = archive_series(archive, include=("sites",), ft_params={"kweight": 1})
    assert len(only_sites["sites"]["spectra"]) == 2
    assert only_sites["frames"]["spectra"] == []

    with pytest.raises(ValueError, match="Unknown section"):
        archive_series(archive, include=("sites", "bogus"))


@pytest.mark.parametrize("kweight", KWEIGHTS)
def test_chi_k_title_and_rules_follow_kweight(archive: Path, kweight: int):
    series = archive_series(archive, ft_params={"kweight": kweight})
    spec = series_to_vega_lite(series, view="chi_k")
    expected = {
        0: "χ(k)",
        1: "kχ(k) (Å⁻¹)",
        2: "k²χ(k) (Å⁻²)",
        3: "k³χ(k) (Å⁻³)",
    }[kweight]
    assert expected in _titles(spec)
    assert _rule_datums(spec) == [3.0, 12.0]
    json.dumps(spec, allow_nan=False)


@pytest.mark.parametrize("kweight", KWEIGHTS)
def test_chi_r_title_and_rmax_rule(archive: Path, kweight: int):
    series = archive_series(archive, ft_params={"kweight": kweight, "rmax": 5.5})
    spec = series_to_vega_lite(series, view="chi_r")
    unit = {1: "Å⁻¹", 2: "Å⁻²", 3: "Å⁻³", 4: "Å⁻⁴"}[kweight + 1]
    assert f"|χ(R)| ({unit})" in _titles(spec)
    assert _rule_datums(spec) == [5.5]
    assert spec["usermeta"]["ft_params"]["rmax"] == 5.5


def test_sites_and_frames_use_the_same_weighting(archive: Path):
    series = archive_series(archive, ft_params={"kweight": 1})
    for view in ("sites", "frames"):
        spec = series_to_vega_lite(series, view=view)
        assert "kχ(k) (Å⁻¹)" in _titles(spec)
        fields = {
            (node["encoding"]["y"]["field"])
            for node in _walk(spec)
            if isinstance(node.get("encoding"), dict)
            and isinstance(node["encoding"].get("y"), dict)
            and "field" in node["encoding"]["y"]
        }
        assert "chi_weighted" in fields


def test_vega_lite_usermeta_carries_ft_context(archive: Path):
    series = archive_series(archive, ft_params={"kmax": 11.0})
    meta = series_to_vega_lite(series, view="chi_k")["usermeta"]
    assert meta["ft_params_source"] == "request"
    assert meta["ft_params"]["kmax"] == 11.0
    assert meta["ft_defaults"] == STORED
    assert meta["ft_limits"]["kweights"] == [0, 1, 2, 3]
    assert meta["ft_warnings"] == []


def test_default_series_keeps_existing_fields(archive: Path):
    series = archive_series(archive)
    assert set(series) >= {
        "schema_version",
        "units",
        "ft_params",
        "source",
        "downsampling",
        "overall",
        "sites",
        "frames",
        "checks",
        "warnings",
        "ft_params_source",
        "ft_defaults",
        "ft_limits",
        "ft_warnings",
    }
    assert (
        series["overall"]["chi"]
        == archive_series(archive, ft_params=dict(STORED))["overall"]["chi"]
    )
