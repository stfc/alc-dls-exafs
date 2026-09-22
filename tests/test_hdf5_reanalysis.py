"""Tests for re-analysing a finished run straight from its results HDF5.

``load_results_from_hdf5`` backs both ``md-exafs analyze results.h5`` and the
notebook's "existing results HDF5" entry point.  Only chi(k) is persisted per
site, so every Fourier transform is redone at load time -- that is what makes
it possible to retune the FT (window, kmin, kmax, dk, kweight) without
re-running any FEFF calculation.

Also guards the notebook analysis form against the key-name mismatch that made
the window selector silently inert: ``create_feff_config`` drops any setting
whose name is not a ``FeffConfig`` field.
"""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from larch_cli_wrapper.feff_utils import FeffConfig, WindowType
from larch_cli_wrapper.hdf5_store import ExafsHDF5Store
from larch_cli_wrapper.pipeline import (
    fourier_config_from_hdf5,
    load_results_from_hdf5,
    write_results_to_hdf5,
)

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "exafs_pipeline.py"


def _write_results(path: Path, n_frames: int = 3, n_sites: int = 2) -> FeffConfig:
    """Write a small but realistic ``results.h5``."""
    cfg = FeffConfig()
    k = np.linspace(0.0, 15.0, 200)
    with ExafsHDF5Store(path, config=cfg) as store:
        store.write_site_results_batch(
            [
                {
                    "frame_index": frame,
                    "site_index": site,
                    "k": k,
                    # Distinct but deterministic spectrum per (frame, site).
                    "chi": np.sin((4 + site) * k) * np.exp(-k / 10) * (1 + 0.1 * frame),
                    "absorber_element": "Fe",
                    "success": True,
                    "path_contributions": None,
                }
                for frame in range(n_frames)
                for site in range(n_sites)
            ]
        )
    return cfg


def test_round_trip_shapes_and_averages(tmp_path):
    h5 = tmp_path / "results.h5"
    cfg = _write_results(h5, n_frames=3, n_sites=2)

    loaded = load_results_from_hdf5(h5, cfg)

    assert len(loaded.groups) == 6
    assert set(loaded.frame_averages) == {0, 1, 2}
    assert set(loaded.site_averages) == {0, 1}
    assert loaded.overall_average is not None
    # Every group carries the metadata the plotting layer needs.
    for task_id, group in loaded.groups.items():
        assert task_id == f"frame_{group.frame_idx:04d}_site_{group.site_idx:04d}"
        assert group.absorber_element == "Fe"
        assert group.k.shape == group.chi.shape
        # chi(R) was computed, not just chi(k) restored.
        assert group.r.size > 0 and group.chir_mag.size > 0


def test_chi_k_is_preserved_exactly(tmp_path):
    """chi(k) is data; it must survive the round trip untouched."""
    h5 = tmp_path / "results.h5"
    cfg = _write_results(h5, n_frames=1, n_sites=1)
    k = np.linspace(0.0, 15.0, 200)
    expected = np.sin(4 * k) * np.exp(-k / 10)

    loaded = load_results_from_hdf5(h5, cfg)
    group = loaded.groups["frame_0000_site_0000"]

    np.testing.assert_allclose(group.k, k, rtol=0, atol=1e-6)
    # Stored as float32, so compare at that precision.
    np.testing.assert_allclose(group.chi, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("window", ["kaiser", "parzen", "welch"])
def test_fourier_transform_is_recomputed_from_config(tmp_path, window):
    """Changing FT settings changes chi(R) without touching stored chi(k)."""
    h5 = tmp_path / "results.h5"
    _write_results(h5, n_frames=1, n_sites=1)

    base = load_results_from_hdf5(h5, FeffConfig(window=WindowType.HANNING))
    other = load_results_from_hdf5(h5, FeffConfig(window=WindowType(window)))

    b = base.groups["frame_0000_site_0000"]
    o = other.groups["frame_0000_site_0000"]

    # chi(k) identical, chi(R) different: the FT really was redone.
    np.testing.assert_allclose(b.chi, o.chi, rtol=0, atol=0)
    assert not np.allclose(b.chir_mag, o.chir_mag), (
        f"window={window} produced an identical chi(R) to hanning"
    )


def test_kweight_and_krange_reach_the_transform(tmp_path):
    h5 = tmp_path / "results.h5"
    _write_results(h5, n_frames=1, n_sites=1)

    a = load_results_from_hdf5(h5, FeffConfig(kweight=1, kmin=2.0, kmax=12.0))
    b = load_results_from_hdf5(h5, FeffConfig(kweight=3, kmin=4.0, kmax=10.0))

    ga = a.groups["frame_0000_site_0000"]
    gb = b.groups["frame_0000_site_0000"]
    assert not np.allclose(ga.chir_mag, gb.chir_mag)
    # The averages must follow the same parameters, not stale defaults.
    assert not np.allclose(a.overall_average.chir_mag, b.overall_average.chir_mag)


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError, match="HDF5 file not found"):
        load_results_from_hdf5(tmp_path / "nope.h5", FeffConfig())


def test_file_without_site_results_raises_valueerror(tmp_path):
    h5 = tmp_path / "empty.h5"
    with ExafsHDF5Store(h5, config=FeffConfig()):
        pass
    with pytest.raises(ValueError, match="No per-site data"):
        load_results_from_hdf5(h5, FeffConfig())


def test_single_frame_still_produces_an_overall_average(tmp_path):
    h5 = tmp_path / "results.h5"
    cfg = _write_results(h5, n_frames=1, n_sites=1)
    loaded = load_results_from_hdf5(h5, cfg)
    assert loaded.overall_average is not None
    assert len(loaded.groups) == 1


# ---------------------------------------------------------------------------
# The archive records the settings it was made with
# ---------------------------------------------------------------------------


def test_fourier_config_is_recovered_from_the_archive(tmp_path):
    h5 = tmp_path / "results.h5"
    _write_results(h5, n_frames=1, n_sites=1)
    written = FeffConfig(
        kmin=3.5, kmax=9.0, dk=2.0, kweight=3, window=WindowType.PARZEN
    )
    with ExafsHDF5Store(h5, config=written, mode="a"):
        pass  # rewrites meta/feff_config

    recovered = fourier_config_from_hdf5(h5)
    for field in ("kmin", "kmax", "dk", "kweight", "window"):
        assert getattr(recovered, field) == getattr(written, field)


def test_fourier_config_leaves_non_fourier_fields_alone(tmp_path):
    """Only the FT fields are replayable; the rest describe the FEFF run."""
    h5 = tmp_path / "results.h5"
    _write_results(h5, n_frames=1, n_sites=1)
    base = FeffConfig(radius=9.9, n_workers=7, edge="L3")

    recovered = fourier_config_from_hdf5(h5, base=base)

    assert recovered.radius == 9.9
    assert recovered.n_workers == 7
    assert recovered.edge == "L3"


def test_default_reload_reproduces_the_original_transform(tmp_path):
    """Re-analysis must not silently swap in library defaults.

    ``FeffConfig()`` defaults to kmin=2.0, but chi(k) diverges below roughly
    2 A^-1 where the EXAFS approximation breaks down.  Reloading an archive
    written with a higher kmin under the default config lets that region back
    through the window and inflates chi(R), so the file's own settings win
    unless a config is passed explicitly.
    """
    h5 = tmp_path / "results.h5"
    written = FeffConfig(kmin=3.0, kmax=12.0)
    _write_results(h5, n_frames=1, n_sites=1)
    with ExafsHDF5Store(h5, config=written, mode="a"):
        pass

    from_file = load_results_from_hdf5(h5)
    explicit = load_results_from_hdf5(h5, written)
    defaults = load_results_from_hdf5(h5, FeffConfig())

    a = from_file.groups["frame_0000_site_0000"].chir_mag
    b = explicit.groups["frame_0000_site_0000"].chir_mag
    c = defaults.groups["frame_0000_site_0000"].chir_mag
    np.testing.assert_allclose(a, b, rtol=0, atol=0)
    # And the default really would have differed, so the test has teeth.
    assert not np.allclose(a, c)


def test_write_then_load_round_trip(tmp_path):
    """``write_results_to_hdf5`` and ``load_results_from_hdf5`` are a pair."""
    from larch import Group
    from larch.xafs import xftf

    from larch_cli_wrapper.exafs_data import create_averaged_group

    cfg = FeffConfig(kmin=3.0, kmax=9.5, dk=2.0, kweight=3, window=WindowType.KAISER)
    k = np.linspace(0.0, 15.0, 300)
    groups = {}
    for frame in range(2):
        for site in range(2):
            g = Group()
            g.k = k
            g.chi = np.sin((4 + site) * k) * np.exp(-k / 9) * (1 + 0.1 * frame)
            xftf(g, **cfg.fourier_params)
            g.frame_idx, g.site_idx = frame, site
            g.absorber_element = "Fe"
            g.task_id = f"frame_{frame:04d}_site_{site:04d}"
            groups[g.task_id] = g
    overall = create_averaged_group(list(groups.values()), cfg.fourier_params)

    out = tmp_path / "nested" / "results.h5"
    assert write_results_to_hdf5(out, cfg, groups, {}, {}, overall) == out
    assert out.exists()

    back = load_results_from_hdf5(out)

    assert set(back.groups) == set(groups)
    assert back.overall_average is not None
    for task_id, original in groups.items():
        # chi(k) is stored as float32, hence the tolerance.
        np.testing.assert_allclose(
            back.groups[task_id].chi, original.chi, rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(
            back.groups[task_id].chir_mag, original.chir_mag, rtol=0, atol=1e-4
        )
    np.testing.assert_allclose(
        back.overall_average.chir_mag, overall.chir_mag, rtol=0, atol=1e-4
    )


# ---------------------------------------------------------------------------
# Notebook form wiring
# ---------------------------------------------------------------------------


def _analysis_form_batch_keys() -> set[str]:
    """Keys the notebook's Stage C form submits, read from its source.

    The form is built as ``mo.md(...).batch(**kwargs).form(...)``; we locate the
    ``.batch`` call whose keywords include the FT parameters.
    """
    tree = ast.parse(NOTEBOOK.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "batch"
        ):
            keys = {kw.arg for kw in node.keywords if kw.arg}
            if "kweight" in keys and "kmin" in keys:
                return keys
    raise AssertionError("Could not find the analysis form's .batch() call")


def test_analysis_form_ft_keys_match_feffconfig_fields():
    """Regression guard for the inert window selector.

    ``create_feff_config`` keeps only settings whose key matches a FeffConfig
    field name and silently discards the rest, so a misnamed key (the form used
    to send ``window_type`` for the ``window`` field) is a no-op the UI gives no
    hint about.
    """
    config_fields = {f.name for f in fields(FeffConfig)}
    # Keys that intentionally do not map onto FeffConfig.
    non_config_keys = {"hdf5_path", "use_stored_ft", "save_archive"}

    keys = _analysis_form_batch_keys()
    unmapped = keys - config_fields - non_config_keys

    assert not unmapped, (
        f"Analysis form sends {sorted(unmapped)}, which create_feff_config() will "
        "silently drop. Rename to match the FeffConfig field, or add to "
        "non_config_keys if deliberately not a config value."
    )
    # The window selector specifically must be wired up.
    assert "window" in keys
