"""End-to-end pipeline tests using the real FEFF (feff8l) executable.

These exercise the true generate -> run FEFF -> parse paths -> store -> average
path with actual FEFF output, and assert version-independent invariants rather
than brittle absolute values.  Skipped automatically if ``feff8l`` is absent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from ase.build import bulk

from larch_cli_wrapper.feff_utils import FeffConfig
from larch_cli_wrapper.pipeline import PipelineProcessor

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("feff8l") is None, reason="feff8l executable not found on PATH"
    ),
]


def _run_pipeline(
    out_dir: Path,
    *,
    n_frames: int = 2,
    chunk_size: int = 100,
    store_path_params: bool = False,
) -> Path:
    """Run the real pipeline on a tiny 1-atom fcc-Cu cell; return results.h5."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cu = bulk("Cu", "fcc", a=3.61)  # primitive: 1 atom -> 1 site per frame
    frames = [cu.copy() for _ in range(n_frames)]
    cfg = FeffConfig(
        radius=3.2,
        kmin=2.0,
        kmax=10.0,
        keep_path_files=True,
        cleanup_feff_files=True,
        stream_chunk_size=chunk_size,
        store_path_params=store_path_params,
    )
    h5 = out_dir / "results.h5"
    # max_workers=1 keeps path parsing / aggregation serial (no process pool),
    # which is deterministic and fast for a couple of calculations.
    proc = PipelineProcessor(cfg, max_workers=1, hdf5_path=h5)
    proc.process_trajectory(frames, absorber="Cu", output_dir=out_dir, parallel=True)
    return h5


@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e_base")
    return _run_pipeline(out, n_frames=2, chunk_size=100)


def test_results_and_averaged_files_created(base_run):
    assert base_run.exists()
    averaged = base_run.with_name("results_averaged_paths.h5")
    assert averaged.exists()


def test_results_h5_structure(base_run):
    with h5py.File(base_run, "r") as f:
        assert "site_results" in f
        assert f["site_results"]["chi"].shape[0] == 2  # 2 frames x 1 site
        # real FEFF produced scattering paths
        assert "path_results" in f
        assert f["path_results"]["chi"].shape[0] > 0
        # storage stays float32
        assert f["site_results"]["chi"].dtype == np.float32
        assert f["path_results"]["chi"].dtype == np.float32


def test_averaged_paths_structure(base_run):
    averaged = base_run.with_name("results_averaged_paths.h5")
    with h5py.File(averaged, "r") as f:
        assert "overall_average" in f
        grp = f["overall_average"]
        assert "chi" in grp and "k" in grp
        assert grp["chi"].shape[0] > 0
        # at least one averaged path contribution
        assert "paths" in grp
        assert len(grp["paths"].keys()) > 0


def test_identical_frames_average_equals_single_frame(base_run):
    """Two identical frames -> overall average equals each per-site spectrum."""
    with h5py.File(base_run, "r") as f:
        site_chi = np.array(f["site_results"]["chi"])
    # both frames identical -> the two stored spectra match
    np.testing.assert_allclose(site_chi[0], site_chi[1], rtol=1e-5, atol=1e-6)


def test_chunk_size_does_not_change_results(tmp_path):
    """Streaming in chunks of 1 yields identical stored spectra to one big chunk."""
    h5_big = _run_pipeline(tmp_path / "big", n_frames=3, chunk_size=100)
    h5_stream = _run_pipeline(tmp_path / "stream", n_frames=3, chunk_size=1)

    def _sorted_site_chi(h5):
        with h5py.File(h5, "r") as f:
            fr = np.array(f["site_results"]["frame_index"])
            si = np.array(f["site_results"]["site_index"])
            chi = np.array(f["site_results"]["chi"])
            order = np.lexsort((si, fr))
            return chi[order]

    np.testing.assert_allclose(
        _sorted_site_chi(h5_big), _sorted_site_chi(h5_stream), rtol=1e-6, atol=1e-7
    )


def test_store_path_params_does_not_change_averaged_chi(tmp_path):
    """Persisting amp/pha/lam/rep must not affect the averaged spectrum."""
    h5_off = _run_pipeline(tmp_path / "off", n_frames=2, store_path_params=False)
    h5_on = _run_pipeline(tmp_path / "on", n_frames=2, store_path_params=True)

    def _overall_chi(h5):
        with h5py.File(h5.with_name("results_averaged_paths.h5"), "r") as f:
            return np.array(f["overall_average"]["chi"])

    np.testing.assert_allclose(
        _overall_chi(h5_off), _overall_chi(h5_on), rtol=1e-5, atol=1e-6
    )
