"""Tests for feffNNNN.dat path reading via larch's FeffDatFile.

larch.xafs.feffdat.FeffDatFile is used to outsource the fragile ASCII parsing.
The fixture file tests/feff0001.dat is a real output from a FEFF8L calculation
on a K-edge site in a KFe2(CN)6 (prussian blue) structure.
"""

import concurrent.futures
import shutil
from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent
WORKSPACE_ROOT = FIXTURES_DIR.parent


@pytest.fixture
def feff_dir(tmp_path):
    """Temporary directory containing a single feff0001.dat fixture file."""
    shutil.copy(FIXTURES_DIR / "feff0001.dat", tmp_path / "feff0001.dat")
    return tmp_path


def test_read_path_contributions_returns_list(feff_dir):
    """_read_path_contributions_from_dir returns a non-empty list."""
    from larch_cli_wrapper.hdf5_store import _read_path_contributions_from_dir

    results = _read_path_contributions_from_dir(feff_dir)
    assert isinstance(results, list)
    assert len(results) == 1


def test_read_path_contributions_keys(feff_dir):
    """Each result dict has all required keys with correct types."""
    from larch_cli_wrapper.hdf5_store import _read_path_contributions_from_dir

    results = _read_path_contributions_from_dir(feff_dir)
    r = results[0]

    assert set(r.keys()) >= {
        "path_index",
        "k",
        "chi",
        "r_eff",
        "nlegs",
        "degeneracy",
        "scatterer",
    }
    assert isinstance(r["path_index"], int)
    assert isinstance(r["k"], np.ndarray)
    assert isinstance(r["chi"], np.ndarray)
    assert isinstance(r["r_eff"], float)
    assert isinstance(r["nlegs"], int)
    assert isinstance(r["degeneracy"], float)
    assert isinstance(r["scatterer"], str)


def test_read_path_contributions_values(feff_dir):
    """Parsed values match the known content of the fixture file."""
    from larch_cli_wrapper.hdf5_store import _read_path_contributions_from_dir

    results = _read_path_contributions_from_dir(feff_dir)
    r = results[0]

    assert r["path_index"] == 1
    assert r["nlegs"] == 2
    # r_eff is read directly from the feff0001.dat header line
    assert abs(r["r_eff"] - 3.2222) < 0.01
    assert r["scatterer"] == "Fe"
    # chi should be a non-trivial signal
    assert r["k"].shape == r["chi"].shape
    assert len(r["k"]) > 10
    assert np.max(np.abs(r["chi"])) > 0.0


def test_parse_unaffected_by_prior_larch_use(feff_dir):
    """Parsing succeeds even after larch autobk/xftf have been called.

    This is the regression test: larch.xafs.feffdat.FeffDatFile must remain
    usable after larch has been used in the same process.
    """
    # Simulate the pipeline: run larch processing before parsing paths
    import numpy as np
    from larch import Group, Interpreter
    from larch.xafs import autobk, xftf

    _larch = Interpreter()
    g = Group(energy=np.linspace(7100, 7900, 400), mu=np.random.rand(400))
    g.energy[0] = 7112.0
    try:
        autobk(_larch=_larch, energy=g.energy, mu=g.mu, group=g)
        xftf(
            _larch=_larch,
            k=getattr(g, "k", np.linspace(0, 15, 200)),
            chi=getattr(g, "chi", np.zeros(200)),
            group=g,
        )
    except Exception:  # noqa: BLE001, S110
        pass  # We only care that the state is modified, not that these succeed

    from larch_cli_wrapper.hdf5_store import _read_path_contributions_from_dir

    results = _read_path_contributions_from_dir(feff_dir)
    assert len(results) == 1, f"Expected 1 path after larch use, got {len(results)}"
    assert np.max(np.abs(results[0]["chi"])) > 0.0


def test_read_path_contributions_concurrent(feff_dir):
    """Concurrent calls from multiple threads all produce correct results."""
    from larch_cli_wrapper.hdf5_store import _read_path_contributions_from_dir

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(_read_path_contributions_from_dir, feff_dir) for _ in range(8)
        ]
        results_list = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert all(len(r) == 1 for r in results_list), (
        f"Expected 1 path per call, got {[len(r) for r in results_list]}"
    )


def test_recompute_path_chi_on_grid(feff_dir):
    """recompute_path_chi_on_grid evaluates χ(k) on an arbitrary fine grid."""
    import numpy as np

    from larch_cli_wrapper.hdf5_store import (
        _read_path_contributions_from_dir,
        recompute_path_chi_on_grid,
    )

    results_coarse = _read_path_contributions_from_dir(feff_dir)
    r_coarse = results_coarse[0]

    k_fine = np.arange(0.05, 20.05, 0.05)
    r_fine = recompute_path_chi_on_grid(r_coarse, k_fine)

    assert r_fine["k"].shape == k_fine.shape
    assert np.allclose(r_fine["k"], k_fine)
    assert np.max(np.abs(r_fine["chi"])) > 0.0

    # Fine-grid chi evaluated back at coarse points should match coarse chi
    # (excluding the k=0 singularity region).
    mask = r_coarse["k"] > 0.1
    chi_fine_at_coarse = np.interp(r_coarse["k"][mask], r_fine["k"], r_fine["chi"])
    assert np.allclose(chi_fine_at_coarse, r_coarse["chi"][mask], atol=1e-3)


def test_read_path_contributions_with_max_paths(feff_dir):
    """_read_path_contributions_from_dir respects max_paths."""
    from larch_cli_wrapper.hdf5_store import _read_path_contributions_from_dir

    # Copy another dummy file to simulate multiple path files
    shutil.copy(feff_dir / "feff0001.dat", feff_dir / "feff0002.dat")
    shutil.copy(feff_dir / "feff0001.dat", feff_dir / "feff0003.dat")

    # Limit to 2 paths
    results = _read_path_contributions_from_dir(feff_dir, max_paths=2)
    assert len(results) == 2


def test_2body_path_has_no_angle(feff_dir):
    """The 2-leg fixture path has no meaningful 3-body angle."""
    from larch_cli_wrapper.hdf5_store import _read_path_contributions_from_dir

    results = _read_path_contributions_from_dir(feff_dir)
    assert results[0]["angle"] is None


def _make_3leg_feff_dat(path: Path) -> None:
    """Write a minimal, synthetic 3-leg (triangular) feffNNNN.dat file.

    Absorber (K) at the origin, N at (2, 0, 0), C at (0, 2, 0). This gives a
    45-degree angle at N (the first non-absorber leg) between N->absorber
    and N->C, and reff = (2 + sqrt(8) + 2) / 2 = 3.4142.
    """
    lines = [
        " comment: synthetic 3-leg test fixture             Feff8L (EXAFS)       0.1",
        " Source:",
        " Structure Summary:  Test",
        " space group: (P1), space number:  (1)",
        " abc:  10.0 10.0 10.0",
        " angles: 90.0 90.0 90.0",
        " sites: 3",
        " POT  SCF  30  4.0000   0, core-hole, AFOLP (folp(0)= 1.150)",
        " Abs   Z=19 Rmt= 2.232 Rnm= 2.257 K  shell",
        " Pot 1 Z= 7 Rmt= 0.701 Rnm= 1.029",
        " Pot 2 Z= 6 Rmt= 0.749 Rnm= 1.019",
        " Gam_ch=6.796E-01 H-L exch",
        " Mu=-1.236E+01 kf=2.747E+00 Vint=-2.078E+01 Rs_int= 1.320",
        " PATH  Rmax= 4.000,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 0.00%",
        " Path    1      icalc       2",
        " -----------------------------------------------------------------------",
        "   3   1.000   3.4142    1.9836  -12.36283 nleg, deg, reff, rnrmav(bohr), edge",  # noqa: E501
        "        x         y         z   pot at#",
        "     0.0000    0.0000    0.0000  0  19 K        absorbing atom",
        "     2.0000    0.0000    0.0000  1   7 N",
        "     0.0000    2.0000    0.0000  2   6 C",
        "    k   real[2*phc]   mag[feff]  phase[feff] red factor   lambda     real[p]@#",  # noqa: E501
    ]
    for i in range(15):
        k = i * 0.5
        lines.append(f"  {k:.3f}  1.0E+01  1.0E-01 -1.0E+01  1.0E+00  3.0E+01  1.5E+00")
    path.write_text("\n".join(lines) + "\n")


def test_3body_angle_computed_from_geometry(tmp_path):
    """A 3-leg path's angle is computed from the absorber/leg geometry."""
    from larch_cli_wrapper.hdf5_store import _read_path_contributions_from_dir

    _make_3leg_feff_dat(tmp_path / "feff0001.dat")
    results = _read_path_contributions_from_dir(tmp_path)

    assert len(results) == 1
    r = results[0]
    assert r["nlegs"] == 3
    assert r["angle"] is not None
    assert abs(r["angle"] - 45.0) < 0.5


def test_3body_scatterer_label_is_canonicalized(tmp_path):
    """Scatterer label is sorted regardless of the atom-listing order in the
    path file, so 'N-C' and 'C-N' from different frames compare equal."""
    from larch_cli_wrapper.hdf5_store import _read_path_contributions_from_dir

    _make_3leg_feff_dat(tmp_path / "feff0001.dat")
    results = _read_path_contributions_from_dir(tmp_path)

    assert results[0]["scatterer"] == "C-N"
