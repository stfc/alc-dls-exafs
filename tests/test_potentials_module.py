"""Tests for md_exafs.potentials."""

from pathlib import Path

from md_exafs.potentials import CANONICAL_POTENTIAL_FILES, PotentialsManager, place_file


def test_manifest_contains_critical_files():
    assert "phase.pad" in CANONICAL_POTENTIAL_FILES
    assert "pot.pad" in CANONICAL_POTENTIAL_FILES
    assert "POTENTIALS" in CANONICAL_POTENTIAL_FILES


def test_place_file_symlink_and_copy(tmp_path: Path):
    src = tmp_path / "pot.pad"
    src.write_text("dummy_potential_data")

    dst_sym = tmp_path / "target_sym" / "pot.pad"
    dst_sym.parent.mkdir()
    assert place_file(src, dst_sym, mode="symlink")
    assert dst_sym.is_symlink()
    assert dst_sym.read_text() == "dummy_potential_data"

    dst_copy = tmp_path / "target_copy" / "pot.pad"
    dst_copy.parent.mkdir()
    assert place_file(src, dst_copy, mode="copy")
    assert not dst_copy.is_symlink()
    assert dst_copy.read_text() == "dummy_potential_data"


def test_distribute_potentials(tmp_path: Path):
    pot_dir = tmp_path / "pot_dir"
    pot_dir.mkdir()
    (pot_dir / "phase.pad").write_text("phase")
    (pot_dir / "pot.pad").write_text("pot")

    t1 = tmp_path / "t1"
    t2 = tmp_path / "t2"
    placed = PotentialsManager.distribute(pot_dir, [t1, t2], mode="copy")
    assert placed == 4
    assert (t1 / "phase.pad").read_text() == "phase"
    assert (t2 / "pot.pad").read_text() == "pot"
