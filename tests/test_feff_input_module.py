"""Tests for md_exafs.feff_input."""

import pytest
from ase.build import bulk

from md_exafs.feff_input import (
    FeffConfig,
    build_feff_inp,
    make_paths_feff_config,
    make_potentials_feff_config,
)


def test_feff_config_presets():
    cfg = FeffConfig.from_preset("quick")
    assert cfg.spectrum_type == "EXAFS"
    assert cfg.edge == "K"
    assert cfg.radius == 4.0


def test_feff_config_control_helpers():
    base = FeffConfig(radius=5.0)
    pot_cfg = make_potentials_feff_config(base)
    assert pot_cfg.control == "1 1 1 0 0 0"

    path_cfg = make_paths_feff_config(base)
    assert path_cfg.control == "0 0 0 1 1 1"


def test_build_feff_inp_cu_bulk():
    atoms = bulk("Cu", "fcc", a=3.61)
    inp = build_feff_inp(atoms, FeffConfig(radius=4.5), absorber_idx=0)
    assert "EDGE      K" in inp or "EDGE  K" in inp or "EDGE" in inp
    assert "RPATH" in inp
    assert "ATOMS" in inp
    assert "POTENTIALS" in inp
    # COREHOLE must be pruned for FEFF8L
    assert (
        "COREHOLE" not in inp
        or "* deleted: COREHOLE" in inp
        or inp.count("COREHOLE") == 0
    )


def test_build_feff_inp_exclude_hydrogen():
    from ase import Atoms

    atoms = Atoms("HCu", positions=[[0, 0, 0], [1.5, 0, 0]])
    cfg = FeffConfig(exclude_hydrogen=True, radius=3.0)
    # Absorber index 1 is Cu
    inp = build_feff_inp(atoms, cfg, absorber_idx=1)
    assert "Cu" in inp
    # Absorber index 0 is H -> should raise ValueError
    with pytest.raises(ValueError, match="hydrogen atom"):
        build_feff_inp(atoms, cfg, absorber_idx=0)
