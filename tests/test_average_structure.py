"""Unit tests for the average_structure function in pipeline.py."""

import numpy as np
import pytest
from ase import Atoms

from larch_cli_wrapper.pipeline import average_structure


def test_basic_average_no_pbc():
    atoms1 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], pbc=False)
    atoms2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], pbc=False)
    avg = average_structure([atoms1, atoms2])
    expected_pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(avg.positions, expected_pos, atol=1e-10)
    assert avg.get_chemical_symbols() == ["H", "H"]


def test_simple_average_with_pbc():
    cell = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    atoms1 = Atoms("H", positions=[[0.5, 0.5, 0.5]], cell=cell, pbc=True)
    atoms2 = Atoms("H", positions=[[0.7, 0.7, 0.7]], cell=cell, pbc=True)
    avg = average_structure([atoms1, atoms2])
    expected_pos = np.array([[0.6, 0.6, 0.6]])
    np.testing.assert_allclose(avg.positions, expected_pos, atol=1e-10)


def test_pbc_unwrapping_across_boundary():
    cell = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    atoms1 = Atoms("H", positions=[[0.9, 0.5, 0.5]], cell=cell, pbc=True)
    atoms2 = Atoms("H", positions=[[0.1, 0.5, 0.5]], cell=cell, pbc=True)
    avg = average_structure([atoms1, atoms2])
    expected_pos = np.array([[0.0, 0.5, 0.5]])
    np.testing.assert_allclose(avg.positions, expected_pos, atol=1e-10)


def test_multiple_atoms_pbc():
    cell = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    atoms1 = Atoms(
        "H2", positions=[[0.8, 0.5, 0.5], [0.2, 0.3, 0.4]], cell=cell, pbc=True
    )
    atoms2 = Atoms(
        "H2", positions=[[0.2, 0.5, 0.5], [0.3, 0.3, 0.4]], cell=cell, pbc=True
    )
    avg = average_structure([atoms1, atoms2])
    expected_pos = np.array([[0.0, 0.5, 0.5], [0.25, 0.3, 0.4]])
    np.testing.assert_allclose(avg.positions, expected_pos, atol=1e-10)


def test_non_orthogonal_cell():
    cell = [[2.0, 0.5, 0.0], [0.0, 2.0, 0.5], [0.5, 0.0, 2.0]]
    atoms1 = Atoms("H", positions=[[0.1, 0.1, 0.1]], cell=cell, pbc=True)
    atoms2 = Atoms("H", positions=[[0.9, 0.9, 0.9]], cell=cell, pbc=True)
    avg = average_structure([atoms1, atoms2])
    assert np.all(avg.positions >= 0)
    assert np.all(avg.positions <= np.max(cell))


def test_mixed_pbc():
    cell = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 10.0]]
    pbc = [True, True, False]
    atoms1 = Atoms("H", positions=[[0.9, 0.5, 5.0]], cell=cell, pbc=pbc)
    atoms2 = Atoms("H", positions=[[0.1, 0.5, 5.0]], cell=cell, pbc=pbc)
    avg = average_structure([atoms1, atoms2])
    expected_pos = np.array([[0.0, 0.5, 5.0]])
    np.testing.assert_allclose(avg.positions, expected_pos, atol=1e-10)


def test_single_structure():
    cell = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    atoms = Atoms("H", positions=[[0.5, 0.5, 0.5]], cell=cell, pbc=True)
    avg = average_structure([atoms])
    np.testing.assert_allclose(avg.positions, atoms.positions, atol=1e-10)


def test_empty_list():
    with pytest.raises(ValueError, match="No structures provided"):
        average_structure([])


def test_inconsistent_atom_count():
    atoms1 = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="Structure 1 has 2 atoms, expected 1"):
        average_structure([atoms1, atoms2])


def test_inconsistent_atom_types():
    atoms1 = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms2 = Atoms("He", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="Structure 1 has different atom types"):
        average_structure([atoms1, atoms2])


def test_different_cells_raises_error():
    cell1 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    cell2 = [[1.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1]]
    atoms1 = Atoms("H", positions=[[0.5, 0.5, 0.5]], cell=cell1, pbc=True)
    atoms2 = Atoms("H", positions=[[0.5, 0.5, 0.5]], cell=cell2, pbc=True)

    with pytest.raises(ValueError, match="Structure 1 has a different cell"):
        average_structure([atoms1, atoms2])


def test_trajectory_with_large_displacements():
    """Test trajectory where atom oscillates across boundary.

    The unwrapping will interpret 0.1 <-> 0.9 as small backward/forward
    jumps (-0.2/+0.2) rather than large forward/backward jumps (+0.8/-0.8),
    resulting in an average near 0.0.
    """
    cell = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    positions = [
        [[0.1, 0.5, 0.5]],
        [[0.9, 0.5, 0.5]],
        [[0.1, 0.5, 0.5]],
        [[0.9, 0.5, 0.5]],
    ]
    structures = [Atoms("H", positions=pos, cell=cell, pbc=True) for pos in positions]
    avg = average_structure(structures)

    # np.unwrap interprets this as [0.1, -0.1, 0.1, -0.1] -> mean ≈ 0.0
    expected_pos = np.array([[0.0, 0.5, 0.5]])
    np.testing.assert_allclose(avg.positions, expected_pos, atol=1e-10)
