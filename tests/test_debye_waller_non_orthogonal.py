import numpy as np
import pytest
from ase import Atoms

from larch_cli_wrapper.debye_waller_core import (
    calculate_grouped_msrd,
    process_trajectory,
)


def test_msrd_non_orthogonal_cell():
    # Define a highly non-orthogonal triclinic unit cell
    cell = [[3.0, 1.0, 0.5], [0.5, 3.0, 1.0], [1.0, 0.5, 3.0]]

    # 3-atom system: Mn (central absorber) and two O neighbors
    symbols = ["Mn", "O", "O"]

    # Create 10 perturbed frames (vibrating trajectory)
    np.random.seed(42)
    structures = []

    base_positions = np.array(
        [
            [0.0, 0.0, 0.0],  # Mn (index 0)
            [1.5, 1.0, 0.8],  # O1 (index 1)
            [-1.0, 1.5, 1.2],  # O2 (index 2)
        ]
    )

    for _ in range(10):
        # Perturb slightly to simulate thermal motion
        pos = base_positions + np.random.normal(0, 0.05, base_positions.shape)
        atoms = Atoms(symbols, positions=pos, cell=cell, pbc=True)
        structures.append(atoms)

    # Calculate the true (reference) 2-body and 3-body MIC lengths and angles
    # directly from Atoms frames
    ref_d_01 = []
    ref_d_02 = []
    ref_angles = []

    for atoms in structures:
        # 2-body reference distances under MIC
        d01 = atoms.get_distance(0, 1, mic=True)
        d02 = atoms.get_distance(0, 2, mic=True)
        ref_d_01.append(d01)
        ref_d_02.append(d02)

        # 3-body angle at Neighbor 1 (O1): O1->Mn against O1->O2
        # In calculate_grouped_msrd, the angle is at n1: n1->absorber against n1->n2.
        # Let's compute this reference angle exactly using Atoms MIC vectors
        v01 = atoms.get_distances(0, [1], mic=True, vector=True)[0]
        v12 = atoms.get_distances(1, [2], mic=True, vector=True)[0]

        # v1: n1 -> absorber
        v1 = -v01
        # v2: n1 -> n2
        v2 = v12

        cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))
        ref_angles.append(angle)

    # Calculate processed (unwrapped & aligned) positions
    # (By default align=True, which does Kabsch alignment and rotates coordinates)
    unwrapped = process_trajectory(structures, align=True)

    # Run the calculate_grouped_msrd function
    res_2b, res_3b = calculate_grouped_msrd(
        structures=structures,
        unwrapped_positions=unwrapped,
        central_indices=[0],
        central_label="Mn.1",
        cutoff=4.0,
        cutoff_3body=4.0,
        exclude_hydrogen=False,
    )

    # Assert that 2-body matches reference within tiny tolerance
    calc_reff_2b = sorted([r["reff"] for r in res_2b])
    expected_reff_2b = sorted([float(np.mean(ref_d_01)), float(np.mean(ref_d_02))])

    # Show difference to aid debugging/assertion error output
    print("CALCULATED 2-BODY REFF:", calc_reff_2b)
    print("EXPECTED 2-BODY REFF:", expected_reff_2b)

    np.testing.assert_allclose(calc_reff_2b, expected_reff_2b, rtol=1e-5)

    # Assert that 3-body angle matches reference within tolerance
    calc_angle_3b = res_3b[0]["angle"]
    expected_angle_3b = float(np.mean(ref_angles))

    print("CALCULATED 3-BODY ANGLE:", calc_angle_3b)
    print("EXPECTED 3-BODY ANGLE:", expected_angle_3b)

    np.testing.assert_allclose(calc_angle_3b, expected_angle_3b, rtol=1e-5)


def test_max_safe_mic_cutoff_orthorhombic():
    """For an orthorhombic cell the safe cutoff is half the smallest axis."""
    from larch_cli_wrapper.debye_waller_core import _max_safe_mic_cutoff

    # a=4, b=5, c=6 orthogonal cell
    cell = np.diag([4.0, 5.0, 6.0])
    expected = 4.0 / 2.0
    assert _max_safe_mic_cutoff(cell) == pytest.approx(expected, rel=1e-10)


def test_max_safe_mic_cutoff_non_orthogonal():
    """Check the safe cutoff for a non-orthogonal parallelepiped.

    The maximum safe MIC cutoff is half the smallest perpendicular distance
    between opposite faces of the cell.
    """
    from larch_cli_wrapper.debye_waller_core import _max_safe_mic_cutoff

    # Rhombohedral-like cell with all sides equal and one angle.
    a = 3.0
    alpha = np.radians(60.0)
    cell = np.array(
        [
            [a, 0.0, 0.0],
            [a * np.cos(alpha), a * np.sin(alpha), 0.0],
            [
                a * np.cos(alpha),
                a * (np.cos(alpha) - np.cos(alpha) ** 2) / np.sin(alpha),
                a
                * np.sqrt(1 - 3 * np.cos(alpha) ** 2 + 2 * np.cos(alpha) ** 3)
                / np.sin(alpha),
            ],
        ]
    )

    # Just verify it's positive and finite, and consistent with direct geometry
    max_cutoff = _max_safe_mic_cutoff(cell)
    assert max_cutoff is not None
    assert max_cutoff > 0.0

    # For equal-length rhombohedral cell, all face distances are equal
    volume = np.abs(np.linalg.det(cell))
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    expected = (volume / area) / 2.0
    assert max_cutoff == pytest.approx(expected, rel=1e-10)


def test_max_safe_mic_cutoff_zero_volume():
    """Zero-volume cells return None (no safe MIC radius)."""
    from larch_cli_wrapper.debye_waller_core import _max_safe_mic_cutoff

    cell = np.zeros((3, 3))
    assert _max_safe_mic_cutoff(cell) is None


def test_msrd_warns_on_cutoff_exceeding_safe_radius(caplog):
    """calculate_grouped_msrd should warn when a cutoff is too large."""
    from larch_cli_wrapper.debye_waller_core import calculate_grouped_msrd

    cell = np.diag([2.0, 2.0, 2.0])  # safe radius = 1.0 Å
    atoms = Atoms(
        "Mn2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=cell, pbc=True
    )
    structures = [atoms, atoms]
    unwrapped = np.array([atoms.get_positions() for atoms in structures])

    # Use a cutoff larger than the safe MIC radius
    with caplog.at_level("WARNING", logger="larch_cli_wrapper.debye_waller_core"):
        calculate_grouped_msrd(
            structures=structures,
            unwrapped_positions=unwrapped,
            central_indices=[0],
            central_label="Mn.1",
            cutoff=1.5,
            cutoff_3body=0,
            exclude_hydrogen=False,
        )

    assert any(
        "exceeds the maximum safe MIC cutoff" in rec.message for rec in caplog.records
    )


def test_msrd_no_warning_for_safe_cutoff(caplog):
    """No warning should be emitted when cutoffs are within the safe radius."""
    from larch_cli_wrapper.debye_waller_core import calculate_grouped_msrd

    cell = np.diag([4.0, 4.0, 4.0])  # safe radius = 2.0 Å
    atoms = Atoms(
        "Mn2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=cell, pbc=True
    )
    structures = [atoms, atoms]
    unwrapped = np.array([atoms.get_positions() for atoms in structures])

    with caplog.at_level("WARNING", logger="larch_cli_wrapper.debye_waller_core"):
        calculate_grouped_msrd(
            structures=structures,
            unwrapped_positions=unwrapped,
            central_indices=[0],
            central_label="Mn.1",
            cutoff=1.5,
            cutoff_3body=0,
            exclude_hydrogen=False,
        )

    cutoff_warnings = [
        rec for rec in caplog.records if "maximum safe MIC cutoff" in rec.message
    ]
    assert len(cutoff_warnings) == 0


def test_msrd_uses_ase_consistent_mic_for_nonorthogonal_cell():
    cell = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.6, 3.7, 0.0],
            [0.0, 0.0, 4.2],
        ]
    )
    frames = [
        Atoms(
            "MnO",
            scaled_positions=[[0.03, 0.5, 0.5], [0.97, 0.5, 0.5]],
            cell=cell,
            pbc=True,
        ),
        Atoms(
            "MnO",
            scaled_positions=[[0.04, 0.5, 0.5], [0.96, 0.5, 0.5]],
            cell=cell,
            pbc=True,
        ),
    ]

    dummy_processed_positions = np.zeros((len(frames), len(frames[0]), 3))

    res_2b, res_3b = calculate_grouped_msrd(
        frames,
        dummy_processed_positions,
        central_indices=[0],
        central_label="Mn",
        cutoff=1.0,
        cutoff_3body=0,
    )

    expected_distances = np.array(
        [atoms.get_distance(0, 1, mic=True) for atoms in frames]
    )

    assert len(res_2b) == 1
    assert res_2b[0]["type"] == "Mn-O"
    assert np.isclose(res_2b[0]["reff"], expected_distances.mean())
    assert np.isclose(res_2b[0]["sigma2"], np.var(expected_distances, ddof=1))
    assert res_2b[0]["atom_indices"] == [(0, 1)]
    assert res_3b == []
