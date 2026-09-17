import numpy as np
import pytest
from ase import Atoms

from larch_cli_wrapper.debye_waller_core import calculate_grouped_msrd


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

    # Run the calculate_grouped_msrd function
    res_2b, res_3b = calculate_grouped_msrd(
        structures=structures,
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

    # A single atom-triplet contributes to this cluster, so the pooling
    # spread (variance across the per-triplet mean angles pooled into the
    # shell) is exactly zero.
    assert res_3b[0]["angle_var"] == 0.0


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

    # Use a cutoff larger than the safe MIC radius
    with caplog.at_level("WARNING", logger="larch_cli_wrapper.debye_waller_core"):
        calculate_grouped_msrd(
            structures=structures,
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

    with caplog.at_level("WARNING", logger="larch_cli_wrapper.debye_waller_core"):
        calculate_grouped_msrd(
            structures=structures,
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

    res_2b, res_3b = calculate_grouped_msrd(
        frames,
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


def test_unwrapped_b_factors_non_orthogonal_cell():
    from larch_cli_wrapper.debye_waller_core import (
        compute_adp_results,
        process_trajectory,
    )

    cell = np.array(
        [
            [25.0, 0.0, 0.0],
            [12.5, 21.6, 0.0],
            [12.5, 7.2, 20.4],
        ]
    )
    # Atoms vibrating near the unit cell boundary
    np.random.seed(42)
    frames = []
    base_pos = np.array(
        [
            [12.5, 7.2, 20.3],  # near boundary in non-orthogonal cell
            [0.1, 0.1, 0.1],
            [12.0, 10.0, 5.0],
        ]
    )
    for _ in range(20):
        pos = base_pos + np.random.normal(0, 0.02, base_pos.shape)
        frames.append(Atoms("MnO2", positions=pos, cell=cell, pbc=True))

    processed = process_trajectory(frames, align=True)
    adp_results = compute_adp_results(frames, processed)
    mean_b = float(np.mean(adp_results["b_factors"]))

    # B-factors must remain physically reasonable (< 2.0 Å²), not inflated to >100 Å²
    assert mean_b < 2.0


def _parse_cif_loops(cif_text):
    """Return (frac_positions, aniso_U_rows) parsed from save_cif_with_adp output."""
    lines = [ln.strip() for ln in cif_text.splitlines() if ln.strip()]
    frac, aniso = [], []
    section = None
    for ln in lines:
        if ln.startswith("_atom_site_B_iso_or_equiv"):
            section = "sites"
            continue
        if ln.startswith("_atom_site_aniso_U_12"):
            section = "aniso"
            continue
        if ln.startswith(("_", "loop_", "data_")):
            continue
        tokens = ln.split()
        if section == "sites":
            frac.append([float(x) for x in tokens[2:5]])
        elif section == "aniso":
            # U11 U22 U33 U23 U13 U12
            aniso.append([float(x) for x in tokens[1:7]])
    return np.array(frac), np.array(aniso)


def test_cif_adp_convention_non_orthogonal():
    """CIF output uses the crystal-axes U convention and correct fractional coords.

    Oracle: the CIF convention satisfies U_cart = A N U_cif N A^T with A the
    column-vector cell matrix and N = diag(a*_i), so reconstructing U_cart
    from the written values must recover the input tensor.
    """
    from larch_cli_wrapper.debye_waller_core import save_cif_with_adp

    cell = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.6, 3.7, 0.0],
            [0.5, 0.9, 4.2],
        ]
    )
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.5, 3.0, size=(2, 3))
    # Symmetric positive-definite Cartesian U per atom
    u_cart = np.empty((2, 3, 3))
    for i in range(2):
        m = rng.normal(0.0, 0.05, size=(3, 3))
        u_cart[i] = m @ m.T + 0.01 * np.eye(3)

    results = {
        "avg_positions": pos,
        "atom_names": ["Mn", "O"],
        "u_tensor": u_cart,
        "avg_cell": cell,
        "b_factors": 8 * np.pi**2 * np.trace(u_cart, axis1=1, axis2=2) / 3,
    }
    frac, aniso = _parse_cif_loops(save_cif_with_adp(results))

    # Fractional coordinates: r = f @ cell (ASE row-vector convention)
    np.testing.assert_allclose(frac @ cell, pos, atol=5e-5)

    # Reconstruct U_cart from the written CIF U^ij values
    a_col = cell.T
    a_star = np.linalg.norm(np.linalg.inv(cell), axis=0)
    n_mat = np.diag(a_star)
    for i in range(2):
        u11, u22, u33, u23, u13, u12 = aniso[i]
        u_cif = np.array(
            [
                [u11, u12, u13],
                [u12, u22, u23],
                [u13, u23, u33],
            ]
        )
        reconstructed = a_col @ n_mat @ u_cif @ n_mat @ a_col.T
        np.testing.assert_allclose(reconstructed, u_cart[i], atol=5e-4)


def test_cif_adp_convention_orthogonal_noop():
    """For a diagonal cell the CIF U values equal the Cartesian tensor."""
    from larch_cli_wrapper.debye_waller_core import cartesian_u_to_cif

    cell = np.diag([4.0, 5.0, 6.0])
    u_cart = np.array(
        [[[0.02, 0.003, 0.001], [0.003, 0.03, 0.002], [0.001, 0.002, 0.04]]]
    )
    np.testing.assert_allclose(cartesian_u_to_cif(u_cart, cell), u_cart, atol=1e-12)
