"""Tests for the layered path-instance architecture in debye_waller_core."""

import numpy as np
import pytest
from ase import Atoms

from larch_cli_wrapper.debye_waller_core import (
    ExternalPath,
    calculate_grouped_msrd,
    canonical_intermediates,
    catalogue_from_sequences,
    compute_instance_statistics,
    enumerate_path_instances,
    group_path_instances,
    kabsch_align,
    match_msrd_paths_to_feff,
    match_path_instances_to_feff,
    merge_congruent_3body_groups,
    sample_path_instances,
)


def _rotation_matrix_z(theta_deg: float) -> np.ndarray:
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _make_frames(
    base_positions: np.ndarray,
    symbols: list[str],
    n_frames: int = 30,
    noise: float = 0.03,
    cell: float | np.ndarray = 20.0,
    seed: int = 1,
) -> list[Atoms]:
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n_frames):
        pos = base_positions + rng.normal(0.0, noise, base_positions.shape)
        frames.append(Atoms(symbols, positions=pos, cell=[cell, cell, cell], pbc=True))
    return frames


# ---------------------------------------------------------------------------
# Kabsch regression (rotation direction bug)
# ---------------------------------------------------------------------------


def test_kabsch_align_recovers_known_rotation():
    """Frames rotated by a known rotation must align back onto the reference."""
    rng = np.random.default_rng(7)
    ref = rng.normal(size=(8, 3))
    # Nontrivial rotation: 37 deg about z followed by 25 deg about x.
    Rz = _rotation_matrix_z(37.0)
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(np.radians(25.0)), -np.sin(np.radians(25.0))],
            [0.0, np.sin(np.radians(25.0)), np.cos(np.radians(25.0))],
        ]
    )
    R_true = Rz @ Rx
    rotated = ref @ R_true + np.array([3.0, -2.0, 1.0])

    positions = np.stack([ref, rotated])
    aligned = kabsch_align(positions, reference_idx=0)

    np.testing.assert_allclose(aligned[0], ref, atol=1e-10)
    np.testing.assert_allclose(aligned[1], ref, atol=1e-10)


# ---------------------------------------------------------------------------
# Catalogue / sampling layers
# ---------------------------------------------------------------------------


def test_canonical_intermediates_reversal():
    assert canonical_intermediates((5, 2)) == (2, 5)
    assert canonical_intermediates((2, 5)) == (2, 5)
    assert canonical_intermediates((3,)) == (3,)


def test_enumerate_respects_cutoff_3body_larger_than_cutoff():
    """A neighbour beyond `cutoff` but within `cutoff_3body` must be seen."""
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    frames = _make_frames(base, ["K", "N", "C"], noise=0.0)
    catalogue = enumerate_path_instances(frames[0], [0], cutoff=3.0, cutoff_3body=5.0)
    n2 = [p for p in catalogue if p.nleg == 2]
    n3 = [p for p in catalogue if p.nleg == 3]
    # 2-body only sees N (2.0 A); the 3-body path must include C (4.0 A).
    assert {p.path_id.intermediates for p in n2} == {(1,)}
    assert {p.path_id.intermediates for p in n3} == {(1, 2)}


def test_enumerate_cutoff_3body_is_reff_not_leg_cutoff():
    """cutoff_3body limits the 3-body *Reff* (half perimeter), not leg length.

    A triangle can have every leg within the cutoff yet an Reff beyond it
    (fat triangle); such paths must be excluded. Conversely a thin triangle
    with Reff within the cutoff is kept even though its perimeter is ~2x the
    cutoff. This matches FEFF's path-length-based path selection.
    """
    # Triangle with absorber legs 2.55/3.61 and n1-n2 leg 4.42:
    # all legs < 5.0, but Reff = (2.55 + 3.61 + 4.42)/2 = 5.29 > 5.0.
    base = np.array([[0.0, 0.0, 0.0], [2.55, 0.0, 0.0], [0.0, 3.61, 0.0]])
    frames = _make_frames(base, ["Cu", "Cu", "Cu"], noise=0.0)

    cat = enumerate_path_instances(frames[0], [0], cutoff=4.0, cutoff_3body=5.0)
    assert [p for p in cat if p.nleg == 3] == []

    cat = enumerate_path_instances(frames[0], [0], cutoff=4.0, cutoff_3body=5.3)
    assert len([p for p in cat if p.nleg == 3]) == 1

    # Collinear A-B-C (2 + 2 + 4): Reff = 4.0, kept with cutoff_3body=4.0
    # even though the total path length (8 A) is ~2x the cutoff.
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    frames = _make_frames(base, ["K", "N", "C"], noise=0.0)
    cat = enumerate_path_instances(frames[0], [0], cutoff=3.0, cutoff_3body=4.0)
    assert len([p for p in cat if p.nleg == 3]) == 1


def test_enumerate_rejects_mixed_absorbers():
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    frames = _make_frames(base, ["K", "N", "C"], noise=0.0)
    with pytest.raises(ValueError, match="same element"):
        enumerate_path_instances(frames[0], [0, 1], cutoff=3.0)


def test_sample_path_instances_legs_and_angles():
    """Collinear A-B-C: internal angle ~180 at B, FEFF beta ~0 at B."""
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    frames = _make_frames(base, ["K", "N", "C"], n_frames=20, noise=0.01)
    catalogue = enumerate_path_instances(frames[0], [0], cutoff=3.0, cutoff_3body=5.0)
    samples = sample_path_instances(frames, catalogue)
    triplet = next(s for s in samples if s.instance.nleg == 3)

    assert triplet.legs.shape == (20, 3)
    np.testing.assert_allclose(triplet.legs[:, 0], 2.0, atol=0.05)
    np.testing.assert_allclose(triplet.legs[:, 1], 2.0, atol=0.05)
    np.testing.assert_allclose(triplet.legs[:, 2], 4.0, atol=0.05)
    np.testing.assert_allclose(triplet.reff, triplet.legs.sum(axis=1) / 2.0, rtol=1e-12)
    # Vertex order is (A, B, C): internal angle at B ~180, beta at B ~0.
    assert triplet.internal_angles is not None
    assert triplet.feff_beta is not None
    assert np.mean(triplet.internal_angles[:, 1]) == pytest.approx(180.0, abs=2.0)
    assert np.mean(triplet.feff_beta[:, 1]) == pytest.approx(0.0, abs=2.0)


def test_sample_remaps_variable_cell_instead_of_rejecting():
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    frames = _make_frames(base, ["K", "N"], n_frames=3, noise=0.0)
    frames[1].set_cell([21.0, 20.0, 20.0])
    catalogue = enumerate_path_instances(frames[0], [0], cutoff=3.0)
    # Variable-cell trajectories are now supported via remapping to the
    # reference cell — sampling must succeed, not raise.
    samples = sample_path_instances(frames, catalogue)
    assert len(samples) == 1
    assert samples[0].reff[0] == pytest.approx(2.0, abs=0.05)


def test_catalogue_from_sequences_orientation_aliases():
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    frames = _make_frames(base, ["K", "N", "C"], noise=0.0)
    catalogue = catalogue_from_sequences(frames[0], 0, [(2, 1), (1, 2)])
    # Both orderings collapse to one canonical instance with two orientations.
    assert len(catalogue) == 1
    inst = catalogue[0]
    assert inst.path_id.intermediates == (1, 2)
    assert sorted(inst.orientations) == [(1, 2), (2, 1)]
    assert inst.reference_reff == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Statistics / grouping layers
# ---------------------------------------------------------------------------


def test_group_reports_thermal_and_effective_variance():
    """Two instances with different means: effective variance must exceed
    the thermal (within-instance) variance."""
    base = np.array(
        [[0.0, 0.0, 0.0], [2.00, 0.0, 0.0], [0.0, 10.0, 0.0], [2.20, 10.0, 0.0]]
    )
    frames = _make_frames(base, ["K", "N", "K", "N"], n_frames=40, noise=0.01)
    res_2b, _ = calculate_grouped_msrd(frames, [0, 2], "K", cutoff=3.0, tol_dist=0.5)
    assert len(res_2b) == 1
    group = res_2b[0]
    assert group["count"] == 2
    assert group["sigma2_thermal_A2"] < group["sigma2"]
    assert group["sigma2_effective_A2"] == group["sigma2"]
    assert group["between_instance_reff_var_A2"] > 0.01


def test_group_result_schema():
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    frames = _make_frames(base, ["K", "N", "C"], n_frames=20, noise=0.01)
    res_2b, res_3b = calculate_grouped_msrd(
        frames, [0], "K", cutoff=3.0, cutoff_3body=5.0
    )
    for key in (
        "sigma2_thermal_A2",
        "between_instance_reff_var_A2",
        "delta_reff_A",
        "third_cumulant_A3",
        "fourth_cumulant_A4",
        "effective_n_samples",
        "absorber",
        "path_ids",
    ):
        assert key in res_2b[0]
    for key in (
        "internal_angle_deg",
        "feff_beta_deg",
        "vertex_angles_deg",
        "feff_beta_seq_deg",
        "leg_lengths_A",
    ):
        assert key in res_3b[0]
    assert res_3b[0]["feff_beta_deg"] == pytest.approx(
        180.0 - res_3b[0]["internal_angle_deg"]
    )
    assert len(res_3b[0]["vertex_angles_deg"]) == 3
    assert len(res_3b[0]["leg_lengths_A"]) == 3


def test_instance_statistics_keys():
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    frames = _make_frames(base, ["K", "N"], n_frames=25, noise=0.02)
    catalogue = enumerate_path_instances(frames[0], [0], cutoff=3.0)
    samples = sample_path_instances(frames, catalogue)
    stats = compute_instance_statistics(samples)
    assert len(stats) == 1
    st = stats[0]
    assert st["n_frames"] == 25
    assert st["mean_reff_A"] == pytest.approx(2.0, abs=0.05)
    assert st["sigma2_A2"] > 0.0
    assert st["effective_n_samples"] >= 1.0


def test_group_path_instances_roundtrip():
    """The layered pipeline must reproduce the wrapper's results."""
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    frames = _make_frames(base, ["K", "N"], n_frames=20, noise=0.02)
    catalogue = enumerate_path_instances(frames[0], [0], cutoff=3.0)
    samples = sample_path_instances(frames, catalogue)
    stats = compute_instance_statistics(samples)
    res_2b, res_3b = group_path_instances(samples, stats)
    wrap_2b, wrap_3b = calculate_grouped_msrd(frames, [0], "K", cutoff=3.0)
    assert [r["reff"] for r in res_2b] == [r["reff"] for r in wrap_2b]
    assert res_3b == wrap_3b == []


def test_reversal_equivalent_triangles_merge_into_one_group():
    """A→B→C→A and A→C→B→A are the same closed path and must not be split
    by the arbitrary canonical atom ordering."""
    # Triangle with absorber legs 2.55 and 3.61 at ~90 deg -> BC ~ 4.42.
    # Structure 1: lower atom index has the LONGER absorber leg.
    frames1 = _make_frames(
        np.array([[0.0, 0.0, 0.0], [0.0, 3.61, 0.0], [2.55, 0.0, 0.0]]),
        ["Cu", "Cu", "Cu"],
        n_frames=10,
        noise=0.01,
        seed=3,
    )
    # Structure 2: lower atom index has the SHORTER absorber leg.
    frames2 = _make_frames(
        np.array([[0.0, 0.0, 0.0], [2.55, 0.0, 0.0], [0.0, 3.61, 0.0]]),
        ["Cu", "Cu", "Cu"],
        n_frames=10,
        noise=0.01,
        seed=4,
    )
    samples = []
    for frames in (frames1, frames2):
        # Triangle legs are 2.55/3.61/4.42 Å -> Reff ~5.29 Å, so the Reff-
        # based 3-body cutoff must exceed that to keep both instances.
        cat = enumerate_path_instances(frames[0], [0], cutoff=4.0, cutoff_3body=5.5)
        samples.extend(sample_path_instances(frames, cat))
    triplets = [s for s in samples if s.instance.nleg == 3]
    assert len(triplets) == 2
    # The two instances disagree on the first-scatterer angle (~35 vs ~55).
    stats = compute_instance_statistics(triplets)
    first_angles = sorted(st["mean_internal_angles_deg"][1] for st in stats)
    assert first_angles[1] - first_angles[0] > 10.0

    _res_2b, res_3b = group_path_instances(triplets, stats, tol_dist=0.1, tol_angle=5.0)
    assert len(res_3b) == 1
    group = res_3b[0]
    assert group["count"] == 2
    # Anchored angle: at the scatterer with the shorter absorber leg (~54.7).
    assert group["internal_angle_deg"] == pytest.approx(54.7, abs=1.0)
    s_min, s_max = group["scatterer_angles_deg"]
    assert s_min == pytest.approx(35.3, abs=1.0)
    assert s_max == pytest.approx(54.7, abs=1.0)


# ---------------------------------------------------------------------------
# FEFF matching additions
# ---------------------------------------------------------------------------


def _feff_path(nlegs, scatterer, reff, **extra):
    pg = {
        "nlegs": nlegs,
        "scatterer": scatterer,
        "r_effs": [reff],
        "r_eff_ref": reff,
        "contribution_pct": 10.0,
    }
    pg.update(extra)
    return pg


def test_rattle_sigma2_transform_is_explicit():
    dw = [{"type": "K-N", "reff": 2.0, "sigma2": 0.01, "count": 1, "absorber": "K"}]
    feff = [_feff_path(4, "N", 4.0)]  # reff/2 = 2.0 matches the bond
    rows = match_msrd_paths_to_feff(dw, None, feff, r_tol=0.1)
    assert rows[0]["nlegs_feff"] == 4
    assert rows[0]["sigma2"] == pytest.approx(0.01)
    assert rows[0]["sigma2_feff"] == pytest.approx(0.04)


def test_non_rattle_4leg_path_is_not_matched():
    """A 4-leg path failing the rattle geometry check must be rejected."""
    dw = [{"type": "K-N", "reff": 2.0, "sigma2": 0.01, "count": 1}]
    feff = [
        _feff_path(4, "N", 4.0, rlegs=(1.0, 2.0, 1.5, 2.5)),  # uneven legs
        _feff_path(4, "N", 4.0, rlegs=(2.0, 2.0, 2.0, 2.0), betas=(90.0, 90.0, 90.0)),
        _feff_path(
            4, "N", 4.0, rlegs=(2.0, 2.0, 2.0, 2.0), betas=(180.0, 180.0, 180.0)
        ),
    ]
    rows = match_msrd_paths_to_feff(dw, None, feff, r_tol=0.1)
    # Only the genuine rattle path (index 2) is eligible.
    assert rows[0]["group_idx"] == 2


def test_ambiguity_flag():
    dw = [{"type": "K-N", "reff": 2.0, "sigma2": 0.01, "count": 1}]
    feff = [_feff_path(2, "N", 2.001), _feff_path(2, "N", 2.002)]
    rows = match_msrd_paths_to_feff(dw, None, feff, r_tol=0.1)
    assert rows[0]["ambiguous"] is True
    feff_far = [_feff_path(2, "N", 2.001), _feff_path(2, "N", 2.09)]
    rows_far = match_msrd_paths_to_feff(dw, None, feff_far, r_tol=0.1)
    assert rows_far[0]["ambiguous"] is False


def test_absorber_gate():
    dw = [{"type": "K-N", "reff": 2.0, "sigma2": 0.01, "count": 1, "absorber": "K"}]
    feff_match = [_feff_path(2, "N", 2.0, absorber="K")]
    feff_mismatch = [_feff_path(2, "N", 2.0, absorber="Rb")]
    rows = match_msrd_paths_to_feff(dw, None, feff_match, r_tol=0.1)
    assert rows[0]["group_idx"] == 0
    rows = match_msrd_paths_to_feff(dw, None, feff_mismatch, r_tol=0.1)
    assert rows[0]["group_idx"] is None


def test_match_accepts_numpy_array_r_effs():
    dw = [{"type": "K-N", "reff": 2.0, "sigma2": 0.01, "count": 1}]
    feff = [_feff_path(2, "N", 2.0)]
    feff[0]["r_effs"] = np.array([2.0, 2.0, 2.0])
    rows = match_msrd_paths_to_feff(dw, None, feff, r_tol=0.1)
    assert rows[0]["group_idx"] == 0


# ---------------------------------------------------------------------------
# Ordered external-path matching
# ---------------------------------------------------------------------------


def test_match_path_instances_to_feff_both_orientations():
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    frames = _make_frames(base, ["K", "N", "C"], n_frames=20, noise=0.01)
    catalogue = enumerate_path_instances(frames[0], [0], cutoff=3.0, cutoff_3body=5.0)
    samples = sample_path_instances(frames, catalogue)

    ext = ExternalPath(
        source="feff",
        run_id="run1",
        external_path_id="0001",
        nleg=3,
        absorber_key="K",
        intermediate_keys=("N", "C"),
        rlegs=(2.0, 2.0, 4.0),
        betas=(0.0, 180.0),
    )
    rows = match_path_instances_to_feff(samples, [ext])
    assert len(rows) == 1
    assert rows[0]["orientation"] == (1, 2)

    # Reversed ordered path must match via the reversed orientation.
    ext_rev = ExternalPath(
        source="feff",
        run_id="run1",
        external_path_id="0002",
        nleg=3,
        absorber_key="K",
        intermediate_keys=("C", "N"),
        rlegs=(4.0, 2.0, 2.0),
        betas=(180.0, 0.0),
    )
    rows_rev = match_path_instances_to_feff(samples, [ext_rev])
    assert len(rows_rev) == 1
    assert rows_rev[0]["orientation"] == (2, 1)


def test_match_path_instances_to_feff_gates():
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    frames = _make_frames(base, ["K", "N", "C"], n_frames=20, noise=0.01)
    catalogue = enumerate_path_instances(frames[0], [0], cutoff=3.0, cutoff_3body=5.0)
    samples = sample_path_instances(frames, catalogue)

    wrong_absorber = ExternalPath(
        source="feff",
        run_id="r",
        external_path_id="x",
        nleg=3,
        absorber_key="Rb",
        intermediate_keys=("N", "C"),
    )
    wrong_species = ExternalPath(
        source="feff",
        run_id="r",
        external_path_id="y",
        nleg=3,
        absorber_key="K",
        intermediate_keys=("O", "C"),
    )
    wrong_rlegs = ExternalPath(
        source="feff",
        run_id="r",
        external_path_id="z",
        nleg=3,
        absorber_key="K",
        intermediate_keys=("N", "C"),
        rlegs=(2.0, 2.0, 6.0),
    )
    assert match_path_instances_to_feff(samples, [wrong_absorber]) == []
    assert match_path_instances_to_feff(samples, [wrong_species]) == []
    assert match_path_instances_to_feff(samples, [wrong_rlegs]) == []


# ---------------------------------------------------------------------------
# Merging congruent 3-body groups (no FEFF file needed)
# ---------------------------------------------------------------------------


def _msrd_3b_group(path_type, reff, angle, leg_lengths, sigma2=0.01, count=24):
    return {
        "type": path_type,
        "reff": reff,
        "sigma2": sigma2,
        "count": count,
        "internal_angle_deg": angle,
        "leg_lengths_A": leg_lengths,
    }


def test_merge_congruent_groups_reproduces_real_mno_case():
    """Regression: the real Mn-Mn-O 4.4723/4.5026 pair (same congruent
    triangle -- legs {2.146, 3.061, 3.737} vs {2.178, 3.061, 3.765} sorted
    -- reported at 55.0 deg vs 90.3 deg because the anchored-angle
    convention picks a different vertex for each), must be merged.
    """
    res_3b = [
        _msrd_3b_group(
            "Mn-Mn-O", 4.4723, 55.0, (2.1464, 3.7367, 3.0615), sigma2=0.012606
        ),
        _msrd_3b_group(
            "Mn-Mn-O", 4.5026, 90.3, (3.0615, 2.1783, 3.7655), sigma2=0.012501
        ),
    ]

    merged = merge_congruent_3body_groups(res_3b, leg_tol=0.1)

    assert len(merged) == 1
    m = merged[0]
    assert m["n_dw_groups"] == 2
    assert m["count_total"] == 48
    assert m["reff_combined"] == pytest.approx(4.4875, abs=1e-3)
    # Combined sigma2 must exceed the naive (between-group-blind) average,
    # since the two populations' Reff means genuinely differ.
    naive_avg = (0.012606 + 0.012501) / 2.0
    assert m["sigma2_combined"] > naive_avg


def test_merge_congruent_groups_leaves_distinct_triangles_separate():
    """Genuinely different triangles (different leg-length sets) at a
    similar Reff must NOT be merged, even if their tolerance windows would
    otherwise overlap on Reff alone."""
    res_3b = [
        _msrd_3b_group("Mn-O-O", 3.66, 91.0, (2.15, 2.15, 3.06)),  # near-neighbour
        _msrd_3b_group("Mn-O-O", 3.69, 45.0, (2.20, 2.20, 1.60)),  # unrelated shape
    ]

    merged = merge_congruent_3body_groups(res_3b, leg_tol=0.1)

    assert len(merged) == 2
    assert all(m["n_dw_groups"] == 1 for m in merged)


def test_merge_congruent_groups_single_group_is_a_no_op():
    res_3b = [_msrd_3b_group("Fe-N-C", 3.50, 160.0, (2.0, 2.0, 5.0), sigma2=0.005)]

    merged = merge_congruent_3body_groups(res_3b)

    assert len(merged) == 1
    m = merged[0]
    assert m["n_dw_groups"] == 1
    assert m["reff_combined"] == pytest.approx(3.50)
    assert m["sigma2_combined"] == pytest.approx(0.005)
    assert m["between_group_variance_A2"] == pytest.approx(0.0)


def test_merge_congruent_groups_transitively_chains_three():
    """A chain of three mutually-congruent (within tol) groups merges into
    one cluster via the union-find over the pairwise congruence graph."""
    res_3b = [
        _msrd_3b_group("Mn-Mn-Mn", 6.34, 54.7, (3.061, 4.320, 5.293)),
        _msrd_3b_group("Mn-Mn-Mn", 6.37, 90.5, (3.061, 4.346, 5.336)),
        _msrd_3b_group("Mn-Mn-Mn", 6.38, 90.5, (3.099, 4.320, 5.336)),
    ]

    merged = merge_congruent_3body_groups(res_3b, leg_tol=0.05)

    assert len(merged) == 1
    assert merged[0]["n_dw_groups"] == 3


def test_variable_cell_trajectory_is_remapped_not_rejected():
    """A variable-cell (NPT) trajectory is remapped to the reference cell
    instead of raising NotImplementedError, and 2-/3-body MSRD is computed.
    """
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    frames = _make_frames(base, ["K", "N", "C"], n_frames=5, noise=0.01)
    # Stretch each frame's cell slightly differently (NPT-style).
    for k, atoms in enumerate(frames):
        scale = 1.0 + 0.01 * (k + 1)
        atoms.set_cell(atoms.get_cell() * scale, scale_atoms=True)

    res_2b, res_3b = calculate_grouped_msrd(
        frames, [0], "K", cutoff=3.0, cutoff_3body=5.0
    )
    assert len(res_2b) >= 1
    assert len(res_3b) >= 1
    # Sanity: the N neighbour at ~2 Å survives with a reasonable Reff.
    assert res_2b[0]["reff"] == pytest.approx(2.0, abs=0.05)


def test_variable_cell_normalize_passes_constant_cell_through():
    """Constant-cell trajectories are returned unchanged (no copies)."""
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    frames = _make_frames(base, ["K", "N"], n_frames=3, noise=0.01)
    from larch_cli_wrapper.debye_waller_core import _normalize_cells

    out = _normalize_cells(frames)
    assert out is frames
