"""Tests for matching MSRD/Debye-Waller path groups to FEFF path groups.

The paths explorer notebook assigns each MSRD (Debye-Waller σ²) group a FEFF
scattering path by nearest effective distance ``Reff``.  The matching logic
lives in :func:`larch_cli_wrapper.debye_waller_core.match_msrd_paths_to_feff`.

The matcher uses a genuine (non-exclusive) nearest-neighbor assignment: each
MSRD group independently finds its nearest eligible FEFF path.  Several MSRD
groups *may* legitimately share the same FEFF path (e.g. a spuriously split
MD shell, or real crystallographic degeneracy) — this is reported via
``shared_with``/``contribution_pct_share`` rather than arbitrated away, since
silently hiding it would throw away a real diagnostic signal.
"""

from __future__ import annotations

import pytest

from larch_cli_wrapper.debye_waller_core import (
    extract_path_matching_diagnostics,
    match_msrd_paths_to_feff,
    match_paths_within_tolerance,
    pool_dw_groups_by_feff_path,
)


def _feff_path(nlegs, scatterer, r_eff, contribution_pct=10.0, angle=None):
    """Build a minimal FEFF path-group dict as consumed by the matcher."""
    return {
        "nlegs": nlegs,
        "scatterer": scatterer,
        "r_effs": [r_eff],
        "r_eff_ref": r_eff,
        "contribution_pct": contribution_pct,
        "angle": angle,
    }


def _msrd_2b(path_type, reff, sigma2=0.005, count=4):
    return {"type": path_type, "reff": reff, "sigma2": sigma2, "count": count}


def _msrd_3b(path_type, reff, angle, sigma2=0.005, count=4, angle_var=None):
    return {
        "type": path_type,
        "reff": reff,
        "angle": angle,
        "angle_var": angle_var,
        "sigma2": sigma2,
        "count": count,
    }


def test_basic_one_to_one_matching():
    """Two well-separated MSRD groups map to their own distinct FEFF paths."""
    res_2b = [
        _msrd_2b("Fe-N", reff=2.00),
        _msrd_2b("Fe-N", reff=3.00),
    ]
    path_groups = [
        _feff_path(nlegs=2, scatterer="N", r_eff=2.01),
        _feff_path(nlegs=2, scatterer="N", r_eff=2.99),
    ]

    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)

    matched = [r["group_idx"] for r in rows if r["group_idx"] is not None]
    assert matched == [0, 1]
    assert len(set(matched)) == len(matched)  # no collisions


def test_no_match_when_out_of_tolerance():
    """An MSRD group with no FEFF path within tolerance stays unmatched."""
    res_2b = [_msrd_2b("Fe-N", reff=2.00)]
    path_groups = [_feff_path(nlegs=2, scatterer="N", r_eff=2.50)]

    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)

    assert rows[0]["group_idx"] is None
    assert rows[0]["feff_note"] == "(no FEFF match)"


def test_scatterer_element_must_match():
    """A closer FEFF path with the wrong scatterer element is not chosen."""
    res_2b = [_msrd_2b("Fe-N", reff=2.00)]
    path_groups = [
        _feff_path(nlegs=2, scatterer="O", r_eff=2.00),  # wrong element, exact R
        _feff_path(nlegs=2, scatterer="N", r_eff=2.05),  # right element, close R
    ]

    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)

    assert rows[0]["group_idx"] == 1


def test_degenerate_msrd_groups_share_one_feff_path():
    """Several MSRD groups may legitimately share a single FEFF path.

    Two MSRD groups (Reff 2.00 and 2.05) both fall within ``r_tol`` of the
    single available FEFF path (Reff 2.02). Both should match it independently
    (true nearest-neighbor, no cross-row exclusivity), with the FEFF path's
    contribution split evenly between them and each row aware of the other
    via ``shared_with``.
    """
    res_2b = [
        _msrd_2b("Fe-N", reff=2.00),
        _msrd_2b("Fe-N", reff=2.05),
    ]
    # Only ONE FEFF single-scattering path for this element.
    path_groups = [
        _feff_path(nlegs=2, scatterer="N", r_eff=2.02, contribution_pct=10.0)
    ]

    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)

    assert rows[0]["group_idx"] == 0
    assert rows[1]["group_idx"] == 0
    assert rows[0]["shared_with"] == [1]
    assert rows[1]["shared_with"] == [0]
    # contribution_pct is split evenly across the two sharers.
    assert rows[0]["contribution_pct"] == 10.0
    assert rows[0]["contribution_pct_share"] == 5.0
    assert rows[1]["contribution_pct_share"] == 5.0


def test_degenerate_sharing_splits_contribution_across_three():
    """contribution_pct_share divides evenly among however many DW groups share it."""
    res_2b = [
        _msrd_2b("Fe-N", reff=2.00),
        _msrd_2b("Fe-N", reff=2.05),
        _msrd_2b("Fe-N", reff=2.10),
    ]
    # Two FEFF paths; the middle MSRD group (2.05) is roughly equidistant but
    # nearer to the first path, so it matches that one instead of the second.
    path_groups = [
        _feff_path(nlegs=2, scatterer="N", r_eff=2.02, contribution_pct=9.0),
        _feff_path(nlegs=2, scatterer="N", r_eff=2.11, contribution_pct=3.0),
    ]

    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)

    matched = [r["group_idx"] for r in rows]
    assert matched == [0, 0, 1]
    assert rows[0]["contribution_pct_share"] == 4.5
    assert rows[1]["contribution_pct_share"] == 4.5
    assert rows[2]["contribution_pct_share"] == 3.0


def test_many_feff_paths_may_share_one_msrd_group():
    """Several FEFF paths are allowed to map to the same MSRD group.

    The exclusivity constraint is one-directional: a FEFF path has a single
    owner, but one MSRD group can own several FEFF paths.  The DW-centric row
    reports the nearest such FEFF path as its representative ``group_idx``.
    """
    res_2b = [_msrd_2b("Fe-N", reff=2.00)]
    path_groups = [
        _feff_path(nlegs=2, scatterer="N", r_eff=1.98),
        _feff_path(nlegs=2, scatterer="N", r_eff=2.01),
    ]

    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)

    # The nearer FEFF path (r_eff 2.01, |Δ|=0.01) represents the MSRD group.
    assert rows[0]["group_idx"] == 1
    assert rows[0]["feff_note"] == "direct ss"


# ---------------------------------------------------------------------------
# 3-body matching: scatterer-pair gating and angle-aware metric
# ---------------------------------------------------------------------------


def test_3body_scatterer_pair_must_match():
    """A 3-body FEFF path with the wrong element pair is never chosen.

    Regression test: previously ``_assign_feff_to_msrd`` did not check the
    FEFF path's scatterer pair at all for 3-body candidates, so an unrelated
    element pair at the right Reff could win purely on distance.
    """
    res_3b = [_msrd_3b("Fe-N-C", reff=3.50, angle=160.0)]
    path_groups = [
        _feff_path(nlegs=3, scatterer="O-O", r_eff=3.50, angle=160.0),  # wrong pair
        _feff_path(nlegs=3, scatterer="N-C", r_eff=3.55, angle=158.0),  # right pair
    ]

    rows = match_msrd_paths_to_feff(
        res_3b=res_3b, res_2b=[], path_groups=path_groups, r_tol=0.5
    )

    assert rows[0]["group_idx"] == 1


def test_3body_scatterer_pair_canonicalized():
    """A FEFF path labeled 'C-N' still matches an MSRD group typed 'N-C'.

    FEFF path atom order can vary frame-to-frame; the scatterer label must be
    canonicalized (sorted) before comparison.
    """
    res_3b = [_msrd_3b("Fe-N-C", reff=3.50, angle=160.0)]
    path_groups = [_feff_path(nlegs=3, scatterer="C-N", r_eff=3.51, angle=159.0)]

    rows = match_msrd_paths_to_feff(
        res_3b=res_3b, res_2b=[], path_groups=path_groups, r_tol=0.5
    )

    assert rows[0]["group_idx"] == 0


def test_3body_angle_distinguishes_same_reff_paths():
    """Two FEFF paths at the same Reff but different angles are distinguished.

    Without an angle-aware metric, the matcher can only see Reff and would
    have no basis to prefer the correct one.
    """
    res_3b = [_msrd_3b("Fe-N-C", reff=3.50, angle=175.0)]  # near-linear MS path
    path_groups = [
        _feff_path(nlegs=3, scatterer="N-C", r_eff=3.50, angle=90.0),  # bent, wrong
        _feff_path(
            nlegs=3, scatterer="N-C", r_eff=3.50, angle=178.0
        ),  # near-linear, right
    ]

    rows = match_msrd_paths_to_feff(
        res_3b=res_3b, res_2b=[], path_groups=path_groups, r_tol=0.5
    )

    assert rows[0]["group_idx"] == 1


def test_3body_falls_back_to_distance_only_without_feff_angle():
    """3-body matching still works (distance-only) when FEFF has no angle.

    Older averaged-paths HDF5 files predate angle extraction; matching must
    degrade gracefully rather than error or silently drop all candidates.
    """
    res_3b = [_msrd_3b("Fe-N-C", reff=3.50, angle=160.0)]
    path_groups = [_feff_path(nlegs=3, scatterer="N-C", r_eff=3.51, angle=None)]

    rows = match_msrd_paths_to_feff(
        res_3b=res_3b, res_2b=[], path_groups=path_groups, r_tol=0.5
    )

    assert rows[0]["group_idx"] == 0
    assert rows[0]["feff_note"] == "direct triangular"


def test_3body_dw_group_missing_angle_does_not_raise():
    """Defense-in-depth: a malformed DW 3-body group with no angle key must
    not crash the KD-tree query (schema normally guarantees res_3b always
    carries an angle, but this should degrade gracefully, not raise)."""
    res_3b = [{"type": "Fe-N-C", "reff": 3.50, "sigma2": 0.005, "count": 4}]
    # FEFF path also has no angle, so it lands in the 1-D (distance-only)
    # bucket that the angle-less DW row can still fall back to.
    path_groups = [
        _feff_path(nlegs=3, scatterer="N-C", r_eff=3.51, angle=None),
    ]

    rows = match_msrd_paths_to_feff(
        res_3b=res_3b, res_2b=[], path_groups=path_groups, r_tol=0.5
    )

    # Falls back to the 1-D (distance-only) bucket rather than raising.
    assert rows[0]["group_idx"] == 0


def test_3body_feff_angle_matched_against_both_scatterer_vertices():
    """A FEFF angle measured at the *other* scatterer vertex still matches.

    The FEFF-side angle is taken at the first-listed path vertex — an
    arbitrary one of the two scatterers — while the DW group's anchored
    ``angle`` is at the scatterer with the shorter absorber leg. For a
    mixed-element triangle these are different angles (here 35° vs 126°);
    matching must compare against both ``scatterer_angles_deg`` entries and
    keep the smaller distance, otherwise every mixed 3-body match fails.
    """
    res_3b = [
        {
            "type": "Mn-Mn-O",
            "reff": 5.64,
            "angle": 125.8,  # anchored (shorter-leg O vertex)
            "scatterer_angles_deg": (34.9, 125.8),
            "sigma2": 0.005,
            "count": 4,
        }
    ]
    path_groups = [
        # FEFF path with its angle at the Mn vertex (first-listed).
        _feff_path(nlegs=3, scatterer="Mn-O", r_eff=5.61, angle=35.5),
    ]

    rows = match_msrd_paths_to_feff(
        res_3b=res_3b, res_2b=[], path_groups=path_groups, r_tol=0.1
    )

    assert rows[0]["group_idx"] == 0
    # Distance uses the closer of the two DW scatterer angles (34.9 vs 35.5),
    # not the anchored one (125.8), so it must be well within tolerance.
    assert rows[0]["match_distance"] < 0.1


# ---------------------------------------------------------------------------
# Symmetric many-to-many candidate matching
# ---------------------------------------------------------------------------


def test_match_paths_within_tolerance_returns_full_candidate_lists():
    """Every FEFF path within tolerance is returned per DW group, sorted."""
    res_2b = [_msrd_2b("Fe-N", reff=2.00)]
    path_groups = [
        _feff_path(nlegs=2, scatterer="N", r_eff=2.01),  # nearest
        _feff_path(nlegs=2, scatterer="N", r_eff=1.95),  # also within tol
        _feff_path(nlegs=2, scatterer="N", r_eff=2.50),  # out of tol
    ]

    result = match_paths_within_tolerance(res_2b, None, path_groups, r_tol=0.1)

    cands = result["dw_to_feff"][("2b", 0)]
    assert [gi for _d, gi, _n, _nl in cands] == [0, 1]  # sorted by distance
    assert set(result["feff_to_dw"].keys()) == {0, 1}


def test_feff_matched_when_within_tol_but_not_nearest():
    """A FEFF path counts as matched if *any* DW group is within tolerance,
    even when that DW group's nearest match is a different FEFF path.

    This is the symmetric-coverage fix: previously only the single nearest
    FEFF path per DW group ever showed as matched, badly undercounting FEFF
    coverage when several FEFF paths are near one DW group.
    """
    res_2b = [_msrd_2b("Fe-N", reff=2.00)]
    path_groups = [
        _feff_path(nlegs=2, scatterer="N", r_eff=2.01),
        _feff_path(nlegs=2, scatterer="N", r_eff=1.99),
    ]

    diag = extract_path_matching_diagnostics(res_2b, None, path_groups, r_tol=0.1)

    assert diag["summary"]["n_feff_matched"] == 2
    feff = [r for r in diag["records"] if r["role"] == "FEFF"]
    assert all(r["status"] == "matched" for r in feff)


def test_rows_expose_all_candidates_nearest_first():
    """match_msrd_paths_to_feff rows list every candidate FEFF path within
    tolerance (nearest first), with group_idx kept as the nearest one."""
    res_2b = [_msrd_2b("Fe-N", reff=2.00)]
    path_groups = [
        _feff_path(nlegs=2, scatterer="N", r_eff=2.03),  # 2nd nearest
        _feff_path(nlegs=2, scatterer="N", r_eff=1.98),  # nearest
    ]

    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)

    assert rows[0]["group_idx"] == 1
    assert rows[0]["candidate_group_idxs"] == [1, 0]
    assert rows[0]["n_candidates"] == 2


# ---------------------------------------------------------------------------
# Diagnostics extractor
# ---------------------------------------------------------------------------


def test_diagnostics_merging_only_without_feff():
    """With no FEFF paths the extractor yields merging (DW-only) records."""
    res_2b = [
        _msrd_2b("Fe-N", reff=2.00, sigma2=0.004, count=6),
        _msrd_2b("Fe-C", reff=3.00, sigma2=0.009, count=2),
    ]

    diag = extract_path_matching_diagnostics(res_2b, None, None, r_tol=0.1)

    recs = diag["records"]
    assert len(recs) == 2
    assert {r["role"] for r in recs} == {"DW"}
    first = recs[0]
    assert first["reff_axis"] == 2.00
    assert first["count"] == 6
    assert abs(first["sigma"] - 0.004**0.5) < 1e-12
    # No FEFF supplied → everything unmatched, no sharing.
    assert diag["summary"]["n_feff"] == 0
    assert diag["summary"]["n_feff_shared"] == 0
    assert all(r["matched"] is False for r in recs)


def test_diagnostics_merging_only_reports_3body_angle_spread():
    """Merge-only diagnostics surface angle_std for 3-body pooling quality."""
    res_3b = [_msrd_3b("Fe-N-C", reff=3.50, angle=160.0, angle_var=9.0)]

    diag = extract_path_matching_diagnostics([], res_3b, None, r_tol=0.1)

    rec = diag["records"][0]
    assert rec["angle"] == 160.0
    assert rec["angle_std"] == 3.0  # sqrt(9.0)


def test_diagnostics_2body_has_no_angle_std():
    res_2b = [_msrd_2b("Fe-N", reff=2.00)]

    diag = extract_path_matching_diagnostics(res_2b, None, None, r_tol=0.1)

    assert diag["records"][0]["angle_std"] is None


def test_diagnostics_matching_records_and_summary():
    """With FEFF paths the extractor annotates match status on both roles."""
    res_2b = [
        _msrd_2b("Fe-N", reff=2.00),
        _msrd_2b("Fe-N", reff=2.05),  # shares the same FEFF path as the group above
    ]
    path_groups = [
        _feff_path(nlegs=2, scatterer="N", r_eff=2.02),  # matches both Fe-N groups
        _feff_path(nlegs=2, scatterer="O", r_eff=5.00),  # no eligible DW group
    ]

    diag = extract_path_matching_diagnostics(res_2b, None, path_groups, r_tol=0.1)

    dw = [r for r in diag["records"] if r["role"] == "DW"]
    feff = [r for r in diag["records"] if r["role"] == "FEFF"]

    # DW: both groups match the same nearby FEFF path (degenerate sharing).
    assert dw[0]["status"] == "matched"
    assert dw[1]["status"] == "matched"
    assert dw[0]["partner_label"] == "path#0"
    assert dw[1]["partner_label"] == "path#0"

    # FEFF: rattle-free 2-leg path axis == raw; the O path is unmatched;
    # path#0 is shared by both DW groups.
    matched_feff = [r for r in feff if r["status"] == "matched"]
    assert len(matched_feff) == 1
    assert matched_feff[0]["idx"] == 0
    assert matched_feff[0]["n_dw_matches"] == 2

    summary = diag["summary"]
    assert summary == {
        "n_dw": 2,
        "n_dw_matched": 2,
        "n_dw_unmatched": 0,
        "n_feff": 2,
        "n_feff_matched": 1,
        "n_feff_unmatched": 1,
        "n_feff_shared": 1,
    }


def test_diagnostics_rattle_axis_is_halved():
    """A 4-leg rattle FEFF path is placed at r_eff/2 on the shared axis."""
    res_2b = [_msrd_2b("Fe-N", reff=2.00)]
    path_groups = [_feff_path(nlegs=4, scatterer="N", r_eff=4.00)]

    diag = extract_path_matching_diagnostics(res_2b, None, path_groups, r_tol=0.1)

    feff = next(r for r in diag["records"] if r["role"] == "FEFF")
    assert feff["reff_raw"] == 4.00
    assert feff["reff_axis"] == 2.00
    assert feff["status"] == "matched"
    assert diag["summary"]["n_feff_shared"] == 0

    # The DW distance is measured on the shared axis (r_eff/2), so the rattle
    # is a perfect match (0.0), not |4.00 - 2.00|.
    dw = next(r for r in diag["records"] if r["role"] == "DW")
    assert dw["status"] == "matched"
    assert abs(dw["distance"]) < 1e-12
    assert dw["partner_reff_axis"] == 2.00


def test_rattle_pipeline_label_includes_absorber_element():
    """A 4-leg rattle labeled with all intermediates ("Mn-O-O") still matches.

    The FEFF pipeline labels 4-leg paths with every intermediate element,
    sorted — including the revisited absorber. Without deriving the rattle
    scatterer from the token multiset, the canonical-key gate compares
    ("Mn","O","O") against the DW group's ("O",) and the rattle can never
    match (it shows up as a spurious "no partner" at r_eff/2).
    """
    res_2b = [_msrd_2b("Mn-O", reff=2.10)]
    path_groups = [
        {**_feff_path(nlegs=4, scatterer="Mn-O-O", r_eff=4.19), "absorber": "Mn"},
    ]

    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)

    assert rows[0]["group_idx"] == 0
    assert rows[0]["feff_note"] == "rattle (4×σ²)"
    assert rows[0]["sigma2_feff"] == 4.0 * rows[0]["sigma2"]


def test_rattle_label_with_wrong_absorber_token_is_rejected():
    """4-leg labels whose singleton token is not the absorber are not rattles.

    "Mn-Mn-O" on an Mn absorber is A→Mn→O→Mn→A (double scattering through
    three distinct atoms), not a collinear rattle — it must never match a
    2-body DW group. Likewise "O-O-O" cannot be a rattle on an Mn absorber.
    """
    res_2b = [_msrd_2b("Mn-O", reff=2.10), _msrd_2b("Mn-Mn", reff=2.56)]
    path_groups = [
        {**_feff_path(nlegs=4, scatterer="Mn-Mn-O", r_eff=4.19), "absorber": "Mn"},
        {**_feff_path(nlegs=4, scatterer="O-O-O", r_eff=4.19), "absorber": "Mn"},
        {**_feff_path(nlegs=4, scatterer="Mn-Mn-Mn", r_eff=5.10), "absorber": "Mn"},
    ]

    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)

    assert rows[0]["group_idx"] is None  # no rattle on the Mn-O shell
    # {Mn:3} is a rattle on the same-element Mn-Mn pair: matches at 5.10/2.
    assert rows[1]["group_idx"] == 2
    assert rows[1]["feff_note"] == "rattle (4×σ²)"


def test_direct_ss_preferred_over_nearer_rattle_representative():
    """The representative match is the direct SS path, not a nearer rattle.

    A 4-leg rattle can sit slightly closer to the DW shell on the matching
    axis (r_eff/2) than the direct single-scattering path. The SS path must
    still *represent* the group — the rattle σ² = 4σ² relation is only an
    approximation — while both remain in the candidate list (many-to-many).
    """
    res_2b = [_msrd_2b("Mn-O", reff=2.146)]
    path_groups = [
        {**_feff_path(nlegs=4, scatterer="Mn-O-O", r_eff=4.38), "absorber": "Mn"},
        _feff_path(nlegs=2, scatterer="O", r_eff=2.077),
    ]
    # rattle axis: 4.38/2 = 2.190 (d=0.044); direct ss: 2.077 (d=0.069).
    # The rattle is nearer but must not become the representative.

    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)

    assert rows[0]["group_idx"] == 1  # direct SS, not the nearer rattle
    assert rows[0]["feff_note"] == "direct ss"
    assert rows[0]["candidate_group_idxs"] == [0, 1]  # both still candidates
    assert rows[0]["sigma2_feff"] == rows[0]["sigma2"]  # no 4x for SS


def test_rattle_is_representative_when_no_ss_path_in_tolerance():
    """A rattle represents the group only when no 2-leg path is available."""
    res_2b = [_msrd_2b("Mn-O", reff=2.146)]
    path_groups = [
        {**_feff_path(nlegs=4, scatterer="Mn-O-O", r_eff=4.38), "absorber": "Mn"},
        _feff_path(nlegs=2, scatterer="O", r_eff=3.50),  # out of tolerance
    ]

    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)

    assert rows[0]["group_idx"] == 0
    assert rows[0]["feff_note"] == "rattle (4×σ²)"
    assert rows[0]["sigma2_feff"] == 4.0 * rows[0]["sigma2"]


# ---------------------------------------------------------------------------
# Pooling DW groups that share one FEFF path (for static-structure fits)
# ---------------------------------------------------------------------------


def test_pool_single_group_is_a_no_op():
    """A FEFF path matched by only one DW group passes through unchanged."""
    res_2b = [_msrd_2b("Fe-N", reff=2.00, sigma2=0.006, count=6)]
    path_groups = [_feff_path(nlegs=2, scatterer="N", r_eff=2.01)]
    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)

    pooled = pool_dw_groups_by_feff_path(rows, path_groups)

    assert len(pooled) == 1
    p = pooled[0]
    assert p["n_dw_groups"] == 1
    assert p["reff_combined"] == pytest.approx(2.00)
    assert p["sigma2_combined"] == pytest.approx(0.006)
    assert p["between_group_variance_A2"] == pytest.approx(0.0)
    assert p["count_total"] == 6


def test_pool_combines_two_groups_via_law_of_total_variance():
    """Two DW groups sharing one FEFF path are combined with the correct
    (verified against a direct pooled-sample calculation) formula: the
    combined variance is the weighted mean of the within-group variances
    *plus* the weighted variance of the group means (the spread a naive
    average of sigma2 would silently discard).
    """
    # Mirrors the real Mn-Mn-O case: two distinct triangle populations with
    # nearly the same Reff, both matching one FEFF path, with a genuinely
    # different angle (so they must NOT be merged at the grouping stage,
    # but the resulting sigma2/Reff still need to be combined for a
    # single-structure fit).
    res_3b = [
        {
            **_msrd_3b("Mn-Mn-O", reff=4.4723, angle=55.0, sigma2=0.012606, count=24),
            "scatterer_angles_deg": (35.03, 55.01),
        },
        {
            **_msrd_3b("Mn-Mn-O", reff=4.5026, angle=90.3, sigma2=0.012501, count=24),
            "scatterer_angles_deg": (54.39, 90.3),
        },
    ]
    path_groups = [_feff_path(nlegs=3, scatterer="Mn-O", r_eff=4.6071, angle=53.17)]
    rows = match_msrd_paths_to_feff([], res_3b, path_groups, r_tol=0.5)
    assert {r["group_idx"] for r in rows} == {0}  # both share the one FEFF path

    pooled = pool_dw_groups_by_feff_path(rows, path_groups)

    assert len(pooled) == 1
    p = pooled[0]
    assert p["n_dw_groups"] == 2
    assert p["count_total"] == 48

    # Hand-computed law-of-total-variance reference (equal weights, since
    # both groups have count=24).
    w1 = w2 = 24.0
    m1, m2 = 4.4723, 4.5026
    v1, v2 = 0.012606, 0.012501
    expected_mean = (w1 * m1 + w2 * m2) / (w1 + w2)
    expected_between = (
        w1 * (m1 - expected_mean) ** 2 + w2 * (m2 - expected_mean) ** 2
    ) / (w1 + w2)
    expected_var = (w1 * v1 + w2 * v2) / (w1 + w2) + expected_between

    assert p["reff_combined"] == pytest.approx(expected_mean)
    assert p["between_group_variance_A2"] == pytest.approx(expected_between)
    assert p["sigma2_combined"] == pytest.approx(expected_var)
    # The combined sigma2 must exceed a naive (between-group-blind) average,
    # since the two populations' means genuinely differ.
    naive_avg = (w1 * v1 + w2 * v2) / (w1 + w2)
    assert p["sigma2_combined"] > naive_avg


def test_pool_rescales_rattle_matches_to_feff_length_scale():
    """A rattle match is rescaled (Reff x2, sigma2_feff = 4*sigma2) before
    being combined with a direct match sharing the same FEFF path, so the
    two contribute on a consistent length/variance scale.
    """
    res_2b = [_msrd_2b("Mn-O", reff=2.10, sigma2=0.010, count=6)]
    path_groups = [
        {**_feff_path(nlegs=4, scatterer="Mn-O-O", r_eff=4.22), "absorber": "Mn"},
    ]
    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)
    assert rows[0]["feff_note"] == "rattle (4×σ²)"

    pooled = pool_dw_groups_by_feff_path(rows, path_groups)

    assert len(pooled) == 1
    p = pooled[0]
    assert p["n_dw_groups"] == 1
    assert p["reff_combined"] == pytest.approx(2.10 * 2.0)
    assert p["sigma2_combined"] == pytest.approx(4.0 * 0.010)


def test_pool_ignores_unmatched_dw_groups():
    """DW groups with no FEFF match (group_idx None) are simply excluded."""
    res_2b = [_msrd_2b("Fe-N", reff=2.00)]
    path_groups = [_feff_path(nlegs=2, scatterer="N", r_eff=5.00)]  # out of tol
    rows = match_msrd_paths_to_feff(res_2b, None, path_groups, r_tol=0.1)

    pooled = pool_dw_groups_by_feff_path(rows, path_groups)

    assert pooled == []
