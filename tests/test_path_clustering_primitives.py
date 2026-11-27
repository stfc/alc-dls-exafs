"""Tests for the shared path clustering/canonicalization primitives.

These are used by MD path pooling (``calculate_grouped_msrd``), FEFF path
pooling (``PathAggregator``), and MSRD<->FEFF matching
(``match_msrd_paths_to_feff``) — see
:mod:`larch_cli_wrapper.debye_waller_core`.
"""

from __future__ import annotations

import numpy as np

from larch_cli_wrapper.debye_waller_core import (
    canonical_scatterer_key,
    cluster_1d_sorted,
    estimate_tolerance_from_elbow,
    path_feature_vector,
)


class TestCanonicalScattererKey:
    def test_single_element(self):
        assert canonical_scatterer_key("O") == ("O",)

    def test_sorts_multi_element_labels(self):
        assert canonical_scatterer_key("N-C") == ("C", "N")
        assert canonical_scatterer_key("C-N") == ("C", "N")

    def test_order_independent_equality(self):
        assert canonical_scatterer_key("Cl-K") == canonical_scatterer_key("K-Cl")


class TestPathFeatureVector:
    def test_distance_only_when_no_angle(self):
        vec = path_feature_vector(3.5)
        np.testing.assert_allclose(vec, [3.5])

    def test_distance_and_arc_length_with_angle(self):
        vec = path_feature_vector(3.5, 180.0)
        # 180 degrees -> pi radians -> arc length = r * pi
        np.testing.assert_allclose(vec, [3.5, 3.5 * np.pi])

    def test_zero_angle_gives_zero_arc_length(self):
        vec = path_feature_vector(2.0, 0.0)
        np.testing.assert_allclose(vec, [2.0, 0.0])


class TestCluster1dSorted:
    def test_empty_input(self):
        assert cluster_1d_sorted(np.array([]), 0.1) == []

    def test_single_value(self):
        clusters = cluster_1d_sorted(np.array([2.0]), 0.1)
        assert len(clusters) == 1
        np.testing.assert_array_equal(clusters[0], [0])

    def test_two_well_separated_clusters(self):
        values = np.array([2.0, 2.01, 2.5, 2.51])
        clusters = cluster_1d_sorted(values, 0.1)
        assert len(clusters) == 2
        np.testing.assert_array_equal(sorted(clusters[0]), [0, 1])
        np.testing.assert_array_equal(sorted(clusters[1]), [2, 3])

    def test_unsorted_input_still_clusters_correctly(self):
        values = np.array([2.51, 2.0, 2.01, 2.5])
        clusters = cluster_1d_sorted(values, 0.1)
        assert len(clusters) == 2
        # indices 1 (2.0) and 2 (2.01) belong to the low cluster
        low_cluster = next(c for c in clusters if 1 in c)
        np.testing.assert_array_equal(sorted(low_cluster), [1, 2])

    def test_running_mean_stops_a_slowly_drifting_chain(self):
        # A new cluster starts as soon as a point strays from the *running
        # mean* of the current cluster by more than tol, even if each
        # consecutive pair of points is individually close together.
        values = np.array([0.0, 0.08, 0.16, 0.24])
        clusters = cluster_1d_sorted(values, 0.1)
        # 0.0 & 0.08 join (running mean 0.04); 0.16 is 0.12 from that mean
        # (> tol) so it starts a new cluster with 0.24.
        assert len(clusters) == 2
        np.testing.assert_array_equal(sorted(clusters[0]), [0, 1])
        np.testing.assert_array_equal(sorted(clusters[1]), [2, 3])

    def test_single_outlier_starts_new_cluster(self):
        values = np.array([2.0, 2.01, 2.02, 5.0])
        clusters = cluster_1d_sorted(values, 0.1)
        assert len(clusters) == 2
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [1, 3]


class TestEstimateToleranceFromElbow:
    def test_too_few_points_returns_none(self):
        result = estimate_tolerance_from_elbow(np.array([0.1]))
        assert result["tol"] is None

    def test_finds_knee_between_tight_and_spread_clusters(self):
        rng = np.random.default_rng(42)
        tight = rng.uniform(0.0, 0.02, size=30)
        spread = rng.uniform(0.3, 1.0, size=10)
        distances = np.concatenate([tight, spread])

        result = estimate_tolerance_from_elbow(distances, min_tol=0.005, max_tol=0.5)

        # The knee should fall somewhere between the tight cluster's max and
        # the spread cluster's min.
        assert 0.015 <= result["tol"] <= 0.3
        assert result["knee_index"] is not None
        assert len(result["sorted_distances"]) == 40

    def test_respects_min_and_max_clamp(self):
        distances = np.array([0.5, 0.5, 0.5, 0.5])
        result = estimate_tolerance_from_elbow(distances, min_tol=0.6, max_tol=1.0)
        assert result["tol"] == 0.6

    def test_ignores_non_finite_distances(self):
        distances = np.array([0.01, 0.02, np.nan, np.inf, 0.5])
        result = estimate_tolerance_from_elbow(distances)
        assert len(result["sorted_distances"]) == 3


def test_cluster_1d_sorted_valley_keeps_broad_unimodal_shell_whole():
    rng = np.random.default_rng(0)
    shell = rng.normal(3.16, 0.15, 5000)
    assert len(cluster_1d_sorted(shell, 0.15)) > 1
    assert len(cluster_1d_sorted(shell, 0.15, "valley")) == 1


def test_cluster_1d_sorted_valley_splits_separated_shells():
    rng = np.random.default_rng(1)
    shells = np.concatenate([rng.normal(3.1, 0.15, 3000), rng.normal(5.4, 0.15, 3000)])
    clusters = cluster_1d_sorted(shells, 0.15, "valley")
    assert len(clusters) == 2
    assert sorted(len(c) for c in clusters) == [3000, 3000]


def test_cluster_1d_sorted_rejects_unknown_mode():
    import pytest

    with pytest.raises(ValueError, match="cluster mode"):
        cluster_1d_sorted(np.array([1.0, 2.0]), 0.1, "gap")


def test_cluster_1d_sorted_valley_handles_small_n_msrd_and_tail_ripples():
    # Small-N MSRD (e.g. single absorber: 6 1NN + 8 3NN = 14 instances) must split.
    small = np.concatenate([np.full(6, 3.16), np.full(8, 5.45)])
    clusters = cluster_1d_sorted(small, 0.10, "valley")
    assert len(clusters) == 2
    assert sorted(len(c) for c in clusters) == [6, 8]

    # Small noise bump in the tail between two tall shells must not cause a 3rd cut.
    rng = np.random.default_rng(42)
    with_ripple = np.concatenate(
        [
            rng.normal(3.16, 0.14, 4000),
            rng.normal(3.75, 0.05, 8),  # tiny tail ripple (<0.5% peak height)
            rng.normal(5.45, 0.15, 4000),
        ]
    )
    clusters_r = cluster_1d_sorted(with_ripple, 0.15, "valley")
    assert len(clusters_r) == 2

    # Angle-scale clustering (bandwidth=15 deg) stays fast and separates 0/90/180 deg.
    angles = np.concatenate(
        [
            rng.normal(5.0, 3.0, 500),
            rng.normal(90.0, 4.0, 500),
            rng.normal(175.0, 3.0, 500),
        ]
    )
    ang_clusters = cluster_1d_sorted(angles, 15.0, "valley")
    assert len(ang_clusters) == 3
