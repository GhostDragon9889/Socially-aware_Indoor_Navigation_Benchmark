"""Meaningful CPU checks for the isolated 45D reference-only diagnostic."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("pose_only_candidate", Path(__file__).with_name("evaluate_pose_diversity.py"))
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def test_known_assignment_entropy_distance_and_inclusive_outlier():
    centers = np.zeros((2, 45))
    centers[1, 0] = 4
    balanced = np.repeat(centers, 2, axis=0)
    row = candidate.cluster_statistics(balanced, centers)
    assert row["coverage_occupied_over_K"] == 1 and row["normalized_entropy"] == 1
    assert row["nearest_center_distance"]["max"] == 0
    stationary = candidate.cluster_statistics(np.repeat(centers[:1], 4, axis=0), centers)
    assert stationary["coverage_occupied_over_K"] == .5 and stationary["normalized_entropy"] == 0
    outlier = np.zeros((1, 45))
    outlier[0, 0] = 100
    failed_pose = candidate.cluster_statistics(np.concatenate((balanced, outlier)), centers)
    assert failed_pose["samples_assigned"] == 5 and failed_pose["assignment_counts"] == [2, 3]
    assert failed_pose["nearest_center_distance"]["max"] == 96


def test_reference_only_basis_reproducible_and_unaffected_by_generated_distribution():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(120, 45))
    centers, provenance = candidate.fit_reference(reference, clusters=(2, 3), cap=80)
    frozen = {k: value.copy() for k, value in centers.items()}
    near = candidate.compare_pose_only(reference, reference, centers, cap=64)
    far = candidate.compare_pose_only(reference+50, reference, centers, cap=64)
    assert near["pose_only_biased_gaussian_MMD2"]["value"] == 0
    assert far["pose_only_biased_gaussian_MMD2"]["value"] > 0
    assert all(np.array_equal(centers[k], frozen[k]) for k in centers)
    repeated, second = candidate.fit_reference(reference, clusters=(2, 3), cap=80)
    assert provenance == second and all(np.array_equal(repeated[k], centers[k]) for k in centers)
    assert provenance["fit_reference_only"] and not provenance["generated_samples_used_for_fit"]
    assert provenance["fit_sample_count"] == 80
    for k in (2, 3):
        restarts = provenance["restarts"][str(k)]
        best = restarts["restarts"][restarts["selected_restart_zero_based"]]["inertia_sum_squared_distance"]
        assert best == min(row["inertia_sum_squared_distance"] for row in restarts["restarts"])


def test_biased_pose_mmd_symmetry_and_explicit_cap():
    rng = np.random.default_rng(18)
    left, right = rng.normal(size=(40, 45)), rng.normal(size=(40, 45))
    centers = {2: left[:2]}
    lr = candidate.compare_pose_only(left, right, centers, cap=32)
    rl = candidate.compare_pose_only(right, left, centers, cap=32)
    assert lr["pose_only_biased_gaussian_MMD2"]["value"] == pytest.approx(rl["pose_only_biased_gaussian_MMD2"]["value"], abs=1e-15)
    assert lr["pose_only_biased_gaussian_MMD2"]["dimensions"] == 45
    assert lr["pose_only_biased_gaussian_MMD2"]["generated_samples"] == 32
    assert lr["clusters"]["2"]["samples_assigned"] == 40


def test_reference_fit_recovers_two_known_group_means():
    reference = np.zeros((20, 45))
    reference[:10, 0] = np.linspace(-2.1, -1.9, 10)
    reference[10:, 0] = np.linspace(1.9, 2.1, 10)
    centers, metadata = candidate.fit_reference(reference, clusters=(2,), n_init=3)
    assert np.sort(centers[2][:, 0]) == pytest.approx([-2, 2], abs=1e-14)
    assert np.array_equal(centers[2][:, 1:], np.zeros((2, 44)))
    assert metadata["restarts"]["2"]["selected_restart_zero_based"] in range(3)


def test_duplicate_reference_poses_kplusplus_does_not_pad_or_collapse_centers():
    unique = np.zeros((3, 45))
    unique[:, 0] = [0, 2, 10]
    repeated = np.repeat(unique, [30, 10, 20], axis=0)
    centers, _ = candidate.fit_reference(repeated, clusters=(3,), n_init=3)
    assert np.sort(centers[3][:, 0]) == pytest.approx([0, 2, 10], abs=1e-14)
    assert candidate.cluster_statistics(repeated, centers[3])["occupied_clusters"] == 3


def test_empty_reference_cluster_relocated_to_real_farthest_pose():
    sample = np.zeros((3, 45))
    sample[:, 0] = [0, 2, 10]
    duplicate_centers = sample[[0, 0, 2]]
    distance2 = candidate._distance2(sample, duplicate_centers)
    labels = distance2.argmin(-1)
    assert np.bincount(labels, minlength=3).tolist() == [2, 0, 1]
    updated = candidate._centroid_update(sample, labels, distance2, 3)
    assert updated[:, 0] == pytest.approx([1, 2, 10], abs=0)
    assert np.array_equal(updated[1], sample[1])
    assert np.unique(candidate._distance2(sample, updated).argmin(-1)).size == 3


def test_equidistant_cluster_tie_uses_first_fixed_reference_center():
    centers = np.zeros((2, 45))
    centers[:, 0] = [-1, 1]
    row = candidate.cluster_statistics(np.zeros((2, 45)), centers)
    assert row["assignment_counts"] == [2, 0]
    assert row["nearest_center_distance"]["mean"] == 1


@pytest.mark.parametrize("bad", [np.zeros((1, 44)), np.zeros((0, 45)), np.full((1, 45), np.nan), np.zeros((1, 45), dtype=bool)])
def test_invalid_pose_cannot_be_dropped_or_reinterpreted(bad):
    with pytest.raises(ValueError):
        candidate.cluster_statistics(bad, np.zeros((2, 45)))


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"n_init": True}, {"clusters": (2, 2)}, {"cap": True}, {"clusters": (True,)}])
def test_invalid_fit_configuration_rejects(kwargs):
    with pytest.raises(ValueError):
        candidate.fit_reference(np.random.default_rng(7).normal(size=(110, 45)), **kwargs)


def test_insufficient_unique_reference_cannot_fake_high_coverage():
    with pytest.raises(ValueError, match="distinct"):
        candidate.fit_reference(np.zeros((110, 45)))


def test_capped_indices_exact_endpoints_no_replacement():
    result = candidate.capped_indices(5000, 2048)
    assert len(np.unique(result)) == 2048 and result[0] == 0 and result[-1] == 4999
    assert np.array_equal(candidate.capped_indices(3), np.arange(3))
