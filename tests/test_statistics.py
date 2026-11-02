"""Tests for paired DeLong test, non-inferiority hypothesis testing, bootstrap CIs, and McNemar."""

import numpy as np
import pytest

from ovaryvision.statistics import (
    bootstrap_paired_difference,
    compute_roc_auc,
    evaluate_non_inferiority,
    generate_comprehensive_statistical_report,
    mcnemar_test_paired,
    paired_cluster_bootstrap_segmentation,
    paired_delong_test,
)


def test_compute_roc_auc():
    """Verify AUC calculation on known edge cases."""
    # Perfect ranking
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert compute_roc_auc(scores, labels) == 1.0

    # Inverted ranking
    inv_scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert compute_roc_auc(inv_scores, labels) == 0.0

    # With ties
    tied_scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert compute_roc_auc(tied_scores, labels) == 0.5


def test_paired_delong_identical_predictions():
    """Verify DeLong test on identical model predictions produces zero difference and p=1.0."""
    scores = np.array([0.1, 0.3, 0.4, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])

    res = paired_delong_test(scores, scores, labels)
    assert res.auc_difference == 0.0
    assert res.p_value_two_tailed == 1.0


def test_paired_delong_distinct_models():
    """Verify DeLong computes positive SE and valid p-value for distinct models."""
    rng = np.random.default_rng(42)
    n = 200
    labels = rng.integers(0, 2, size=n)
    s1 = np.clip(labels * 0.6 + rng.normal(0, 0.3, size=n), 0, 1)
    s2 = np.clip(labels * 0.4 + rng.normal(0, 0.3, size=n), 0, 1)

    res = paired_delong_test(s1, s2, labels)
    assert res.auc_model_1 > res.auc_model_2
    assert res.se_difference > 0.0
    assert 0.0 <= res.p_value_two_tailed <= 1.0


def test_non_inferiority_test_decisions():
    """Verify non-inferiority is declared when test model matches or exceeds reference, and rejected when inferior."""
    rng = np.random.default_rng(101)
    n = 300
    labels = rng.integers(0, 2, size=n)

    # Model B baseline
    s_b = np.clip(labels * 0.5 + rng.normal(0, 0.2, size=n), 0, 1)
    # Model C matches or slightly outperforms Model B (diff >= 0 > -0.02)
    s_c = np.clip(s_b + 0.05 * labels, 0, 1)

    ni_equal = evaluate_non_inferiority(s_c, s_b, labels, margin_delta=-0.02)
    assert ni_equal.observed_auc_diff >= 0.0
    assert ni_equal.ci_lower > -0.02
    assert ni_equal.p_value < 0.01
    assert ni_equal.non_inferiority_declared is True

    # Model C is severely degraded (inferior)
    s_c_bad = np.clip(rng.uniform(0, 1, size=n), 0, 1)  # random guessing
    ni_bad = evaluate_non_inferiority(s_c_bad, s_b, labels, margin_delta=-0.02)
    assert ni_bad.non_inferiority_declared is False
    assert ni_bad.p_value > 0.5


def test_bootstrap_paired_difference():
    """Verify stratified paired bootstrap confidence interval generation."""
    rng = np.random.default_rng(42)
    n = 100
    labels = np.array([0] * 50 + [1] * 50)
    s1 = np.clip(labels * 0.6 + rng.normal(0, 0.2, size=n), 0, 1)
    s2 = np.clip(labels * 0.5 + rng.normal(0, 0.2, size=n), 0, 1)

    boot = bootstrap_paired_difference(s1, s2, labels, metric_fn="auc", num_resamples=200, seed=42)
    assert boot.ci_lower_95 <= boot.observed_difference <= boot.ci_upper_95
    assert boot.num_resamples == 200


def test_mcnemar_test_paired():
    """Verify McNemar's paired discordance test on known 2x2 contingency matrix."""
    # 10 samples:
    # 6 both correct, 2 both wrong, 2 Model 1 only correct, 0 Model 2 only correct
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    s1 = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1])
    s2 = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.9])

    res = mcnemar_test_paired(s1, s2, labels, threshold=0.5)
    assert res.n_concordant_correct >= 6
    assert 0.0 <= res.p_value <= 1.0


def test_comprehensive_statistical_report():
    """Verify end-to-end report generation with primary and sensitivity non-inferiority margins."""
    rng = np.random.default_rng(42)
    n = 150
    labels = rng.integers(0, 2, size=n)
    s_c = np.clip(labels * 0.5 + rng.normal(0, 0.2, size=n), 0, 1)
    s_b = np.clip(labels * 0.52 + rng.normal(0, 0.2, size=n), 0, 1)

    report = generate_comprehensive_statistical_report(
        scores_arm_c=s_c,
        scores_arm_b=s_b,
        labels=labels,
        primary_delta=-0.02,
        sensitivity_deltas=[-0.01, -0.02, -0.03],
        num_bootstrap=200,
        seed=42,
    )

    assert report.sample_size == 150
    assert len(report.non_inferiority_sensitivities) == 3
    assert report.bootstrap_auc.metric_name == "auc"
    assert report.bootstrap_brier.metric_name == "brier"
    assert report.mcnemar_test.threshold_used == 0.5


def test_paired_cluster_bootstrap_multiplicity_preservation():
    """Verify that cluster bootstrap strictly preserves multiplicity and does NOT collapse via set union.

    If cluster A (diff = +1.0, 1 scan) and cluster B (diff = 0.0, 1 scan) are sampled such that
    cluster A appears twice and cluster B once (multiplicity [2, 1]), the pooled difference
    MUST be (2*1.0 + 1*0.0) / (2+1) = 2/3 = 0.6667.
    If duplicate clusters were collapsed via set union {A, B}, the result would incorrectly be (1+0)/2 = 0.5.
    """
    scores_c = np.array([1.0, 0.0])
    scores_b = np.array([0.0, 0.0])
    cluster_ids = np.array(["cluster_A", "cluster_B"])

    # Run cluster bootstrap
    res = paired_cluster_bootstrap_segmentation(
        scores_c=scores_c,
        scores_b=scores_b,
        cluster_ids=cluster_ids,
        metric_name="dsc",
        num_resamples=1000,
        seed=42,
    )

    assert res.num_clusters == 2
    assert res.num_samples == 2
    assert np.isclose(res.observed_difference, 0.5)

    # In a 2-cluster bootstrap, possible draws of (k1, k2) with replacement are:
    # (A, A) -> mean = 1.0  (prob = 1/4)
    # (A, B) -> mean = 0.5  (prob = 1/4)
    # (B, A) -> mean = 0.5  (prob = 1/4)
    # (B, B) -> mean = 0.0  (prob = 1/4)
    # Expected bootstrap mean = 0.25*1.0 + 0.50*0.5 + 0.25*0.0 = 0.500
    assert np.isclose(res.bootstrap_mean, 0.5, atol=0.05)

    # Now verify with unequal cluster sizes where set union collapse would fail dramatically:
    # Cluster A has 2 scans, diff = +1.0 each (sum = +2.0, count = 2)
    # Cluster B has 1 scan, diff = 0.0 (sum = 0.0, count = 1)
    scores_c_unequal = np.array([1.0, 1.0, 0.0])
    scores_b_unequal = np.array([0.0, 0.0, 0.0])
    cluster_ids_unequal = np.array(["cluster_A", "cluster_A", "cluster_B"])

    res_unequal = paired_cluster_bootstrap_segmentation(
        scores_c=scores_c_unequal,
        scores_b=scores_b_unequal,
        cluster_ids=cluster_ids_unequal,
        metric_name="dsc",
        num_resamples=2000,
        seed=1337,
    )

    # If (A, B) is drawn: (2*1.0 + 1*0.0) / (2+1) = 2/3 = 0.6667
    # If (A, A) is drawn: (4*1.0) / 4 = 1.0
    # If (B, B) is drawn: (2*0.0) / 2 = 0.0
    # Expected mean = 0.25*1.0 + 0.50*(2/3) + 0.25*0.0 = 0.25 + 0.3333 = 0.5833
    assert np.isclose(res_unequal.bootstrap_mean, 0.5833, atol=0.03)


def test_paired_cluster_bootstrap_with_valid_mask_for_lpr():
    """Verify that LPR evaluation correctly applies valid_mask (overlap cohort only)."""
    scores_c = np.array([0.9, 0.8, 0.0, 0.5])
    scores_b = np.array([0.7, 0.8, 0.0, 0.2])
    cluster_ids = np.array(["c1", "c1", "c2", "c3"])
    valid_mask = np.array([True, True, False, True]) # c2 has zero overlap, excluded!

    res = paired_cluster_bootstrap_segmentation(
        scores_c=scores_c,
        scores_b=scores_b,
        cluster_ids=cluster_ids,
        valid_mask=valid_mask,
        metric_name="lpr",
        num_resamples=500,
        seed=42,
    )

    assert res.num_samples == 3
    assert res.num_clusters == 3
    # Valid differences: (0.9-0.7)=0.2, (0.8-0.8)=0.0, (0.5-0.2)=0.3
    # Observed mean = (0.2 + 0.0 + 0.3) / 3 = 0.1667
    assert np.isclose(res.observed_difference, 0.1666667)
    assert res.ci_lower_95 <= res.observed_difference <= res.ci_upper_95


def test_paired_cluster_bootstrap_empty_draw_conditional_rejection():
    """Verify that iterations with zero valid overlap observations are redrawn conditionally, never imputed.

    Setup: 4 clusters, where 3 clusters have ZERO valid overlap scans, and 1 cluster has 1 valid overlap scan (diff = +0.2).
    In repeated cluster sampling, draws of only {c1, c2, c3} yield zero valid observations.
    The test verifies:
    1. Zero imputation occurs (no artificial 0 or 1 values).
    2. Empty draws are conditionally rejected and redrawn.
    3. empty_draws_count is accurately tracked.
    4. Since only cluster 4 is valid, all valid conditional resamples yield +0.2 exactly.
    """
    scores_c = np.array([0.0, 0.0, 0.0, 0.8])
    scores_b = np.array([0.0, 0.0, 0.0, 0.6])
    cluster_ids = np.array(["empty_1", "empty_2", "empty_3", "valid_4"])
    valid_mask = np.array([False, False, False, True])

    res = paired_cluster_bootstrap_segmentation(
        scores_c=scores_c,
        scores_b=scores_b,
        cluster_ids=cluster_ids,
        valid_mask=valid_mask,
        metric_name="lpr",
        num_resamples=500,
        seed=42,
    )

    assert res.num_samples == 1
    assert res.num_clusters == 4
    assert np.isclose(res.observed_difference, 0.2)
    assert res.empty_draws_count > 0, f"Expected empty draws from 3/4 empty clusters, got {res.empty_draws_count}"
    assert np.isclose(res.bootstrap_mean, 0.2)
    assert np.isclose(res.ci_lower_95, 0.2)
    assert np.isclose(res.ci_upper_95, 0.2)


def test_paired_cluster_bootstrap_bca_method_selection():
    """Verify that BCa confidence interval is computed and selected for valid cluster distributions."""
    rng = np.random.default_rng(42)
    # 20 clusters with 5 scans each = 100 scans
    cids = np.repeat([f"cluster_{i}" for i in range(20)], 5)
    # Paired differences with positive trend
    sb = rng.uniform(0.6, 0.85, size=100)
    sc = sb + rng.normal(0.04, 0.02, size=100)

    res = paired_cluster_bootstrap_segmentation(
        scores_c=sc,
        scores_b=sb,
        cluster_ids=cids,
        metric_name="dsc",
        num_resamples=1000,
        seed=42,
    )

    assert res.ci_method == "bca"
    assert res.ci_lower_95 <= res.observed_difference <= res.ci_upper_95
    assert res.ci_lower_percentile_95 <= res.observed_difference <= res.ci_upper_percentile_95
    # Acceleration and bias correction are populated finite numbers
    assert np.isfinite(res.bca_acceleration_a)
    assert np.isfinite(res.bca_bias_correction_z0)


def test_paired_cluster_bootstrap_constant_statistic():
    """Verify BCa numerical stability on constant-statistic dataset (zero variance)."""
    # 10 clusters with identical differences of exactly +0.15
    cids = np.repeat([f"cluster_{i}" for i in range(10)], 4)
    sb = np.full(40, 0.70)
    sc = np.full(40, 0.85)  # Diff = +0.15 constant across all scans and clusters

    res = paired_cluster_bootstrap_segmentation(
        scores_c=sc,
        scores_b=sb,
        cluster_ids=cids,
        metric_name="dsc",
        num_resamples=500,
        seed=42,
    )

    assert np.isclose(res.observed_difference, 0.15)
    # Mid-p tie handling guarantees zero artificial bias when all replicates are tied with observed
    assert np.isclose(res.bca_bias_correction_z0, 0.0)
    # Zero jackknife variance guarantees zero acceleration
    assert np.isclose(res.bca_acceleration_a, 0.0)
    assert np.isclose(res.ci_lower_95, 0.15)
    assert np.isclose(res.ci_upper_95, 0.15)
    assert np.isclose(res.ci_lower_percentile_95, 0.15)
    assert np.isclose(res.ci_upper_percentile_95, 0.15)


def test_paired_cluster_bootstrap_many_tied_replicates():
    """Verify BCa behavior on discrete dataset with many tied bootstrap replicates."""
    rng = np.random.default_rng(1337)
    # 15 clusters, discrete differences sampled from {-0.1, 0.0, +0.1}
    cids = np.repeat([f"cluster_{i}" for i in range(15)], 4)
    sb = np.full(60, 0.75)
    diffs = rng.choice([-0.1, 0.0, 0.1], size=60, p=[0.25, 0.50, 0.25])
    sc = sb + diffs

    res = paired_cluster_bootstrap_segmentation(
        scores_c=sc,
        scores_b=sb,
        cluster_ids=cids,
        metric_name="dsc",
        num_resamples=1000,
        seed=42,
    )

    assert np.isfinite(res.bca_bias_correction_z0)
    assert np.isfinite(res.bca_acceleration_a)
    assert res.ci_method in ["bca", "percentile_fallback"]
    assert res.ci_lower_95 <= res.observed_difference <= res.ci_upper_95
    assert res.ci_lower_percentile_95 <= res.observed_difference <= res.ci_upper_percentile_95
