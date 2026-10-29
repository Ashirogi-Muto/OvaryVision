"""Tests for paired DeLong test, non-inferiority hypothesis testing, bootstrap CIs, and McNemar."""

import numpy as np
import pytest

from ovaryvision.statistics import (
    bootstrap_paired_difference,
    compute_roc_auc,
    evaluate_non_inferiority,
    generate_comprehensive_statistical_report,
    mcnemar_test_paired,
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
