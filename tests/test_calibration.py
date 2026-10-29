"""Tests for probability calibration, temperature scaling, ECE, MCE, and Brier score."""

import numpy as np
import pytest
from scipy.special import expit

from ovaryvision.calibration import (
    apply_temperature_scaling,
    compute_brier_score,
    compute_calibration_bins,
    evaluate_probability_calibration,
    fit_temperature_scaling,
)


def test_brier_score_known_cases():
    """Verify Brier score on boundary and analytical test cases."""
    # Perfect predictions
    perfect_probs = np.array([0.0, 1.0, 0.0, 1.0])
    labels = np.array([0, 1, 0, 1])
    assert compute_brier_score(perfect_probs, labels) == 0.0

    # Inverted predictions
    inverted_probs = np.array([1.0, 0.0, 1.0, 0.0])
    assert compute_brier_score(inverted_probs, labels) == 1.0

    # Constant 0.5 guessing
    random_probs = np.array([0.5, 0.5, 0.5, 0.5])
    assert compute_brier_score(random_probs, labels) == 0.25


def test_temperature_scaling_overconfident_logits():
    """Verify temperature scaling recovers true scale factor on over-confident synthetic logits."""
    rng = np.random.default_rng(42)
    n = 2000

    # True latent log-odds
    true_logits = rng.normal(0, 1.5, size=n)
    true_probs = expit(true_logits)
    labels = (rng.uniform(0, 1, size=n) < true_probs).astype(int)

    # Overconfident model has logits scaled by 2.5
    overconfident_logits = true_logits * 2.5

    t_star = fit_temperature_scaling(overconfident_logits, labels)

    # Optimal temperature should be close to 2.5
    assert 2.0 < t_star < 3.0


def test_apply_temperature_scaling_bounds():
    """Verify calibrated probabilities lie strictly in [0, 1] and temperature scaling modifies probabilities."""
    logits = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    t = 2.0
    cal_probs = apply_temperature_scaling(logits, temperature=t)

    assert np.all(cal_probs >= 0.0)
    assert np.all(cal_probs <= 1.0)
    # Higher temperature softens probabilities toward 0.5
    raw_probs = expit(logits)
    assert abs(cal_probs[4] - 0.5) < abs(raw_probs[4] - 0.5)

    with pytest.raises(ValueError, match="Temperature must be positive"):
        apply_temperature_scaling(logits, temperature=-1.0)


def test_compute_calibration_bins_uniform_and_quantile():
    """Verify uniform and quantile binning calculate non-negative ECE and MCE."""
    rng = np.random.default_rng(123)
    probs = rng.uniform(0, 1, size=500)
    labels = (rng.uniform(0, 1, size=500) < probs).astype(int)

    ece_u, mce_u, bins_u = compute_calibration_bins(probs, labels, num_bins=10, strategy="uniform")
    ece_q, mce_q, bins_q = compute_calibration_bins(probs, labels, num_bins=10, strategy="quantile")

    assert 0.0 <= ece_u <= 1.0
    assert 0.0 <= mce_u <= 1.0
    assert len(bins_u) == 10

    assert 0.0 <= ece_q <= 1.0
    assert 0.0 <= mce_q <= 1.0
    assert len(bins_q) == 10


def test_evaluate_probability_calibration_workflow():
    """Verify end-to-end calibration workflow reduces ECE on overconfident validation/test splits."""
    rng = np.random.default_rng(42)
    n = 1000

    true_logits_val = rng.normal(0, 1.0, size=n)
    probs_val = expit(true_logits_val)
    val_labels = (rng.uniform(0, 1, size=n) < probs_val).astype(int)
    val_logits = true_logits_val * 3.0  # severely overconfident

    true_logits_test = rng.normal(0, 1.0, size=n)
    probs_test = expit(true_logits_test)
    test_labels = (rng.uniform(0, 1, size=n) < probs_test).astype(int)
    test_logits = true_logits_test * 3.0

    report = evaluate_probability_calibration(
        val_logits=val_logits,
        val_labels=val_labels,
        test_logits=test_logits,
        test_labels=test_labels,
        num_bins=10,
    )

    assert report.optimal_temperature > 1.5
    # Calibrated ECE should improve compared to uncalibrated
    assert report.calibrated_ece < report.uncalibrated_ece
    assert report.calibrated_brier <= report.uncalibrated_brier

