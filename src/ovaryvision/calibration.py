"""Post-hoc temperature scaling, probability calibration, ECE, MCE, and Brier score evaluation."""

from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from ovaryvision.logging_utils import get_logger

logger = get_logger("calibration")


class CalibrationBin(BaseModel):
    """Metrics for a single probability calibration bin."""
    bin_index: int
    bin_lower: float
    bin_upper: float
    sample_count: int
    mean_confidence: float
    observed_frequency: float
    calibration_error: float


class CalibrationReport(BaseModel):
    """Comprehensive probability calibration report before and after temperature scaling."""
    optimal_temperature: float
    uncalibrated_ece: float
    calibrated_ece: float
    uncalibrated_mce: float
    calibrated_mce: float
    uncalibrated_brier: float
    calibrated_brier: float
    binning_method: str
    num_bins: int
    uncalibrated_bins: List[CalibrationBin] = Field(default_factory=list)
    calibrated_bins: List[CalibrationBin] = Field(default_factory=list)


def binary_nll_from_logits(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    """Compute negative log-likelihood (binary cross entropy) given raw logits and temperature.

    Uses numerically stable formulation:
        BCE(z, y) = max(z, 0) - z * y + log(1 + exp(-|z|))
    """
    t = max(temperature, 1e-4)
    z = logits / t
    max_val = np.maximum(z, 0.0)
    loss = max_val - z * labels + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(loss))


def fit_temperature_scaling(
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
    temp_bounds: Tuple[float, float] = (0.05, 10.0),
) -> float:
    """Fit optimal temperature T* strictly on validation logits to minimize Negative Log-Likelihood.

    Args:
        validation_logits: 1D array of validation uncalibrated logits (shape N,).
        validation_labels: 1D array of binary ground-truth labels (shape N,) in {0, 1}.
        temp_bounds: Search interval for temperature parameter.

    Returns:
        Optimal temperature T* > 0.
    """
    if len(validation_logits) != len(validation_labels):
        raise ValueError(
            f"Logits length ({len(validation_logits)}) must match labels length ({len(validation_labels)})"
        )
    if len(validation_logits) == 0:
        raise ValueError("Cannot fit temperature scaling on empty validation set.")

    logits = np.asarray(validation_logits, dtype=np.float64)
    labels = np.asarray(validation_labels, dtype=np.float64)

    def obj(t: float) -> float:
        return binary_nll_from_logits(logits, labels, temperature=t)

    res = minimize_scalar(obj, bounds=temp_bounds, method="bounded")
    if not res.success:
        logger.warning(f"Temperature optimization did not converge cleanly: {res.message}. Using default T=1.0.")
        return 1.0

    optimal_t = float(res.x)
    logger.info(f"Optimal temperature fitted on validation fold: T* = {optimal_t:.4f}")
    return optimal_t


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by temperature and return calibrated probabilities in [0, 1]."""
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    scaled_logits = np.asarray(logits, dtype=np.float64) / temperature
    return expit(scaled_logits)


def compute_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean squared probability error: Brier = (1/N) * sum((p_i - y_i)^2)."""
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probs - y) ** 2))


def compute_calibration_bins(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> Tuple[float, float, List[CalibrationBin]]:
    """Partition predicted probabilities into bins and compute ECE, MCE, and bin statistics.

    Args:
        probabilities: Calibrated or uncalibrated predicted probabilities in [0, 1].
        labels: Binary ground-truth labels in {0, 1}.
        num_bins: Number of probability bins (default 10).
        strategy: 'uniform' (equal width intervals) or 'quantile' (equal sample size).

    Returns:
        Tuple of (ece, mce, list_of_calibration_bins).
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n = len(probs)
    if n == 0:
        raise ValueError("Cannot evaluate calibration on empty predictions array.")

    bins: List[CalibrationBin] = []
    total_ece = 0.0
    max_error = 0.0

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        bin_edges = np.quantile(probs, quantiles)
        # Ensure strictly monotonic edges
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        for i in range(1, len(bin_edges)):
            if bin_edges[i] <= bin_edges[i - 1]:
                bin_edges[i] = bin_edges[i - 1] + 1e-6
    else:
        raise ValueError(f"Unknown binning strategy '{strategy}'. Must be 'uniform' or 'quantile'.")

    for i in range(num_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        if i == num_bins - 1:
            in_bin = (probs >= low) & (probs <= high)
        else:
            in_bin = (probs >= low) & (probs < high)

        count = int(np.sum(in_bin))
        if count > 0:
            mean_conf = float(np.mean(probs[in_bin]))
            obs_freq = float(np.mean(y[in_bin]))
            calib_err = abs(obs_freq - mean_conf)
            weighted_err = (count / n) * calib_err
            total_ece += weighted_err
            max_error = max(max_error, calib_err)
        else:
            mean_conf = float((low + high) / 2.0)
            obs_freq = 0.0
            calib_err = 0.0

        bins.append(
            CalibrationBin(
                bin_index=i,
                bin_lower=float(low),
                bin_upper=float(high),
                sample_count=count,
                mean_confidence=mean_conf,
                observed_frequency=obs_freq,
                calibration_error=calib_err,
            )
        )

    return float(total_ece), float(max_error), bins


def evaluate_probability_calibration(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    num_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> CalibrationReport:
    """Complete calibration workflow: fit T* on validation fold, evaluate test fold before and after.

    Strict Isolation Rule:
        T* is optimized strictly on `val_logits` and `val_labels`.
        The test set is evaluated out-of-sample and never informs the temperature fitting.

    Args:
        val_logits: Validation fold logits for fitting T*.
        val_labels: Validation fold true binary labels.
        test_logits: Test fold logits for evaluation.
        test_labels: Test fold true binary labels.
        num_bins: Number of bins for ECE/MCE.
        strategy: Binning strategy ('uniform' or 'quantile').

    Returns:
        CalibrationReport documenting pre- and post-scaling ECE, MCE, Brier score, and bin profiles.
    """
    # 1. Fit T* on validation set
    optimal_t = fit_temperature_scaling(val_logits, val_labels)

    # 2. Compute uncalibrated and calibrated test probabilities
    uncal_probs = expit(np.asarray(test_logits, dtype=np.float64))
    cal_probs = apply_temperature_scaling(test_logits, optimal_t)

    # 3. Compute Brier scores
    uncal_brier = compute_brier_score(uncal_probs, test_labels)
    cal_brier = compute_brier_score(cal_probs, test_labels)

    # 4. Compute ECE and MCE
    uncal_ece, uncal_mce, uncal_bins = compute_calibration_bins(
        uncal_probs, test_labels, num_bins=num_bins, strategy=strategy
    )
    cal_ece, cal_mce, cal_bins = compute_calibration_bins(
        cal_probs, test_labels, num_bins=num_bins, strategy=strategy
    )

    report = CalibrationReport(
        optimal_temperature=optimal_t,
        uncalibrated_ece=uncal_ece,
        calibrated_ece=cal_ece,
        uncalibrated_mce=uncal_mce,
        calibrated_mce=cal_mce,
        uncalibrated_brier=uncal_brier,
        calibrated_brier=cal_brier,
        binning_method=strategy,
        num_bins=num_bins,
        uncalibrated_bins=uncal_bins,
        calibrated_bins=cal_bins,
    )

    logger.info(
        f"Test calibration results (T*={optimal_t:.4f}): "
        f"ECE {uncal_ece:.4f} -> {cal_ece:.4f}, Brier {uncal_brier:.4f} -> {cal_brier:.4f}"
    )

    return report
