"""Statistical comparison and hypothesis testing: Paired DeLong test, bootstrap CIs, non-inferiority, and McNemar's test."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import chi2, norm

from ovaryvision.logging_utils import get_logger

logger = get_logger("statistics")


class BootstrapConfidenceInterval(BaseModel):
    """Percentile and empirical bootstrap confidence bounds for a metric difference."""
    metric_name: str
    observed_difference: float  # Value(Arm C) - Value(Arm B)
    ci_lower_95: float
    ci_upper_95: float
    bootstrap_mean: float
    bootstrap_std: float
    num_resamples: int


class NonInferiorityResult(BaseModel):
    """Non-inferiority hypothesis test outcome for AUC_C - AUC_B > delta."""
    margin_delta: float
    observed_auc_diff: float  # AUC_C - AUC_B
    se_delong: float
    z_score: float
    p_value: float
    non_inferiority_declared: bool
    confidence_level: float = 0.95
    ci_lower: float


class PairedDeLongResult(BaseModel):
    """DeLong et al. (1988) paired ROC comparison test results."""
    auc_model_1: float
    auc_model_2: float
    auc_difference: float  # auc_1 - auc_2
    covariance_delong: float
    se_difference: float
    z_statistic: float
    p_value_two_tailed: float


class McNemarResult(BaseModel):
    """McNemar's test for paired classification decision discordance."""
    n_concordant_correct: int  # Both correct
    n_concordant_incorrect: int  # Both incorrect
    n_discordant_1_only: int  # Model 1 correct, Model 2 incorrect
    n_discordant_2_only: int  # Model 2 correct, Model 1 incorrect
    chi2_statistic: float
    p_value: float
    threshold_used: float


class ComprehensiveComparisonReport(BaseModel):
    """Full statistical comparison artifact between two experimental arms."""
    arm_1_name: str
    arm_2_name: str
    sample_size: int
    positives_count: int
    negatives_count: int
    delong_test: PairedDeLongResult
    non_inferiority_primary: NonInferiorityResult
    non_inferiority_sensitivities: List[NonInferiorityResult] = Field(default_factory=list)
    bootstrap_auc: BootstrapConfidenceInterval
    bootstrap_brier: BootstrapConfidenceInterval
    mcnemar_test: McNemarResult


def compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Area Under the Receiver Operating Characteristic curve via Mann-Whitney U statistic.

    Args:
        scores: Continuous predicted probabilities or decision scores.
        labels: Binary ground-truth labels in {0, 1}.

    Returns:
        Empirical ROC AUC in [0, 1].
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    pos = s[y == 1]
    neg = s[y == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Cannot compute AUC when either positive or negative class is absent.")

    # Sort ranks handling ties with average rank
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)

    # Handle ties
    sorted_s = s[order]
    unique_vals, indices, counts = np.unique(sorted_s, return_index=True, return_counts=True)
    for idx, count in zip(indices, counts):
        if count > 1:
            tie_rank = np.mean(np.arange(idx + 1, idx + count + 1))
            ranks[order[idx : idx + count]] = tie_rank

    sum_pos_ranks = np.sum(ranks[y == 1])
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def compute_delong_covariance(
    scores_1: np.ndarray,
    scores_2: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute DeLong et al. (1988) empirical AUCs and variance-covariance matrix.

    Returns:
        (auc_1, auc_2, var_1, var_2, cov_12)
    """
    s1 = np.asarray(scores_1, dtype=np.float64)
    s2 = np.asarray(scores_2, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)

    if m == 0 or n == 0:
        raise ValueError("Both positive and negative classes must be present.")

    # Structural components V10 (for positive samples) and V01 (for negative samples)
    # V10_k(X_i) = (1/n) * sum_{j=1}^n psi(X_i, Y_j)
    # V01_k(Y_j) = (1/m) * sum_{i=1}^m psi(X_i, Y_j)

    pos_s1 = s1[pos_idx]
    neg_s1 = s1[neg_idx]
    pos_s2 = s2[pos_idx]
    neg_s2 = s2[neg_idx]

    # Model 1 comparisons
    # Matrix of shape (m, n): pos vs neg
    diff1 = pos_s1[:, np.newaxis] - neg_s1[np.newaxis, :]
    psi1 = np.where(diff1 > 0, 1.0, np.where(diff1 == 0, 0.5, 0.0))

    # Model 2 comparisons
    diff2 = pos_s2[:, np.newaxis] - neg_s2[np.newaxis, :]
    psi2 = np.where(diff2 > 0, 1.0, np.where(diff2 == 0, 0.5, 0.0))

    auc1 = float(np.mean(psi1))
    auc2 = float(np.mean(psi2))

    v10_1 = np.mean(psi1, axis=1)  # shape (m,)
    v10_2 = np.mean(psi2, axis=1)  # shape (m,)

    v01_1 = np.mean(psi1, axis=0)  # shape (n,)
    v01_2 = np.mean(psi2, axis=0)  # shape (n,)

    # Covariance matrices
    # S10: covariance of (v10_1, v10_2)
    s10 = np.cov(np.vstack([v10_1, v10_2]), ddof=1) if m > 1 else np.zeros((2, 2))
    # S01: covariance of (v01_1, v01_2)
    s01 = np.cov(np.vstack([v01_1, v01_2]), ddof=1) if n > 1 else np.zeros((2, 2))

    # S = (1/m)*S10 + (1/n)*S01
    s_mat = (s10 / m) + (s01 / n)
    var1 = float(s_mat[0, 0])
    var2 = float(s_mat[1, 1])
    cov12 = float(s_mat[0, 1])

    return auc1, auc2, var1, var2, cov12


def paired_delong_test(
    scores_1: np.ndarray,
    scores_2: np.ndarray,
    labels: np.ndarray,
) -> PairedDeLongResult:
    """Execute paired DeLong test for difference between two ROC curves on the same cohort."""
    auc1, auc2, var1, var2, cov12 = compute_delong_covariance(scores_1, scores_2, labels)
    diff = auc1 - auc2
    var_diff = max(var1 + var2 - 2.0 * cov12, 1e-12)
    se_diff = float(np.sqrt(var_diff))

    z_stat = float(diff / se_diff)
    p_val = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))

    return PairedDeLongResult(
        auc_model_1=auc1,
        auc_model_2=auc2,
        auc_difference=diff,
        covariance_delong=cov12,
        se_difference=se_diff,
        z_statistic=z_stat,
        p_value_two_tailed=p_val,
    )


def evaluate_non_inferiority(
    scores_test: np.ndarray,
    scores_reference: np.ndarray,
    labels: np.ndarray,
    margin_delta: float = -0.02,
    alpha: float = 0.05,
) -> NonInferiorityResult:
    """Test non-inferiority hypothesis: H0: AUC_test - AUC_ref <= delta vs H1: AUC_test - AUC_ref > delta.

    Args:
        scores_test: Predicted probabilities of test arm (e.g. Arm C Masked).
        scores_reference: Predicted probabilities of reference arm (e.g. Arm B Inpainted).
        labels: True binary labels in {0, 1}.
        margin_delta: Non-inferiority margin delta (default -0.02; pre-registered in SPEC.md).
        alpha: Significance level (default 0.05).

    Returns:
        NonInferiorityResult with one-sided z-statistic, p-value, and declaration status.
    """
    delong = paired_delong_test(scores_test, scores_reference, labels)
    diff = delong.auc_difference
    se = delong.se_difference

    # One-sided z-statistic: (diff - delta) / SE
    z_ni = float((diff - margin_delta) / se)
    # P-value for upper-tail test: P(Z > z_ni) under H0: diff = delta
    p_ni = float(1.0 - norm.cdf(z_ni))

    # One-sided 1-alpha confidence lower bound: diff - z_{1-alpha} * SE
    z_crit = float(norm.ppf(1.0 - alpha))
    ci_lower = float(diff - z_crit * se)

    declared = ci_lower > margin_delta

    return NonInferiorityResult(
        margin_delta=margin_delta,
        observed_auc_diff=diff,
        se_delong=se,
        z_score=z_ni,
        p_value=p_ni,
        non_inferiority_declared=declared,
        confidence_level=1.0 - alpha,
        ci_lower=ci_lower,
    )


def bootstrap_paired_difference(
    scores_1: np.ndarray,
    scores_2: np.ndarray,
    labels: np.ndarray,
    metric_fn: str = "auc",  # 'auc' or 'brier'
    num_resamples: int = 1000,
    seed: int = 42,
) -> BootstrapConfidenceInterval:
    """Stratified paired bootstrap confidence interval for difference (Model 1 - Model 2)."""
    s1 = np.asarray(scores_1, dtype=np.float64)
    s2 = np.asarray(scores_2, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)

    # Observed point estimates
    if metric_fn == "auc":
        obs_1 = compute_roc_auc(s1, y)
        obs_2 = compute_roc_auc(s2, y)
        obs_diff = obs_1 - obs_2
    elif metric_fn == "brier":
        obs_1 = float(np.mean((s1 - y) ** 2))
        obs_2 = float(np.mean((s2 - y) ** 2))
        obs_diff = obs_1 - obs_2
    else:
        raise ValueError(f"Unsupported metric_fn: '{metric_fn}'")

    rng = np.random.default_rng(seed)
    diffs = np.empty(num_resamples, dtype=np.float64)

    for b in range(num_resamples):
        # Stratified sampling with replacement
        boot_pos = rng.choice(pos_idx, size=m, replace=True)
        boot_neg = rng.choice(neg_idx, size=n, replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])

        b_s1 = s1[boot_idx]
        b_s2 = s2[boot_idx]
        b_y = y[boot_idx]

        if metric_fn == "auc":
            diffs[b] = compute_roc_auc(b_s1, b_y) - compute_roc_auc(b_s2, b_y)
        else:
            diffs[b] = float(np.mean((b_s1 - b_y) ** 2) - np.mean((b_s2 - b_y) ** 2))

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    return BootstrapConfidenceInterval(
        metric_name=metric_fn,
        observed_difference=float(obs_diff),
        ci_lower_95=float(ci_low),
        ci_upper_95=float(ci_high),
        bootstrap_mean=float(np.mean(diffs)),
        bootstrap_std=float(np.std(diffs, ddof=1)),
        num_resamples=num_resamples,
    )


def mcnemar_test_paired(
    scores_1: np.ndarray,
    scores_2: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> McNemarResult:
    """Perform McNemar's test for paired classification decisions with continuity correction."""
    pred1 = (np.asarray(scores_1, dtype=np.float64) >= threshold).astype(int)
    pred2 = (np.asarray(scores_2, dtype=np.float64) >= threshold).astype(int)
    y = np.asarray(labels, dtype=np.int32)

    corr1 = (pred1 == y)
    corr2 = (pred2 == y)

    n11 = int(np.sum(corr1 & corr2))
    n00 = int(np.sum((~corr1) & (~corr2)))
    n10 = int(np.sum(corr1 & (~corr2)))  # Model 1 correct, Model 2 incorrect
    n01 = int(np.sum((~corr1) & corr2))  # Model 2 correct, Model 1 incorrect

    total_discordant = n10 + n01
    if total_discordant == 0:
        chi2_stat = 0.0
        p_val = 1.0
    else:
        # Edwards continuity correction
        num = max(0.0, abs(n10 - n01) - 1.0) ** 2
        chi2_stat = float(num / total_discordant)
        p_val = float(1.0 - chi2.cdf(chi2_stat, df=1))

    return McNemarResult(
        n_concordant_correct=n11,
        n_concordant_incorrect=n00,
        n_discordant_1_only=n10,
        n_discordant_2_only=n01,
        chi2_statistic=chi2_stat,
        p_value=p_val,
        threshold_used=threshold,
    )


def generate_comprehensive_statistical_report(
    scores_arm_c: np.ndarray,
    scores_arm_b: np.ndarray,
    labels: np.ndarray,
    arm_c_name: str = "Arm C (Masked)",
    arm_b_name: str = "Arm B (Inpainted)",
    primary_delta: float = -0.02,
    sensitivity_deltas: Optional[List[float]] = None,
    num_bootstrap: int = 1000,
    seed: int = 42,
) -> ComprehensiveComparisonReport:
    """Generate the full pre-registered statistical evaluation report comparing Arm C against Arm B."""
    if sensitivity_deltas is None:
        sensitivity_deltas = [-0.01, -0.02, -0.03]

    y = np.asarray(labels, dtype=np.int32)
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))

    # 1. DeLong paired test
    delong = paired_delong_test(scores_arm_c, scores_arm_b, y)

    # 2. Primary non-inferiority test
    primary_ni = evaluate_non_inferiority(scores_arm_c, scores_arm_b, y, margin_delta=primary_delta)

    # 3. Sensitivity analysis over delta bounds
    sensitivities: List[NonInferiorityResult] = []
    for d in sensitivity_deltas:
        sensitivities.append(evaluate_non_inferiority(scores_arm_c, scores_arm_b, y, margin_delta=d))

    # 4. Bootstrap intervals
    boot_auc = bootstrap_paired_difference(
        scores_arm_c, scores_arm_b, y, metric_fn="auc", num_resamples=num_bootstrap, seed=seed
    )
    boot_brier = bootstrap_paired_difference(
        scores_arm_c, scores_arm_b, y, metric_fn="brier", num_resamples=num_bootstrap, seed=seed
    )

    # 5. McNemar's test
    mcnemar = mcnemar_test_paired(scores_arm_c, scores_arm_b, y, threshold=0.5)

    return ComprehensiveComparisonReport(
        arm_1_name=arm_c_name,
        arm_2_name=arm_b_name,
        sample_size=len(y),
        positives_count=pos_count,
        negatives_count=neg_count,
        delong_test=delong,
        non_inferiority_primary=primary_ni,
        non_inferiority_sensitivities=sensitivities,
        bootstrap_auc=boot_auc,
        bootstrap_brier=boot_brier,
        mcnemar_test=mcnemar,
    )
