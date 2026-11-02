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


class ClusterBootstrapResult(BaseModel):
    """Paired pseudo-patient-cluster bootstrap result with BCa and percentile intervals."""
    metric_name: str
    observed_difference: float  # Mean(Arm C) - Mean(Arm B)
    ci_lower_95: float          # Primary CI lower bound (BCa if valid, else percentile fallback)
    ci_upper_95: float          # Primary CI upper bound (BCa if valid, else percentile fallback)
    ci_method: str = "bca"      # 'bca' or 'percentile_fallback'
    ci_lower_percentile_95: float = 0.0  # Standard unadjusted percentile lower bound
    ci_upper_percentile_95: float = 0.0  # Standard unadjusted percentile upper bound
    bca_bias_correction_z0: float = 0.0
    bca_acceleration_a: float = 0.0
    bootstrap_mean: float
    bootstrap_std: float
    p_value: float
    num_clusters: int
    num_samples: int
    num_resamples: int
    empty_draws_count: int = 0


def paired_cluster_bootstrap_segmentation(
    scores_c: np.ndarray,
    scores_b: np.ndarray,
    cluster_ids: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    metric_name: str = "dsc",
    num_resamples: int = 2000,
    seed: int = 42,
    max_redraws_per_resample: int = 200,
) -> ClusterBootstrapResult:
    """Paired pseudo-patient-cluster bootstrap with multiplicity preservation.

    Preserves cluster resampling multiplicities:
    When a cluster is drawn `m` times with replacement, all scans in that cluster
    appear exactly `m` times in the bootstrap sample, contributing `m` times to both
    numerator and denominator. Duplicate draws are NOT collapsed via set union.

    Edge Case Handling for Conditional Metrics (e.g. LPR):
    When estimating metrics defined conditionally on sub-populations (such as the caliper
    overlap cohort for LPR), an iteration where all sampled clusters contain zero valid
    observations is degenerate (denominator = 0). Imputing an arbitrary constant (0 or 1)
    is mathematically invalid because it introduces artificial variance and bias.
    Following conditional bootstrap theory (Davison & Hinkley, 1997, Sec. 3.8), degenerate
    draws are handled via rejection sampling: any draw with zero valid observations is rejected
    and redrawn until a valid conditional realization is obtained, guaranteeing exactly
    `num_resamples` valid draws. The total count of redrawn empty draws is recorded.

    Args:
        scores_c: Metric values for Arm C (shape N).
        scores_b: Metric values for Arm B (shape N).
        cluster_ids: Cluster identifier for each scan (shape N).
        valid_mask: Optional boolean mask (shape N) indicating eligible scans
                    (e.g., overlap cohort for LPR). If None, all scans are valid.
        metric_name: Name of metric (e.g. 'dsc', 'biou_d2', 'lpr').
        num_resamples: Number of bootstrap iterations (default 2000).
        seed: Random seed for determinism.
        max_redraws_per_resample: Maximum allowable consecutive redrawing attempts.

    Returns:
        ClusterBootstrapResult containing observed difference, 95% BCa/percentile bounds,
        bootstrap standard error, empirical p-value, sample counts, and empty redraw count.
    """
    sc = np.asarray(scores_c, dtype=np.float64)
    sb = np.asarray(scores_b, dtype=np.float64)
    clean_cids = []
    for idx_c, cid in enumerate(cluster_ids):
        if cid is None or (isinstance(cid, (float, np.floating)) and np.isnan(cid)) or str(cid).lower() in ("nan", "none", ""):
            clean_cids.append(f"SINGLETON_{idx_c}")
        else:
            clean_cids.append(str(cid))
    cids = np.asarray(clean_cids, dtype=str)

    # 1. Align all inputs to full cohort length
    n_total = len(sc)
    if valid_mask is not None:
        vmask = np.asarray(valid_mask, dtype=bool)
        if len(vmask) != n_total:
            raise ValueError(f"valid_mask length ({len(vmask)}) does not match scores length ({n_total})")
    else:
        vmask = np.ones(n_total, dtype=bool)

    if not np.any(vmask):
        raise ValueError(f"No valid observations provided for cluster bootstrap ({metric_name})")

    # Observed point estimate evaluated strictly on valid observations
    valid_diff = sc[vmask] - sb[vmask]
    observed_diff = float(np.mean(valid_diff))

    # Identify unique clusters across the full cohort
    unique_clusters, cluster_inverse = np.unique(cids, return_inverse=True)
    num_clusters = len(unique_clusters)

    # Pre-calculate per-cluster sum of differences and valid cluster size
    cluster_diff_sums = np.zeros(num_clusters, dtype=np.float64)
    cluster_valid_sizes = np.zeros(num_clusters, dtype=np.float64)
    for k in range(num_clusters):
        idx = np.where(cluster_inverse == k)[0]
        # Only valid observations in cluster k contribute to diff sum and size
        valid_in_k = idx[vmask[idx]]
        if len(valid_in_k) > 0:
            cluster_diff_sums[k] = np.sum(sc[valid_in_k] - sb[valid_in_k])
            cluster_valid_sizes[k] = len(valid_in_k)

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(num_resamples, dtype=np.float64)
    total_empty_draws = 0

    for b in range(num_resamples):
        redraw_count = 0
        while True:
            # Sample cluster indices with replacement (preserves multiplicity)
            sampled_k = rng.choice(num_clusters, size=num_clusters, replace=True)
            multiplicities = np.bincount(sampled_k, minlength=num_clusters)

            sampled_total_count = np.sum(multiplicities * cluster_valid_sizes)
            if sampled_total_count > 0:
                sampled_total_diff = np.sum(multiplicities * cluster_diff_sums)
                boot_diffs[b] = sampled_total_diff / sampled_total_count
                break

            # Handle degenerate draw: redraw cluster sample conditionally
            redraw_count += 1
            total_empty_draws += 1
            if redraw_count > max_redraws_per_resample:
                raise RuntimeError(
                    f"Cluster bootstrap for {metric_name} failed: exceeded {max_redraws_per_resample} "
                    "consecutive redraws due to zero valid observations. Cluster pool too sparse."
                )

    # 1. Standard empirical percentile interval
    alpha = 0.05
    perc_low = float(np.percentile(boot_diffs, 100.0 * (alpha / 2.0)))
    perc_high = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha / 2.0)))

    # 2. Bias-Corrected and Accelerated (BCa) adjustment
    # Bias correction z0 with mid-p tie convention:
    # prop_less = P(boot < obs) + 0.5 * P(boot == obs)
    # This prevents artificial bias distortion on tied bootstrap replicates and constant datasets.
    is_tied = np.isclose(boot_diffs, observed_diff, atol=1e-12)
    prop_less = float(np.mean(boot_diffs < (observed_diff - 1e-12)) + 0.5 * np.mean(is_tied))
    # Finite-sample clipping rule to prevent infinite z0: [1/(B+1), B/(B+1)]
    prop_less = float(np.clip(prop_less, 1.0 / (num_resamples + 1), num_resamples / (num_resamples + 1)))
    z0 = float(norm.ppf(prop_less))

    # Acceleration a via cluster-level jackknife (leave-one-cluster-out)
    jack_valid = True
    jack_diffs = np.zeros(num_clusters, dtype=np.float64)
    for k in range(num_clusters):
        jack_count = np.sum(cluster_valid_sizes) - cluster_valid_sizes[k]
        if jack_count <= 0:
            jack_valid = False
            break
        jack_sum = np.sum(cluster_diff_sums) - cluster_diff_sums[k]
        jack_diffs[k] = jack_sum / jack_count

    if jack_valid and num_clusters > 1:
        mean_jack = float(np.mean(jack_diffs))
        jack_dev = mean_jack - jack_diffs
        denom = 6.0 * (np.sum(jack_dev ** 2) ** 1.5)
        a = float(np.sum(jack_dev ** 3) / denom) if denom > 1e-12 else 0.0
    else:
        a = 0.0

    z_low = float(norm.ppf(alpha / 2.0))        # -1.95996
    z_high = float(norm.ppf(1.0 - alpha / 2.0))  # +1.95996

    method_used = "percentile_fallback"
    ci_low, ci_high = perc_low, perc_high

    if np.isfinite(z0) and np.isfinite(a):
        denom_low = 1.0 - a * (z0 + z_low)
        denom_high = 1.0 - a * (z0 + z_high)

        if denom_low > 1e-8 and denom_high > 1e-8:
            arg_low = z0 + (z0 + z_low) / denom_low
            arg_high = z0 + (z0 + z_high) / denom_high
            if np.isfinite(arg_low) and np.isfinite(arg_high):
                q_low = float(norm.cdf(arg_low))
                q_high = float(norm.cdf(arg_high))
                if 0.0 < q_low < q_high < 1.0:
                    ci_low = float(np.percentile(boot_diffs, 100.0 * q_low))
                    ci_high = float(np.percentile(boot_diffs, 100.0 * q_high))
                    method_used = "bca"

    boot_mean = float(np.mean(boot_diffs))
    boot_std = float(np.std(boot_diffs, ddof=1))

    # Two-sided empirical p-value
    p_val = 2.0 * min(float(np.mean(boot_diffs <= 0)), float(np.mean(boot_diffs >= 0)))
    p_val = min(1.0, max(p_val, 1.0 / num_resamples))

    return ClusterBootstrapResult(
        metric_name=metric_name,
        observed_difference=observed_diff,
        ci_lower_95=ci_low,
        ci_upper_95=ci_high,
        ci_method=method_used,
        ci_lower_percentile_95=perc_low,
        ci_upper_percentile_95=perc_high,
        bca_bias_correction_z0=z0,
        bca_acceleration_a=a,
        bootstrap_mean=boot_mean,
        bootstrap_std=boot_std,
        p_value=float(p_val),
        num_clusters=num_clusters,
        num_samples=int(np.sum(vmask)),
        num_resamples=num_resamples,
        empty_draws_count=total_empty_draws,
    )


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
