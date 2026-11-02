#!/usr/bin/env python3
"""
OvaryVision — Statistical Inference Verification Script (Final Corrected Version).

Inspects, recalculates, and validates inferential statistics across all 21 paired cluster
comparisons, evaluating:
1. Original percentile-tail p-values with clipping floor.
2. Exact BCa-inverted p-values (dual to BCa 95% CIs).
3. Null-centered bootstrap p-values with add-one finite-sample correction and finite resolution.
4. Cluster-robust jackknife Wald p-values.
5. Training-seed variability (Seed 42, 1337, 2026) vs scan-level bootstrap uncertainty.
6. Family-wise multiple comparison adjustments (primary 7-test family vs full 21-test family).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_jackknife_acceleration_and_se(
    c_diff_sums: np.ndarray, c_sizes: np.ndarray
) -> Tuple[float, float]:
    """Compute acceleration a and jackknife standard error via leave-one-cluster-out."""
    n_clusters = len(c_diff_sums)
    jack_diffs = np.zeros(n_clusters, dtype=np.float64)
    tot_sum = np.sum(c_diff_sums)
    tot_cnt = np.sum(c_sizes)

    for k in range(n_clusters):
        jk_cnt = tot_cnt - c_sizes[k]
        jack_diffs[k] = (tot_sum - c_diff_sums[k]) / jk_cnt if jk_cnt > 0 else 0.0

    mean_jack = float(np.mean(jack_diffs))
    jack_dev = mean_jack - jack_diffs
    denom = 6.0 * (np.sum(jack_dev ** 2) ** 1.5)
    a = float(np.sum(jack_dev ** 3) / denom) if denom > 1e-12 else 0.0
    se_jack = float(np.sqrt((n_clusters - 1) / n_clusters * np.sum(jack_dev ** 2)))
    return a, se_jack


def evaluate_cluster_inference(
    s1: np.ndarray,
    s2: np.ndarray,
    cids: np.ndarray,
    valid_m: np.ndarray | None = None,
    num_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run full inferential calculations across four p-value formulations."""
    if valid_m is not None:
        vmask = np.asarray(valid_m, dtype=bool)
    else:
        vmask = np.ones(len(s1), dtype=bool)

    diff = s1[vmask] - s2[vmask]
    obs_diff = float(np.mean(diff))

    u_clusters, c_inv = np.unique(cids, return_inverse=True)
    n_clusters = len(u_clusters)

    c_diff_sums = np.zeros(n_clusters, dtype=np.float64)
    c_sizes = np.zeros(n_clusters, dtype=np.float64)
    for k in range(n_clusters):
        idx = np.where(c_inv == k)[0]
        v_idx = idx[vmask[idx]]
        if len(v_idx) > 0:
            c_diff_sums[k] = np.sum(s1[v_idx] - s2[v_idx])
            c_sizes[k] = len(v_idx)

    # 1. Jackknife acceleration and SE
    a, se_jack = compute_jackknife_acceleration_and_se(c_diff_sums, c_sizes)

    # 2. Cluster Bootstrap Resampling (preserving pairing and multiplicity)
    rng = np.random.default_rng(seed)
    boot = np.empty(num_resamples, dtype=np.float64)
    for b in range(num_resamples):
        while True:
            sk = rng.choice(n_clusters, size=n_clusters, replace=True)
            mult = np.bincount(sk, minlength=n_clusters)
            cnt = np.sum(mult * c_sizes)
            if cnt > 0:
                boot[b] = np.sum(mult * c_diff_sums) / cnt
                break

    se_boot = float(np.std(boot, ddof=1))

    # 3. Median bias correction z0 with mid-p rank tie correction
    is_tied = np.isclose(boot, obs_diff, atol=1e-12)
    prop_less = float(np.mean(boot < (obs_diff - 1e-12)) + 0.5 * np.mean(is_tied))
    prop_less = float(np.clip(prop_less, 1.0 / (num_resamples + 1), num_resamples / (num_resamples + 1)))
    z0 = float(norm.ppf(prop_less))

    # 4. Standard Percentile Interval
    alpha = 0.05
    perc_low = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    perc_high = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))

    # 5. BCa Confidence Interval
    z_low = norm.ppf(alpha / 2.0)
    z_high = norm.ppf(1.0 - alpha / 2.0)
    d_low = 1.0 - a * (z0 + z_low)
    d_high = 1.0 - a * (z0 + z_high)
    q_low = float(norm.cdf(z0 + (z0 + z_low) / d_low))
    q_high = float(norm.cdf(z0 + (z0 + z_high) / d_high))
    bca_low = float(np.percentile(boot, 100.0 * q_low))
    bca_high = float(np.percentile(boot, 100.0 * q_high))

    # -------------------------------------------------------------
    # P-VALUE FORMULATIONS
    # -------------------------------------------------------------
    # A. Original code p-value (empirical percentile tail with 1/B clipping floor)
    p_orig = float(2.0 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
    p_orig = min(1.0, max(p_orig, 1.0 / num_resamples))

    # B. BCa-Inverted p-value (exact dual to BCa confidence interval)
    is_tied_0 = np.isclose(boot, 0.0, atol=1e-12)
    p_less_0 = float(np.mean(boot < -1e-12) + 0.5 * np.mean(is_tied_0))
    p_less_0 = float(np.clip(p_less_0, 1.0 / (num_resamples + 1), num_resamples / (num_resamples + 1)))
    zq0 = float(norm.ppf(p_less_0))
    denom_w = 1.0 + a * (zq0 - z0)
    if denom_w > 1e-8:
        w = (zq0 - z0) / denom_w - z0
        p_bca = float(2.0 * min(norm.cdf(w), 1.0 - norm.cdf(w)))
        p_bca = min(1.0, max(p_bca, 1.0 / (num_resamples + 1)))
    else:
        p_bca = p_orig

    # C. Null-Centered Bootstrap p-value with add-one finite-sample correction
    # Imposing H0: theta = 0 by centering resamples at 0 (boot - obs_diff)
    boot_null = boot - obs_diff
    num_extreme = int(np.sum(np.abs(boot_null) >= (np.abs(obs_diff) - 1e-12)))
    p_null = float((1.0 + num_extreme) / (1.0 + num_resamples))

    # D. Cluster-Robust Jackknife Wald p-value
    z_wald = obs_diff / se_jack if se_jack > 1e-12 else 0.0
    p_wald = float(2.0 * (1.0 - norm.cdf(abs(z_wald))))

    return {
        "obs_diff": obs_diff,
        "bca_ci_lower": bca_low,
        "bca_ci_upper": bca_high,
        "perc_ci_lower": perc_low,
        "perc_ci_upper": perc_high,
        "z0_bias": z0,
        "acceleration": a,
        "se_jackknife": se_jack,
        "se_bootstrap": se_boot,
        "p_val_original": p_orig,
        "p_val_bca_inverted": p_bca,
        "p_val_null_centered": p_null,
        "p_val_wald": p_wald,
        "num_extreme_null": num_extreme,
        "num_clusters": n_clusters,
        "num_samples": int(np.sum(vmask)),
    }


def format_p_value(p: float, num_resamples: int = 2000) -> str:
    """Format p-value with inequality when at finite resolution floor."""
    min_resolvable = 1.0 / (num_resamples + 1)
    if p <= min_resolvable * 1.01:
        return "< 0.001"
    elif p < 0.001:
        return f"{p:.4f}"
    else:
        return f"{p:.4f}"


def main():
    base_dir = Path(".")
    output_dir = base_dir / "outputs/final_research_audit/statistical_inference_verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []
    log_lines.append(f"Inference verification started at {datetime.now(timezone.utc).isoformat()}")

    # Load 3-seed per-scan metrics
    seeds = [42, 1337, 2026]
    dfs = {
        s: pd.read_csv(base_dir / f"outputs/benchmark_runs/segmentation_benchmark/seed{s}_test_evaluation/seed{s}_per_scan_test_metrics.csv")
        for s in seeds
    }
    base = dfs[42].copy()
    arms = ["arm_a", "arm_b", "arm_c"]
    is_normal = base["is_normal"].to_numpy().astype(bool)
    is_tumour = ~is_normal
    is_overlap = is_tumour & base["is_overlap"].to_numpy().astype(bool)

    # Compute scan-level averages across 3 seeds
    for arm in arms:
        for m in ["dsc", "iou", "biou"]:
            col = f"{arm}_{m}"
            stk = np.stack([dfs[s][col].to_numpy() for s in seeds], axis=1)
            base[f"{col}_mean"] = np.nan
            base.loc[is_tumour, f"{col}_mean"] = np.mean(stk[is_tumour], axis=1)
        col = f"{arm}_lpr"
        stk = np.stack([dfs[s][col].to_numpy() for s in seeds], axis=1)
        base[f"{col}_mean"] = np.nan
        base.loc[is_overlap, f"{col}_mean"] = np.mean(stk[is_overlap], axis=1)
        for m in ["fpr_intra", "fpr_extra", "clean_rej"]:
            col = f"{arm}_{m}"
            stk = np.stack([dfs[s][col].to_numpy() for s in seeds], axis=1)
            base[f"{col}_mean"] = np.nan
            base.loc[is_normal, f"{col}_mean"] = np.mean(stk[is_normal], axis=1)

    tumour_df = base[~base["is_normal"]].copy()
    overlap_mask = tumour_df["is_overlap"].to_numpy().astype(bool)
    normal_df = base[base["is_normal"]].copy()

    comparisons = [
        ("Arm C (Masked) vs Arm B (Inpainted)", "arm_c", "arm_b", True, "Primary Confirmatory Family (M=7)"),
        ("Arm C (Masked) vs Arm A (Raw)", "arm_c", "arm_a", False, "Secondary Benchmark Family (M=21)"),
        ("Arm B (Inpainted) vs Arm A (Raw)", "arm_b", "arm_a", False, "Secondary Benchmark Family (M=21)"),
    ]

    all_results = []

    for comp_name, a1, a2, is_primary, family_name in comparisons:
        # 1. Tumour metrics
        for m, cohort_name, vm, tier in [
            ("dsc", "All Tumour Scans (N=382)", None, "Primary Endpoint" if is_primary else "Secondary Endpoint"),
            ("iou", "All Tumour Scans (N=382)", None, "Primary Endpoint" if is_primary else "Secondary Endpoint"),
            ("biou", "All Tumour Scans (N=382)", None, "Primary Endpoint" if is_primary else "Secondary Endpoint"),
            ("lpr", "Overlap Cohort (N=193)", overlap_mask, "Primary Endpoint" if is_primary else "Secondary Endpoint"),
        ]:
            s1 = tumour_df[f"{a1}_{m}_mean"].to_numpy()
            s2 = tumour_df[f"{a2}_{m}_mean"].to_numpy()
            cids = tumour_df["pseudo_patient_cluster"].to_numpy()
            res = evaluate_cluster_inference(s1, s2, cids, vm, num_resamples=2000, seed=42)

            # Per-seed differences
            seed_diffs = []
            for s in seeds:
                df = dfs[s]
                sub = df.loc[is_overlap] if m == "lpr" else df.loc[is_tumour]
                val1 = sub[f"{a1}_{m}"].mean()
                val2 = sub[f"{a2}_{m}"].mean()
                seed_diffs.append(val1 - val2)

            res["comparison"] = comp_name
            res["metric"] = m if m != "biou" else "biou_d2"
            res["cohort"] = cohort_name
            res["hypothesis_family"] = family_name
            res["hypothesis_tier"] = tier
            res["is_primary_family"] = is_primary
            res["diff_seed42"] = seed_diffs[0]
            res["diff_seed1337"] = seed_diffs[1]
            res["diff_seed2026"] = seed_diffs[2]
            res["seed_mean"] = float(np.mean(seed_diffs))
            res["seed_sd"] = float(np.std(seed_diffs, ddof=1))
            all_results.append(res)

        # 2. Normal ovary metrics
        for m, cohort_name, tier in [
            ("clean_rej", "Normal Ovary Scans (N=87)", "Primary Safety Endpoint" if is_primary else "Secondary Safety Endpoint"),
            ("fpr_intra", "Normal Ovary Scans (N=87)", "Primary Safety Endpoint" if is_primary else "Secondary Safety Endpoint"),
            ("fpr_extra", "Normal Ovary Scans (N=87)", "Primary Safety Endpoint" if is_primary else "Secondary Safety Endpoint"),
        ]:
            s1 = normal_df[f"{a1}_{m}_mean"].to_numpy()
            s2 = normal_df[f"{a2}_{m}_mean"].to_numpy()
            cids = normal_df["pseudo_patient_cluster"].to_numpy()
            res = evaluate_cluster_inference(s1, s2, cids, None, num_resamples=2000, seed=42)

            seed_diffs = []
            for s in seeds:
                df = dfs[s]
                val1 = df.loc[is_normal, f"{a1}_{m}"].mean()
                val2 = df.loc[is_normal, f"{a2}_{m}"].mean()
                seed_diffs.append(val1 - val2)

            res["comparison"] = comp_name
            res["metric"] = m
            res["cohort"] = cohort_name
            res["hypothesis_family"] = family_name
            res["hypothesis_tier"] = tier
            res["is_primary_family"] = is_primary
            res["diff_seed42"] = seed_diffs[0]
            res["diff_seed1337"] = seed_diffs[1]
            res["diff_seed2026"] = seed_diffs[2]
            res["seed_mean"] = float(np.mean(seed_diffs))
            res["seed_sd"] = float(np.std(seed_diffs, ddof=1))
            all_results.append(res)

    results_df = pd.DataFrame(all_results)

    # Multiplicity corrections:
    # Rule 1: Compare adjusted p-values against alpha = 0.05
    # Rule 2: Alternatively compare raw p-values against alpha / M
    # Rule 3: NEVER compare adjusted p-values against alpha / M
    m_full = 21
    m_primary = 7
    alpha_nominal = 0.05
    alpha_bonf_primary = alpha_nominal / m_primary   # 0.007142857
    alpha_bonf_full = alpha_nominal / m_full         # 0.002380952

    # Arithmetic verification: p_adj = min(1.0, M * p_raw)
    results_df["p_bonf_full_21"] = np.clip(results_df["p_val_bca_inverted"] * m_full, 0.0, 1.0)
    results_df["p_bonf_primary_7"] = np.nan
    primary_mask = results_df["is_primary_family"]
    results_df.loc[primary_mask, "p_bonf_primary_7"] = np.clip(
        results_df.loc[primary_mask, "p_val_bca_inverted"] * m_primary, 0.0, 1.0
    )

    # Verification of decision consistency:
    # p_raw < alpha/M  <===>  p_adj < alpha (0.05)
    results_df["sig_nominal_05"] = results_df["p_val_bca_inverted"] < alpha_nominal
    results_df["sig_bonf_primary"] = results_df["p_bonf_primary_7"] < alpha_nominal
    results_df["sig_bonf_full"] = results_df["p_bonf_full_21"] < alpha_nominal

    # Add display formatted strings for manuscript tables
    results_df["p_val_bca_display"] = [format_p_value(p) for p in results_df["p_val_bca_inverted"]]
    results_df["p_val_null_display"] = [format_p_value(p) for p in results_df["p_val_null_centered"]]
    results_df["p_bonf_primary_display"] = [
        format_p_value(p) if not np.isnan(p) else "—" for p in results_df["p_bonf_primary_7"]
    ]
    results_df["p_bonf_full_display"] = [format_p_value(p) for p in results_df["p_bonf_full_21"]]

    # Interpretations with clear labelling
    interpretations = []
    for _, row in results_df.iterrows():
        comp = row["comparison"]
        m = row["metric"]
        sig_nom = row["sig_nominal_05"]
        sig_bonf_p = row["sig_bonf_primary"]
        sig_bonf_f = row["sig_bonf_full"]
        is_prim = row["is_primary_family"]

        if not sig_nom:
            interp = "Non-significant (95% BCa CI spans 0)."
        elif is_prim:
            if sig_bonf_p:
                interp = "Confirmed Significant (Survives primary Bonferroni M=7, p_adj < 0.05)."
            else:
                interp = "Secondary Nominally Significant (p_raw < 0.05, fails primary FWER control p_adj > 0.05)."
        else:
            if sig_bonf_f:
                interp = "Exploratory Significant (Survives full Bonferroni M=21, p_adj < 0.05)."
            else:
                interp = "Exploratory Nominally Significant (p_raw < 0.05, fails full FWER control p_adj > 0.05)."
        interpretations.append(interp)

    results_df["verdict"] = interpretations

    # Column ordering
    cols_order = [
        "comparison",
        "metric",
        "cohort",
        "hypothesis_family",
        "hypothesis_tier",
        "obs_diff",
        "bca_ci_lower",
        "bca_ci_upper",
        "diff_seed42",
        "diff_seed1337",
        "diff_seed2026",
        "seed_mean",
        "seed_sd",
        "se_jackknife",
        "se_bootstrap",
        "p_val_original",
        "p_val_bca_inverted",
        "p_val_bca_display",
        "p_val_null_centered",
        "p_val_null_display",
        "p_val_wald",
        "p_bonf_primary_7",
        "p_bonf_primary_display",
        "p_bonf_full_21",
        "p_bonf_full_display",
        "sig_nominal_05",
        "sig_bonf_primary",
        "sig_bonf_full",
        "z0_bias",
        "acceleration",
        "verdict",
    ]
    results_df = results_df[cols_order]

    # Save CSV
    out_csv = output_dir / "recalculated_inferential_results.csv"
    results_df.to_csv(out_csv, index=False)
    log_lines.append(f"Saved full recalculated inferential results to {out_csv} ({len(results_df)} rows).")

    # Save clean manuscript tables
    # 1. Primary family table
    primary_df = results_df[results_df["hypothesis_family"] == "Primary Confirmatory Family (M=7)"].copy()
    primary_out = output_dir / "manuscript_table_primary_family.csv"
    primary_df.to_csv(primary_out, index=False)
    log_lines.append(f"Saved primary family manuscript table to {primary_out} ({len(primary_df)} rows).")

    # 2. Full 21 comparisons table
    full_out = output_dir / "manuscript_table_full_21_comparisons.csv"
    results_df.to_csv(full_out, index=False)
    log_lines.append(f"Saved full 21 comparisons manuscript table to {full_out} ({len(results_df)} rows).")

    # Complete execution log
    log_lines.append(f"Inference verification completed successfully at {datetime.now(timezone.utc).isoformat()}")
    with open(output_dir / "verification_execution_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"Script finished. {len(results_df)} comparisons processed and verified.")


if __name__ == "__main__":
    main()
