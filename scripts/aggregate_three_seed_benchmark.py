#!/usr/bin/env python3
"""
OvaryVision — Three-Seed, Three-Arm Segmentation Benchmark Aggregator.

Consolidates training summaries, per-seed test evaluations, and produces
pooled multi-seed statistics, cluster bootstrap inference, and publication figures.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ovaryvision.statistics import paired_cluster_bootstrap_segmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] aggregate_benchmark: %(message)s",
)
logger = logging.getLogger("aggregate_benchmark")


def parse_args(args: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate 3-Seed 3-Arm Segmentation Benchmark Results")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("outputs/benchmark_runs/segmentation_benchmark"),
        help="Root directory containing benchmark run directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmark_runs/segmentation_benchmark/final_three_seed_benchmark"),
        help="Target directory for final aggregated benchmark deliverables",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 1337, 2026],
        help="Random seeds to aggregate",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
        help="Number of paired cluster bootstrap resamples for pooled analysis",
    )
    return parser.parse_args(args)


def collect_training_summaries(benchmark_dir: Path, seeds: List[int]) -> pd.DataFrame:
    """Collect run_summary.json and history.csv across all 9 runs."""
    arms = ["arm_a_raw", "arm_b_inpainted", "arm_c_masked"]
    records = []

    for seed in seeds:
        for arm in arms:
            run_dir = benchmark_dir / f"{arm}_seed{seed}"
            summary_file = run_dir / "run_summary.json"
            history_file = run_dir / "history.csv"

            if not summary_file.exists():
                logger.warning(f"Missing summary file: {summary_file}")
                continue

            with open(summary_file, "r") as f:
                s_data = json.load(f)

            train_time_total = np.nan
            if history_file.exists():
                h_df = pd.read_csv(history_file)
                if "train_time_sec" in h_df.columns:
                    train_time_total = float(h_df["train_time_sec"].sum())

            records.append({
                "arm": arm,
                "seed": seed,
                "architecture": s_data.get("architecture", "resnet34"),
                "resolution": s_data.get("resolution", 512),
                "batch_size": s_data.get("batch_size", 8),
                "epochs": s_data.get("epochs", 40),
                "best_epoch": s_data.get("best_epoch", 0),
                "best_val_loss": s_data.get("best_val_loss", np.nan),
                "total_train_time_sec": train_time_total,
                "total_train_time_min": round(train_time_total / 60.0, 2) if not np.isnan(train_time_total) else np.nan,
                "status": s_data.get("status", "UNKNOWN"),
            })

    return pd.DataFrame(records)


def collect_per_seed_evaluations(benchmark_dir: Path, seeds: List[int]) -> Dict[str, pd.DataFrame]:
    """Load per-scan metrics and per-seed summaries from test evaluations."""
    per_scan_dfs = {}
    tumour_summaries = []
    normal_summaries = []
    bootstrap_summaries = []

    for seed in seeds:
        eval_dir = benchmark_dir / f"seed{seed}_test_evaluation"
        per_scan_csv = eval_dir / f"seed{seed}_per_scan_test_metrics.csv"
        tumour_csv = eval_dir / f"seed{seed}_tumour_metrics_summary.csv"
        normal_csv = eval_dir / f"seed{seed}_normal_ovary_metrics_summary.csv"
        boot_csv = eval_dir / f"seed{seed}_bootstrap_inferential_report.csv"

        if per_scan_csv.exists():
            df = pd.read_csv(per_scan_csv)
            df["seed"] = seed
            per_scan_dfs[seed] = df
        else:
            logger.warning(f"Missing per-scan test metrics: {per_scan_csv}")

        if tumour_csv.exists():
            tdf = pd.read_csv(tumour_csv)
            tdf["seed"] = seed
            tumour_summaries.append(tdf)

        if normal_csv.exists():
            ndf = pd.read_csv(normal_csv)
            ndf["seed"] = seed
            normal_summaries.append(ndf)

        if boot_csv.exists():
            bdf = pd.read_csv(boot_csv)
            bdf["seed"] = seed
            bootstrap_summaries.append(bdf)

    return {
        "per_scan": per_scan_dfs,
        "tumour_summary": pd.concat(tumour_summaries, ignore_index=True) if tumour_summaries else pd.DataFrame(),
        "normal_summary": pd.concat(normal_summaries, ignore_index=True) if normal_summaries else pd.DataFrame(),
        "bootstrap_summary": pd.concat(bootstrap_summaries, ignore_index=True) if bootstrap_summaries else pd.DataFrame(),
    }


def compute_pooled_three_seed_metrics(
    per_scan_dfs: Dict[int, pd.DataFrame],
    bootstrap_iterations: int = 2000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute pooled multi-seed metrics and perform paired cluster bootstrap on scan-level averages.
    
    Guarantees:
    - Avoids treating 3*469 observations as independent (does not inflate sample size).
    - Uses scan-level average score across seeds m_i = (1/3) sum_s m_{i,s}.
    - Preserves cluster multiplicity across test clusters.
    - Separates between-seed variance from cluster bootstrap uncertainty.
    """
    seeds = sorted(list(per_scan_dfs.keys()))
    base_df = per_scan_dfs[seeds[0]].copy()
    num_scans = len(base_df)

    arms = ["arm_a", "arm_b", "arm_c"]
    metrics_tumour = ["dsc", "iou", "biou", "lpr"]
    metrics_normal = ["fpr_intra", "fpr_extra", "clean_rej"]

    is_normal = base_df["is_normal"].to_numpy().astype(bool)
    is_tumour = ~is_normal
    is_overlap = is_tumour & base_df["is_overlap"].to_numpy().astype(bool)

    # Compute scan-level mean across seeds cleanly per cohort subset
    for arm in arms:
        for m in ["dsc", "iou", "biou"]:
            col_name = f"{arm}_{m}"
            stacked = np.stack([per_scan_dfs[s][col_name].to_numpy() for s in seeds], axis=1)
            mean_vals = np.full(num_scans, np.nan)
            sd_vals = np.full(num_scans, np.nan)
            mean_vals[is_tumour] = np.mean(stacked[is_tumour], axis=1)
            sd_vals[is_tumour] = np.std(stacked[is_tumour], axis=1, ddof=1) if len(seeds) > 1 else 0.0
            base_df[f"{col_name}_mean"] = mean_vals
            base_df[f"{col_name}_sd"] = sd_vals

        # LPR: valid only on overlap cohort
        col_name = f"{arm}_lpr"
        stacked = np.stack([per_scan_dfs[s][col_name].to_numpy() for s in seeds], axis=1)
        mean_vals = np.full(num_scans, np.nan)
        sd_vals = np.full(num_scans, np.nan)
        mean_vals[is_overlap] = np.mean(stacked[is_overlap], axis=1)
        sd_vals[is_overlap] = np.std(stacked[is_overlap], axis=1, ddof=1) if len(seeds) > 1 else 0.0
        base_df[f"{col_name}_mean"] = mean_vals
        base_df[f"{col_name}_sd"] = sd_vals

        for m in metrics_normal:
            col_name = f"{arm}_{m}"
            stacked = np.stack([per_scan_dfs[s][col_name].to_numpy() for s in seeds], axis=1)
            mean_vals = np.full(num_scans, np.nan)
            sd_vals = np.full(num_scans, np.nan)
            mean_vals[is_normal] = np.mean(stacked[is_normal], axis=1)
            sd_vals[is_normal] = np.std(stacked[is_normal], axis=1, ddof=1) if len(seeds) > 1 else 0.0
            base_df[f"{col_name}_mean"] = mean_vals
            base_df[f"{col_name}_sd"] = sd_vals

    tumour_df = base_df[~base_df["is_normal"]].copy()
    overlap_df = tumour_df[tumour_df["is_overlap"]].copy()
    normal_df = base_df[base_df["is_normal"]].copy()

    # 1. Multi-seed Tumour Summary Table
    t_summary_rows = []
    for arm in arms:
        arm_label = "Arm A (Raw)" if arm == "arm_a" else ("Arm B (Inpainted)" if arm == "arm_b" else "Arm C (Masked)")
        d_mean = float(tumour_df[f"{arm}_dsc_mean"].mean())
        d_sd = float(tumour_df[f"{arm}_dsc_mean"].std())
        d_med = float(tumour_df[f"{arm}_dsc_mean"].median())
        d_iqr = float(tumour_df[f"{arm}_dsc_mean"].quantile(0.75) - tumour_df[f"{arm}_dsc_mean"].quantile(0.25))

        i_mean = float(tumour_df[f"{arm}_iou_mean"].mean())
        i_sd = float(tumour_df[f"{arm}_iou_mean"].std())
        i_med = float(tumour_df[f"{arm}_iou_mean"].median())
        i_iqr = float(tumour_df[f"{arm}_iou_mean"].quantile(0.75) - tumour_df[f"{arm}_iou_mean"].quantile(0.25))

        b_mean = float(tumour_df[f"{arm}_biou_mean"].mean())
        b_sd = float(tumour_df[f"{arm}_biou_mean"].std())
        b_med = float(tumour_df[f"{arm}_biou_mean"].median())
        b_iqr = float(tumour_df[f"{arm}_biou_mean"].quantile(0.75) - tumour_df[f"{arm}_biou_mean"].quantile(0.25))

        l_mean = float(overlap_df[f"{arm}_lpr_mean"].mean())
        l_sd = float(overlap_df[f"{arm}_lpr_mean"].std())
        l_med = float(overlap_df[f"{arm}_lpr_mean"].median())
        l_iqr = float(overlap_df[f"{arm}_lpr_mean"].quantile(0.75) - overlap_df[f"{arm}_lpr_mean"].quantile(0.25))

        t_summary_rows.append({
            "arm": arm_label,
            "n_tumour_scans": len(tumour_df),
            "dsc_mean": d_mean,
            "dsc_sd": d_sd,
            "dsc_median": d_med,
            "dsc_iqr": d_iqr,
            "iou_mean": i_mean,
            "iou_sd": i_sd,
            "iou_median": i_med,
            "iou_iqr": i_iqr,
            "biou_mean": b_mean,
            "biou_sd": b_sd,
            "biou_median": b_med,
            "biou_iqr": b_iqr,
            "n_overlap_scans": len(overlap_df),
            "lpr_mean": l_mean,
            "lpr_sd": l_sd,
            "lpr_median": l_med,
            "lpr_iqr": l_iqr,
        })
    pooled_tumour_summary = pd.DataFrame(t_summary_rows)

    # 2. Multi-seed Normal Ovary Summary Table
    n_summary_rows = []
    for arm in arms:
        arm_label = "Arm A (Raw)" if arm == "arm_a" else ("Arm B (Inpainted)" if arm == "arm_b" else "Arm C (Masked)")
        intra_m = float(normal_df[f"{arm}_fpr_intra_mean"].mean())
        intra_sd = float(normal_df[f"{arm}_fpr_intra_mean"].std())
        intra_med = float(normal_df[f"{arm}_fpr_intra_mean"].median())

        extra_m = float(normal_df[f"{arm}_fpr_extra_mean"].mean())
        extra_sd = float(normal_df[f"{arm}_fpr_extra_mean"].std())
        extra_med = float(normal_df[f"{arm}_fpr_extra_mean"].median())

        clean_rate = float(normal_df[f"{arm}_clean_rej_mean"].mean())
        clean_count = int(np.round(clean_rate * len(normal_df)))

        n_summary_rows.append({
            "arm": arm_label,
            "n_normal_scans": len(normal_df),
            "intra_ovary_fpr_mean": intra_m,
            "intra_ovary_fpr_sd": intra_sd,
            "intra_ovary_fpr_median": intra_med,
            "extra_ovary_fpr_mean": extra_m,
            "extra_ovary_fpr_sd": extra_sd,
            "extra_ovary_fpr_median": extra_med,
            "clean_rejection_rate": clean_rate,
            "clean_rejection_count": clean_count,
        })
    pooled_normal_summary = pd.DataFrame(n_summary_rows)

    # 3. Paired Cluster Bootstrap on Scan-Level Multi-Seed Averages
    logger.info("Computing pooled paired cluster bootstrap across 3-seed averaged scan predictions...")
    comparisons = [
        ("Arm C (Masked) vs Arm B (Inpainted)", "arm_c", "arm_b"),
        ("Arm C (Masked) vs Arm A (Raw)", "arm_c", "arm_a"),
        ("Arm B (Inpainted) vs Arm A (Raw)", "arm_b", "arm_a"),
    ]

    clusters_tumour = tumour_df["pseudo_patient_cluster"].to_numpy()
    overlap_mask_tumour = tumour_df["is_overlap"].to_numpy().astype(bool)
    clusters_normal = normal_df["pseudo_patient_cluster"].to_numpy()

    pooled_bootstrap_results = []
    for comp_name, a1, a2 in comparisons:
        for metric, valid_m in [
            ("dsc", None),
            ("iou", None),
            ("biou", None),
            ("lpr", overlap_mask_tumour),
        ]:
            col_1 = f"{a1}_{metric}_mean"
            col_2 = f"{a2}_{metric}_mean"
            s1 = tumour_df[col_1].to_numpy()
            s2 = tumour_df[col_2].to_numpy()

            res = paired_cluster_bootstrap_segmentation(
                scores_c=s1,
                scores_b=s2,
                cluster_ids=clusters_tumour,
                valid_mask=valid_m,
                metric_name=metric if metric != "biou" else "biou_d2",
                num_resamples=bootstrap_iterations,
                seed=42,
            )

            pooled_bootstrap_results.append({
                "comparison": comp_name,
                "metric": metric if metric != "biou" else "biou_d2",
                "cohort": "Overlap Cohort (N=193)" if metric == "lpr" else "All Tumour Scans (N=382)",
                "delta_mean": float(res.observed_difference),
                "ci_lower_bca": float(res.ci_lower_95),
                "ci_upper_bca": float(res.ci_upper_95),
                "ci_method": res.ci_method,
                "ci_lower_percentile": float(res.ci_lower_percentile_95),
                "ci_upper_percentile": float(res.ci_upper_percentile_95),
                "bca_fallback_used": res.ci_method != "bca",
                "z0_bias": float(res.bca_bias_correction_z0),
                "acceleration": float(res.bca_acceleration_a),
                "bootstrap_mean": float(res.bootstrap_mean),
                "bootstrap_std": float(res.bootstrap_std),
                "p_value": float(res.p_value),
                "num_clusters": int(res.num_clusters),
                "num_samples": int(res.num_samples),
                "empty_draws_rejected": int(res.empty_draws_count),
            })

    for comp_name, a1, a2 in comparisons:
        for metric in ["clean_rej", "fpr_intra", "fpr_extra"]:
            s1 = normal_df[f"{a1}_{metric}_mean"].to_numpy()
            s2 = normal_df[f"{a2}_{metric}_mean"].to_numpy()

            res = paired_cluster_bootstrap_segmentation(
                scores_c=s1,
                scores_b=s2,
                cluster_ids=clusters_normal,
                valid_mask=None,
                metric_name=metric,
                num_resamples=bootstrap_iterations,
                seed=42,
            )

            pooled_bootstrap_results.append({
                "comparison": comp_name,
                "metric": metric,
                "cohort": "Normal Ovary Scans (N=87)",
                "delta_mean": float(res.observed_difference),
                "ci_lower_bca": float(res.ci_lower_95),
                "ci_upper_bca": float(res.ci_upper_95),
                "ci_method": res.ci_method,
                "ci_lower_percentile": float(res.ci_lower_percentile_95),
                "ci_upper_percentile": float(res.ci_upper_percentile_95),
                "bca_fallback_used": res.ci_method != "bca",
                "z0_bias": float(res.bca_bias_correction_z0),
                "acceleration": float(res.bca_acceleration_a),
                "bootstrap_mean": float(res.bootstrap_mean),
                "bootstrap_std": float(res.bootstrap_std),
                "p_value": float(res.p_value),
                "num_clusters": int(res.num_clusters),
                "num_samples": int(res.num_samples),
                "empty_draws_rejected": int(res.empty_draws_count),
            })

    pooled_bootstrap_df = pd.DataFrame(pooled_bootstrap_results)
    return pooled_tumour_summary, pooled_normal_summary, pooled_bootstrap_df


def generate_pooled_plots(
    pooled_tumour: pd.DataFrame,
    pooled_normal: pd.DataFrame,
    pooled_bootstrap: pd.DataFrame,
    tumour_summary_per_seed: pd.DataFrame,
    normal_summary_per_seed: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate consolidated multi-seed publication figures."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#2b5c8f", "#d95f02", "#1b9e77"]  # Arm A: Blue, Arm B: Orange, Arm C: Green
    arm_labels = ["Arm A\n(Raw)", "Arm B\n(Inpainted)", "Arm C\n(Masked)"]

    # 1. Tumour Metrics Bar Chart with 3-Seed Std Error
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    metrics = [
        ("dsc", "Dice Similarity (DSC)", "dsc_mean"),
        ("iou", "IoU (Jaccard)", "iou_mean"),
        ("biou", "Boundary IoU (d=2)", "biou_mean"),
        ("lpr", "LPR (In-Caliper Sensitivity)", "lpr_mean"),
    ]

    for ax, (m_id, m_title, col_name) in zip(axes, metrics):
        means = pooled_tumour[col_name].to_numpy()
        # Compute inter-seed std dev across the 3 seeds
        seed_stds = []
        for arm in ["Arm A (Raw)", "Arm B (Inpainted)", "Arm C (Masked)"]:
            s_vals = tumour_summary_per_seed[tumour_summary_per_seed["arm"] == arm][col_name]
            seed_stds.append(float(s_vals.std()) if len(s_vals) > 1 else 0.0)

        bars = ax.bar(arm_labels, means, yerr=seed_stds, color=colors, alpha=0.75, width=0.5, capsize=4)
        ax.set_title(m_title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for bar, m_val, s_err in zip(bars, means, seed_stds):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + (s_err if not np.isnan(s_err) else 0.0) + 0.01,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle("OvaryVision 3-Seed Pooled: Tumour Segmentation Performance (Mean ± Inter-Seed SD)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "three_seed_tumour_metrics_comparison.png", dpi=300)
    plt.close()

    # 2. Normal Ovary Specificity & Rejection
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    norm_metrics = [
        ("intra_ovary_fpr_mean", "Intra-Ovarian FPR\n(Tumour on Normal Stroma)", "%"),
        ("extra_ovary_fpr_mean", "Extra-Ovarian FPR\n(Tumour outside Stroma)", "%"),
        ("clean_rejection_rate", "Clean Rejection Rate\n(Zero Predicted Tumour)", "%"),
    ]

    for ax, (col_name, n_title, unit) in zip(axes, norm_metrics):
        means = pooled_normal[col_name].to_numpy() * 100.0
        seed_stds = []
        for arm in ["Arm A (Raw)", "Arm B (Inpainted)", "Arm C (Masked)"]:
            s_vals = normal_summary_per_seed[normal_summary_per_seed["arm"] == arm][col_name] * 100.0
            seed_stds.append(float(s_vals.std()) if len(s_vals) > 1 else 0.0)

        bars = ax.bar(arm_labels, means, yerr=seed_stds, color=colors, alpha=0.75, width=0.5, capsize=4)
        ax.set_title(n_title, fontsize=10, fontweight="bold")
        ax.set_ylabel(f"Rate ({unit})")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for bar, m_val, s_err in zip(bars, means, seed_stds):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + (s_err if not np.isnan(s_err) else 0.0) + 0.5,
                f"{h:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle("OvaryVision 3-Seed Pooled: Normal Ovary False-Positive Rates (Mean ± Inter-Seed SD)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "three_seed_normal_ovary_comparison.png", dpi=300)
    plt.close()

    # 3. Forest Plot for Primary Comparison (Arm C vs Arm B)
    c_vs_b = pooled_bootstrap[pooled_bootstrap["comparison"] == "Arm C (Masked) vs Arm B (Inpainted)"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(c_vs_b))
    deltas = c_vs_b["delta_mean"].to_numpy()
    ci_low = c_vs_b["ci_lower_bca"].to_numpy()
    ci_high = c_vs_b["ci_upper_bca"].to_numpy()
    xerr = [deltas - ci_low, ci_high - deltas]

    ax.errorbar(deltas, y_pos, xerr=xerr, fmt="o", color="#1b9e77", ecolor="#1b9e77", elinewidth=2, capsize=4, markersize=6)
    ax.axvline(0.0, color="red", linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    labels = [f"{r['metric'].upper()} ({r['cohort']})" for _, r in c_vs_b.iterrows()]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Pooled Paired Difference Delta (Arm C - Arm B) with BCa 95% CI", fontweight="bold")
    ax.set_title("Pooled 3-Seed Primary Pre-Specified Comparison: Arm C (Masked) vs Arm B (Inpainted)", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plots_dir / "three_seed_bootstrap_forest_plot.png", dpi=300)
    plt.close()
    logger.info(f"Generated pooled diagnostic figures in {plots_dir}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== OvaryVision 3-Seed Benchmark Aggregator ===")
    logger.info(f"Benchmark Dir: {args.benchmark_dir}")
    logger.info(f"Output Dir:    {args.output_dir}")
    logger.info(f"Seeds:         {args.seeds}")

    # 1. Collect all 9 run training summaries
    train_summary_df = collect_training_summaries(args.benchmark_dir, args.seeds)
    train_summary_csv = args.output_dir / "nine_runs_training_summary.csv"
    train_summary_df.to_csv(train_summary_csv, index=False)
    logger.info(f"Saved 9-run training summary: {train_summary_csv} ({len(train_summary_df)} runs)")

    # 2. Collect per-seed evaluations
    eval_data = collect_per_seed_evaluations(args.benchmark_dir, args.seeds)
    if not eval_data["per_scan"]:
        logger.error("No per-scan test evaluations found. Ensure test evaluations are completed first.")
        return 1

    per_seed_tumour_csv = args.output_dir / "per_seed_tumour_metrics_summary.csv"
    eval_data["tumour_summary"].to_csv(per_seed_tumour_csv, index=False)

    per_seed_normal_csv = args.output_dir / "per_seed_normal_ovary_metrics_summary.csv"
    eval_data["normal_summary"].to_csv(per_seed_normal_csv, index=False)

    per_seed_boot_csv = args.output_dir / "per_seed_bootstrap_inferential_report.csv"
    eval_data["bootstrap_summary"].to_csv(per_seed_boot_csv, index=False)

    # 3. Compute Pooled Multi-Seed Metrics & Bootstrap
    pooled_tumour, pooled_normal, pooled_bootstrap = compute_pooled_three_seed_metrics(
        eval_data["per_scan"],
        bootstrap_iterations=args.bootstrap_iterations,
    )

    pooled_tumour_csv = args.output_dir / "three_seed_pooled_tumour_summary.csv"
    pooled_tumour.to_csv(pooled_tumour_csv, index=False)
    logger.info(f"Saved pooled tumour summary: {pooled_tumour_csv}")

    pooled_normal_csv = args.output_dir / "three_seed_pooled_normal_summary.csv"
    pooled_normal.to_csv(pooled_normal_csv, index=False)
    logger.info(f"Saved pooled normal summary: {pooled_normal_csv}")

    pooled_boot_csv = args.output_dir / "three_seed_pooled_inferential_report.csv"
    pooled_bootstrap.to_csv(pooled_boot_csv, index=False)
    logger.info(f"Saved pooled inferential report: {pooled_boot_csv}")

    # 4. Generate Diagnostic Figures
    try:
        generate_pooled_plots(
            pooled_tumour,
            pooled_normal,
            pooled_bootstrap,
            eval_data["tumour_summary"],
            eval_data["normal_summary"],
            args.output_dir,
        )
    except Exception as e:
        logger.warning(f"Could not generate pooled plots (matplotlib error): {e}")

    # 5. Master JSON Summary
    master_summary = {
        "benchmark_name": "OvaryVision Three-Seed Three-Arm Segmentation Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "seeds": args.seeds,
        "completed_runs_count": len(train_summary_df),
        "training_summary": train_summary_df.to_dict("records"),
        "pooled_tumour_summary": pooled_tumour.to_dict("records"),
        "pooled_normal_summary": pooled_normal.to_dict("records"),
        "pooled_bootstrap_results": pooled_bootstrap.to_dict("records"),
    }
    master_json_path = args.output_dir / "final_benchmark_summary.json"
    with open(master_json_path, "w") as f:
        json.dump(master_summary, f, indent=2)
    logger.info(f"Saved master JSON summary: {master_json_path}")

    logger.info("=== Three-Seed Benchmark Aggregation Successfully Finished ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

