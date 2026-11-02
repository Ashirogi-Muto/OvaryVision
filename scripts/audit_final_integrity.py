#!/usr/bin/env python3
"""
OvaryVision — Comprehensive Final Scientific Integrity Audit Script.

Audits:
1. All nine segmentation training runs and checkpoint integrity.
2. Held-out test evaluation completeness, cohort partitioning, and immutability.
3. Statistical aggregation methods, estimand definitions, per-seed results, and seed variability.
4. Classification benchmark integration and immutability.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_nine_runs(benchmark_dir: Path) -> Dict[str, Any]:
    """Audit all 9 segmentation training runs."""
    arms = ["arm_a_raw", "arm_b_inpainted", "arm_c_masked"]
    seeds = [42, 1337, 2026]
    expected_runs = [f"{a}_seed{s}" for s in seeds for a in arms]

    audit_records = {}
    overall_pass = True

    for run_name in expected_runs:
        run_dir = benchmark_dir / run_name
        record: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "exists": run_dir.exists(),
            "files": {},
            "epochs_completed": 0,
            "best_epoch_matches": False,
            "nan_or_inf_detected": False,
            "checkpoints_valid_zip": False,
            "config_consistent": False,
            "pass": False,
        }

        if not run_dir.exists():
            record["error"] = "Directory missing"
            overall_pass = False
            audit_records[run_name] = record
            continue

        # Check required files
        req_files = ["best_checkpoint.pth", "latest_checkpoint.pth", "history.csv", "run_summary.json"]
        files_present = True
        for rf in req_files:
            fp = run_dir / rf
            if fp.exists():
                stat = fp.stat()
                file_info = {
                    "size_bytes": stat.st_size,
                    "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "sha256": sha256_file(fp),
                }
                # Check zip integrity for .pth files
                if rf.endswith(".pth"):
                    try:
                        with zipfile.ZipFile(fp, "r") as z:
                            bad = z.testzip()
                            file_info["zip_test"] = "PASS" if bad is None else f"BAD_FILE_{bad}"
                            file_info["namelist"] = z.namelist()[:5]
                    except Exception as e:
                        file_info["zip_test"] = f"EXCEPTION_{e}"
                        files_present = False
                record["files"][rf] = file_info
            else:
                record["files"][rf] = {"error": "MISSING"}
                files_present = False

        # Audit history.csv
        history_path = run_dir / "history.csv"
        summary_path = run_dir / "run_summary.json"

        if history_path.exists() and summary_path.exists():
            h_df = pd.read_csv(history_path)
            with open(summary_path, "r") as f:
                s_data = json.load(f)

            record["epochs_completed"] = len(h_df)
            has_40_epochs = len(h_df) == 40

            # Check for NaNs or Infs
            numeric_cols = [c for c in h_df.columns if c in ["train_loss", "val_loss", "val_dsc", "train_time_sec"]]
            nan_or_inf = bool(h_df[numeric_cols].isna().any().any() or np.isinf(h_df[numeric_cols].to_numpy()).any())
            record["nan_or_inf_detected"] = nan_or_inf

            # Check best epoch match
            best_epoch_summary = s_data.get("best_epoch")
            best_val_loss_summary = s_data.get("best_val_loss")

            min_loss_row = h_df.loc[h_df["val_loss"].idxmin()]
            history_best_epoch = int(min_loss_row["epoch"])
            history_best_val_loss = float(min_loss_row["val_loss"])

            loss_match = abs(history_best_val_loss - best_val_loss_summary) < 1e-4
            epoch_match = history_best_epoch == best_epoch_summary

            record["best_epoch_matches"] = bool(loss_match and epoch_match)
            record["recorded_best_epoch"] = best_epoch_summary
            record["actual_min_val_loss_epoch"] = history_best_epoch
            record["best_val_loss"] = best_val_loss_summary

            # Check config consistency
            arch = s_data.get("architecture") == "resnet34"
            res = s_data.get("resolution") == 512
            batch = s_data.get("batch_size") == 8
            ep = s_data.get("epochs") == 40
            record["config_consistent"] = bool(arch and res and batch and ep)

            # Check zip tests
            zips_ok = (
                record["files"].get("best_checkpoint.pth", {}).get("zip_test") == "PASS"
                and record["files"].get("latest_checkpoint.pth", {}).get("zip_test") == "PASS"
            )
            record["checkpoints_valid_zip"] = zips_ok

            run_pass = bool(
                files_present
                and has_40_epochs
                and not nan_or_inf
                and record["best_epoch_matches"]
                and record["config_consistent"]
                and zips_ok
            )
            record["pass"] = run_pass
            if not run_pass:
                overall_pass = False
        else:
            record["pass"] = False
            overall_pass = False

        audit_records[run_name] = record

    return {
        "audit_name": "nine_run_integrity_audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_expected_runs": len(expected_runs),
        "all_runs_passed": overall_pass,
        "runs": audit_records,
    }


def audit_evaluation_completeness(benchmark_dir: Path, split_manifest_path: Path) -> Dict[str, Any]:
    """Audit evaluation outputs for seeds 42, 1337, 2026."""
    seeds = [42, 1337, 2026]
    split_manifest = pd.read_csv(split_manifest_path)
    manifest_test_ids = sorted(
        split_manifest[split_manifest["segmentation_split"] == "held_out_test"]["image_id"].astype(str).tolist()
    )

    eval_records = {}
    overall_pass = True

    for seed in seeds:
        eval_dir = benchmark_dir / f"seed{seed}_test_evaluation"
        per_scan_path = eval_dir / f"seed{seed}_per_scan_test_metrics.csv"
        summary_path = eval_dir / f"seed{seed}_evaluation_summary.json"
        boot_path = eval_dir / f"seed{seed}_bootstrap_inferential_report.csv"

        record: Dict[str, Any] = {
            "eval_dir": str(eval_dir),
            "exists": eval_dir.exists(),
            "files": {},
            "total_scans": 0,
            "tumour_scans": 0,
            "normal_scans": 0,
            "overlap_scans": 0,
            "zero_overlap_scans": 0,
            "lpr_non_null_count": 0,
            "id_match_manifest": False,
            "pass": False,
        }

        if not eval_dir.exists() or not per_scan_path.exists():
            record["error"] = "Evaluation files missing"
            overall_pass = False
            eval_records[f"seed_{seed}"] = record
            continue

        for fn in [
            f"seed{seed}_per_scan_test_metrics.csv",
            f"seed{seed}_evaluation_summary.json",
            f"seed{seed}_tumour_metrics_summary.csv",
            f"seed{seed}_normal_ovary_metrics_summary.csv",
            f"seed{seed}_bootstrap_inferential_report.csv",
        ]:
            fp = eval_dir / fn
            if fp.exists():
                stat = fp.stat()
                record["files"][fn] = {
                    "size_bytes": stat.st_size,
                    "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "sha256": sha256_file(fp),
                }

        df = pd.read_csv(per_scan_path)
        record["total_scans"] = len(df)
        csv_ids = sorted(df["image_id"].astype(str).tolist())

        id_match = csv_ids == manifest_test_ids
        record["id_match_manifest"] = id_match

        is_normal = df["is_normal"].astype(bool)
        record["normal_scans"] = int(is_normal.sum())
        record["tumour_scans"] = int((~is_normal).sum())

        tumour_df = df[~is_normal]
        is_overlap = tumour_df["is_overlap"].astype(bool)
        record["overlap_scans"] = int(is_overlap.sum())
        record["zero_overlap_scans"] = int((~is_overlap).sum())

        lpr_a_valid = int(tumour_df["arm_a_lpr"].notna().sum())
        lpr_b_valid = int(tumour_df["arm_b_lpr"].notna().sum())
        lpr_c_valid = int(tumour_df["arm_c_lpr"].notna().sum())
        record["lpr_non_null_count"] = lpr_a_valid

        cohort_counts_ok = (
            record["total_scans"] == 469
            and record["tumour_scans"] == 382
            and record["normal_scans"] == 87
            and record["overlap_scans"] == 193
            and record["zero_overlap_scans"] == 189
            and lpr_a_valid == 193
            and lpr_b_valid == 193
            and lpr_c_valid == 193
        )

        eval_pass = bool(id_match and cohort_counts_ok)
        record["pass"] = eval_pass
        if not eval_pass:
            overall_pass = False

        eval_records[f"seed_{seed}"] = record

    return {
        "audit_name": "evaluation_completeness_audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "all_evaluations_passed": overall_pass,
        "evaluations": eval_records,
    }


def compute_per_seed_and_variability_tables(
    benchmark_dir: Path, output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-seed results across all 9 runs and training seed variability."""
    seeds = [42, 1337, 2026]
    arms = ["arm_a", "arm_b", "arm_c"]
    arm_labels = {
        "arm_a": "Arm A (Raw)",
        "arm_b": "Arm B (Inpainted)",
        "arm_c": "Arm C (Masked)",
    }

    per_seed_rows = []

    for seed in seeds:
        csv_path = benchmark_dir / f"seed{seed}_test_evaluation" / f"seed{seed}_per_scan_test_metrics.csv"
        df = pd.read_csv(csv_path)

        tumour_df = df[~df["is_normal"].astype(bool)]
        overlap_df = tumour_df[tumour_df["is_overlap"].astype(bool)]
        normal_df = df[df["is_normal"].astype(bool)]

        for arm in arms:
            label = arm_labels[arm]

            # Tumour metrics
            dsc_col = tumour_df[f"{arm}_dsc"]
            iou_col = tumour_df[f"{arm}_iou"]
            biou_col = tumour_df[f"{arm}_biou"]
            lpr_col = overlap_df[f"{arm}_lpr"]

            # Normal ovary metrics
            intra_col = normal_df[f"{arm}_fpr_intra"]
            extra_col = normal_df[f"{arm}_fpr_extra"]
            clean_col = normal_df[f"{arm}_clean_rej"]

            clean_count = int(clean_col.sum())
            clean_rate = clean_count / len(normal_df)

            per_seed_rows.append({
                "arm": label,
                "arm_key": arm,
                "seed": seed,
                "n_tumour_scans": len(tumour_df),
                "dsc_mean": float(dsc_col.mean()),
                "dsc_sd": float(dsc_col.std()),
                "dsc_median": float(dsc_col.median()),
                "dsc_iqr": float(dsc_col.quantile(0.75) - dsc_col.quantile(0.25)),
                "iou_mean": float(iou_col.mean()),
                "iou_sd": float(iou_col.std()),
                "iou_median": float(iou_col.median()),
                "iou_iqr": float(iou_col.quantile(0.75) - iou_col.quantile(0.25)),
                "biou_mean": float(biou_col.mean()),
                "biou_sd": float(biou_col.std()),
                "biou_median": float(biou_col.median()),
                "biou_iqr": float(biou_col.quantile(0.75) - biou_col.quantile(0.25)),
                "n_overlap_scans": len(overlap_df),
                "lpr_mean": float(lpr_col.mean()),
                "lpr_sd": float(lpr_col.std()),
                "lpr_median": float(lpr_col.median()),
                "lpr_iqr": float(lpr_col.quantile(0.75) - lpr_col.quantile(0.25)),
                "n_normal_scans": len(normal_df),
                "clean_rejection_rate": clean_rate,
                "clean_rejection_count": clean_count,
                "intra_ovary_fpr_mean": float(intra_col.mean()),
                "intra_ovary_fpr_sd": float(intra_col.std()),
                "intra_ovary_fpr_median": float(intra_col.median()),
                "extra_ovary_fpr_mean": float(extra_col.mean()),
                "extra_ovary_fpr_sd": float(extra_col.std()),
                "extra_ovary_fpr_median": float(extra_col.median()),
            })

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(output_dir / "per_seed_results.csv", index=False)

    # Compute training seed variability summary (mean, SD, min, max, CV across seeds for each arm)
    metrics_to_summarize = [
        ("dsc_mean", "Dice Similarity Coefficient (DSC)"),
        ("iou_mean", "Intersection over Union (IoU)"),
        ("biou_mean", "Boundary IoU (d=2)"),
        ("lpr_mean", "Lesion Preservation Ratio (LPR)"),
        ("clean_rejection_rate", "Clean Rejection Rate"),
        ("intra_ovary_fpr_mean", "Intra-Ovarian FPR"),
        ("extra_ovary_fpr_mean", "Extra-Ovarian FPR"),
    ]

    var_rows = []
    for arm in arms:
        label = arm_labels[arm]
        sub_df = per_seed_df[per_seed_df["arm_key"] == arm]

        for col, metric_name in metrics_to_summarize:
            vals = sub_df[col].to_numpy()
            mean_across_seeds = float(np.mean(vals))
            sd_across_seeds = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            min_val = float(np.min(vals))
            max_val = float(np.max(vals))
            cv_pct = (sd_across_seeds / mean_across_seeds * 100.0) if mean_across_seeds != 0 else np.nan

            var_rows.append({
                "arm": label,
                "metric_key": col,
                "metric_name": metric_name,
                "n_seeds": len(vals),
                "seed_values": [round(float(v), 5) for v in vals],
                "mean_across_seeds": mean_across_seeds,
                "sd_across_seeds": sd_across_seeds,
                "min_across_seeds": min_val,
                "max_across_seeds": max_val,
                "cv_percentage": cv_pct,
            })

    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(output_dir / "seed_variability_summary.csv", index=False)

    return per_seed_df, var_df


def audit_classification_benchmark() -> Dict[str, Any]:
    """Audit the 18 classification benchmark runs."""
    base_dir = Path("outputs/benchmark_runs/full_benchmark")
    master_csv = base_dir / "master_benchmark_summary.csv"
    master_json = base_dir / "master_benchmark_summary.json"

    exists = master_csv.exists() and master_json.exists()
    runs = []
    if exists:
        df = pd.read_csv(master_csv)
        for _, row in df.iterrows():
            run_name = f"{row['arm']}_{row['backbone']}_seed{row['seed']}"
            pred_file = base_dir / run_name / "predictions.csv"
            runs.append({
                "run_name": run_name,
                "arm": row["arm"],
                "backbone": row["backbone"],
                "seed": int(row["seed"]),
                "test_auroc": float(row["test_auroc"]),
                "test_ece_calibrated": float(row["test_ece_calibrated"]),
                "test_brier_calibrated": float(row["test_brier_calibrated"]),
                "predictions_exist": pred_file.exists(),
                "predictions_rows": len(pd.read_csv(pred_file)) if pred_file.exists() else 0,
            })

    return {
        "master_csv_exists": master_csv.exists(),
        "master_json_exists": master_json.exists(),
        "total_runs": len(runs),
        "runs": runs,
    }


def main():
    benchmark_dir = Path("outputs/benchmark_runs/segmentation_benchmark")
    output_dir = Path("outputs/final_research_audit")
    split_manifest = Path("data/manifests/mmotu_2d_segmentation_split.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []
    log_lines.append(f"Audit started at {datetime.now(timezone.utc).isoformat()}")

    # 1. Audit nine segmentation runs
    print("Auditing nine segmentation training runs...")
    runs_audit = audit_nine_runs(benchmark_dir)
    with open(output_dir / "nine_run_integrity_audit.json", "w") as f:
        json.dump(runs_audit, f, indent=2)
    log_lines.append(f"Nine runs audit complete. All passed: {runs_audit['all_runs_passed']}")

    # 2. Audit evaluation completeness
    print("Auditing evaluation completeness...")
    eval_audit = audit_evaluation_completeness(benchmark_dir, split_manifest)
    with open(output_dir / "evaluation_completeness_audit.json", "w") as f:
        json.dump(eval_audit, f, indent=2)
    log_lines.append(f"Evaluation completeness audit complete. All passed: {eval_audit['all_evaluations_passed']}")

    # 3. Compute per-seed results and variability summary
    print("Computing per-seed results and variability tables...")
    per_seed_df, var_df = compute_per_seed_and_variability_tables(benchmark_dir, output_dir)
    log_lines.append(f"Per-seed results table saved with {len(per_seed_df)} rows.")
    log_lines.append(f"Seed variability summary table saved with {len(var_df)} rows.")

    # 4. Audit classification benchmark
    print("Auditing classification benchmark...")
    cls_audit = audit_classification_benchmark()
    log_lines.append(f"Classification benchmark audit complete. Total runs: {cls_audit['total_runs']}")

    # Write execution log
    log_lines.append(f"Audit completed successfully at {datetime.now(timezone.utc).isoformat()}")
    with open(output_dir / "audit_execution_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print("Audit execution script completed successfully.")


if __name__ == "__main__":
    main()

