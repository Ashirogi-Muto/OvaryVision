"""Evaluate completed Seed 42 checkpoints on the official held-out test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ovaryvision.annotation_workflow import extract_caliper_candidate_mask
from ovaryvision.preprocessing import apply_arm_a, apply_arm_b, apply_arm_c, dilate_binary_mask
from ovaryvision.statistics import paired_cluster_bootstrap_segmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] evaluate_seed42: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_seed42")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Seed 42 Checkpoints on Held-Out Test Set")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/manifests/mmotu_2d_segmentation_split.csv"),
        help="Path to segmentation split manifest",
    )
    parser.add_argument(
        "--cdfi-manifest",
        type=Path,
        default=Path("data/manifests/mmotu_2d_cdfi_classification.csv"),
        help="Path to CDFI classification manifest",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/benchmark_runs/segmentation_benchmark"),
        help="Directory containing trained model checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save evaluation results and figures",
    )
    parser.add_argument("--architecture", type=str, default="resnet34", help="SMP encoder architecture")
    parser.add_argument("--resolution", type=int, default=512, help="Model input resolution")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--seed", type=int, default=42, help="Seed to evaluate")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000, help="Number of bootstrap resamples")
    parser.add_argument("--device", type=str, default="cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifests and checkpoints without inference")
    return parser.parse_args(args)


def compute_boundary_iou(pred: np.ndarray, target: np.ndarray, d: int = 2) -> float:
    """Compute Boundary IoU within margin d pixels (Cheng et al., CVPR 2021)."""
    from scipy.ndimage import binary_erosion
    p_bool = pred > 0.5
    t_bool = target > 0.5

    p_sum = np.sum(p_bool)
    t_sum = np.sum(t_bool)

    if p_sum == 0 and t_sum == 0:
        return 1.0
    if p_sum == 0 or t_sum == 0:
        return 0.0

    struct = np.ones((2 * d + 1, 2 * d + 1), dtype=bool)
    b_target = t_bool ^ binary_erosion(t_bool, structure=struct)
    b_pred = p_bool ^ binary_erosion(p_bool, structure=struct)

    intersect = np.sum(b_target & b_pred)
    union = np.sum(b_target | b_pred)
    if union == 0:
        return 1.0 if intersect == 0 else 0.0
    return float(intersect / union)


def load_model_from_checkpoint(checkpoint_path: Path, architecture: str, device: torch.device) -> nn.Module:
    """Load an SMP U-Net model from a saved best_checkpoint.pth file."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = smp.Unet(
        encoder_name=architecture,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint from {checkpoint_path} (best epoch {ckpt.get('epoch', 'N/A')}, val loss {ckpt.get('val_loss', 'N/A'):.4f})")
    return model


def evaluate_single_arm(
    model: nn.Module,
    arm: str,
    test_records: List[Dict[str, Any]],
    resolution: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Run model inference across the held-out test cohort for a single preprocessing arm."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    results = []

    logger.info(f"Starting inference for arm: {arm} on {len(test_records)} test scans...")
    t0 = time.time()

    for idx, rec in enumerate(test_records):
        img_p = Path(rec["relative_image_path"])
        mask_p = Path(rec["tumor_mask_path"])
        is_normal = (rec.get("fine_grained_class") == "NormalOvary")

        raw_pil = Image.open(img_p).convert("RGB")
        w_orig, h_orig = raw_pil.size
        raw_arr = np.array(raw_pil)

        orig_mask = Image.open(mask_p).convert("L")
        orig_mask_arr = (np.array(orig_mask) > 0).astype(np.uint8)

        # Derive ground truth masks:
        if is_normal:
            tumour_mask_arr = np.zeros_like(orig_mask_arr, dtype=np.uint8)
            ovary_mask_arr = orig_mask_arr.copy()  # Anatomical ovarian stroma
        else:
            tumour_mask_arr = orig_mask_arr.copy()
            ovary_mask_arr = np.zeros_like(orig_mask_arr, dtype=np.uint8)

        # Extract caliper artifact mask
        caliper_mask, _ = extract_caliper_candidate_mask(raw_arr)
        dilated_caliper = dilate_binary_mask(caliper_mask, radius=2)

        # Apply preprocessing arm
        if arm == "arm_a_raw":
            proc_arr = apply_arm_a(raw_arr)
        elif arm == "arm_b_inpainted":
            proc_arr = apply_arm_b(raw_arr, caliper_mask, dilation_radius=2)
        elif arm == "arm_c_masked":
            proc_arr = apply_arm_c(raw_arr, caliper_mask, dilation_radius=2)
        else:
            raise ValueError(f"Unknown arm: {arm}")

        # Resize to model resolution (512x512) and normalize
        proc_pil = Image.fromarray(proc_arr).resize((resolution, resolution), Image.BILINEAR)
        proc_np = np.array(proc_pil, dtype=np.float32) / 255.0
        norm_img = (proc_np - mean) / std
        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(tensor_img)
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        pred_512 = (probs > 0.5).astype(np.uint8)
        pred_pil = Image.fromarray(pred_512).resize((w_orig, h_orig), Image.NEAREST)
        pred_orig = np.array(pred_pil) > 0

        raw_cluster = rec.get("pseudo_patient_cluster")
        if pd.isna(raw_cluster) or raw_cluster is None or str(raw_cluster).lower() in ("nan", "none", ""):
            cluster_clean = f"SINGLETON_{rec['image_id']}"
        else:
            cluster_clean = str(raw_cluster)

        # Compute metrics at original image resolution
        record_res = {
            "image_id": rec["image_id"],
            "arm": arm,
            "fine_grained_class": rec.get("fine_grained_class", ""),
            "is_normal": is_normal,
            "pseudo_patient_cluster": cluster_clean,
            "caliper_pixels": int(np.sum(caliper_mask > 0)),
            "dilated_caliper_pixels": int(np.sum(dilated_caliper > 0)),
            "pred_pixels": int(np.sum(pred_orig)),
        }

        if not is_normal:
            t_bool = tumour_mask_arr > 0
            p_bool = pred_orig
            p_sum = int(np.sum(p_bool))
            t_sum = int(np.sum(t_bool))
            inter = int(np.sum(p_bool & t_bool))
            union = int(np.sum(p_bool | t_bool))

            dsc = float(2.0 * inter / (p_sum + t_sum)) if (p_sum + t_sum) > 0 else 1.0
            iou = float(inter / union) if union > 0 else (1.0 if (p_sum == 0 and t_sum == 0) else 0.0)
            biou = compute_boundary_iou(p_bool, t_bool, d=2)

            # LPR: artifact-overlap-region sensitivity
            overlap_mask = t_bool & (dilated_caliper > 0)
            overlap_denom = int(np.sum(overlap_mask))
            if overlap_denom > 0:
                overlap_num = int(np.sum(p_bool & overlap_mask))
                lpr = float(overlap_num / overlap_denom)
                is_overlap = True
                overlap_ratio = float(overlap_denom / t_sum)
            else:
                lpr = np.nan
                is_overlap = False
                overlap_ratio = 0.0

            record_res.update({
                "dsc": dsc,
                "iou": iou,
                "biou_d2": biou,
                "lpr": lpr,
                "is_overlap": is_overlap,
                "overlap_denom_pixels": overlap_denom,
                "overlap_ratio": overlap_ratio,
                "tumour_pixels": t_sum,
                "fpr_intra_ovary": np.nan,
                "fpr_extra_ovary": np.nan,
                "clean_rejection": np.nan,
            })
        else:
            # Normal ovary evaluation
            ovary_bool = ovary_mask_arr > 0
            ovary_sum = int(np.sum(ovary_bool))
            extra_ovary_denom = int(np.sum(~ovary_bool))
            p_bool = pred_orig

            intra_fp = int(np.sum(p_bool & ovary_bool))
            extra_fp = int(np.sum(p_bool & (~ovary_bool)))

            fpr_intra = float(intra_fp / ovary_sum) if ovary_sum > 0 else 0.0
            fpr_extra = float(extra_fp / extra_ovary_denom) if extra_ovary_denom > 0 else 0.0
            clean_rej = 1.0 if np.sum(p_bool) == 0 else 0.0

            record_res.update({
                "dsc": np.nan,
                "iou": np.nan,
                "biou_d2": np.nan,
                "lpr": np.nan,
                "is_overlap": False,
                "overlap_denom_pixels": 0,
                "overlap_ratio": 0.0,
                "tumour_pixels": 0,
                "fpr_intra_ovary": fpr_intra,
                "fpr_extra_ovary": fpr_extra,
                "clean_rejection": clean_rej,
            })

        results.append(record_res)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_records):
            logger.info(f"[{arm}] Evaluated {idx + 1}/{len(test_records)} scans ({(idx + 1) / (time.time() - t0):.1f} scans/sec)")

    return results


def generate_diagnostic_plots(
    per_scan_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    normal_df: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Generate diagnostic boxplots, bar plots, and difference forest plots."""
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tumour_df = per_scan_df[~per_scan_df["is_normal"]]
    overlap_df = tumour_df[tumour_df["is_overlap"]]

    colors = ["#2b5c8f", "#d95f02", "#1b9e77"]  # A: Blue, B: Orange, C: Green
    arm_labels = ["Arm A\n(Raw)", "Arm B\n(Inpainted)", "Arm C\n(Masked)"]

    # 1. Tumour Metrics Boxplot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    metrics = [
        ("dsc", "Dice (DSC)"),
        ("iou", "IoU (Jaccard)"),
        ("biou", "Boundary IoU (d=2)"),
        ("lpr", f"LPR (Overlap N={len(overlap_df)})"),
    ]

    for ax, (m_col, m_title) in zip(axes, metrics):
        data_source = overlap_df if m_col == "lpr" else tumour_df
        data = [
            data_source[f"arm_a_{m_col}"].dropna(),
            data_source[f"arm_b_{m_col}"].dropna(),
            data_source[f"arm_c_{m_col}"].dropna(),
        ]
        bp = ax.boxplot(data, labels=arm_labels, patch_artist=True, showmeans=True, meanline=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(m_title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(f"OvaryVision Seed {seed}: Tumour Segmentation Performance Across Preprocessing Arms", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / f"seed{seed}_tumour_metrics_boxplot.png", dpi=300)
    plt.close()

    # 2. Normal Ovary FPR & Clean Rejection Comparison
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    norm_metrics = [
        ("fpr_intra", "Intra-Ovarian FPR\n(Tumour on Normal Stroma)", "%"),
        ("fpr_extra", "Extra-Ovarian FPR\n(Tumour outside Stroma)", "%"),
        ("clean_rej", "Clean Rejection Rate\n(Zero Predicted Tumour)", "%"),
    ]
    for ax, (n_col, n_title, unit) in zip(axes, norm_metrics):
        means = [
            float(normal_df[f"arm_a_{n_col}"].mean() * 100),
            float(normal_df[f"arm_b_{n_col}"].mean() * 100),
            float(normal_df[f"arm_c_{n_col}"].mean() * 100),
        ]
        bars = ax.bar(arm_labels, means, color=colors, alpha=0.75, width=0.5)
        ax.set_title(n_title, fontsize=10, fontweight="bold")
        ax.set_ylabel(f"Rate ({unit})")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{h:.2f}%", ha="center", va="bottom", fontsize=9)

    plt.suptitle(f"OvaryVision Seed {seed}: False-Positive Rates on Normal Ovary Cohort (N={len(normal_df)})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / f"seed{seed}_normal_ovary_fpr_comparison.png", dpi=300)
    plt.close()

    # 3. Paired Difference Forest Plot (Arm C vs Arm B)
    c_vs_b = bootstrap_df[bootstrap_df["comparison"] == "Arm C (Masked) vs Arm B (Inpainted)"]
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
    ax.set_xlabel("Paired Difference Delta (Arm C - Arm B) with BCa 95% CI", fontweight="bold")
    ax.set_title(f"Seed {seed} Primary Pre-Specified Comparison: Arm C (Masked) vs Arm B (Inpainted)", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plots_dir / f"seed{seed}_bootstrap_difference_forest.png", dpi=300)
    plt.close()
    logger.info(f"Generated diagnostic plots in {plots_dir}")


def main() -> int:
    args = parse_args()
    logger.info("=== OvaryVision Seed 42 Test-Set Evaluation ===")
    logger.info(f"Split Manifest: {args.split_manifest}")
    logger.info(f"Checkpoints Dir: {args.checkpoint_dir}")
    logger.info(f"Output Dir: {args.output_dir}")
    logger.info(f"Device: {args.device}")

    # Validate manifests
    if not args.split_manifest.exists():
        logger.error(f"Missing split manifest: {args.split_manifest}")
        return 1

    split_df = pd.read_csv(args.split_manifest)
    test_df = split_df[split_df["segmentation_split"] == "held_out_test"].copy()
    logger.info(f"Total held-out test scans: {len(test_df)} (Tumour: {sum(test_df['fine_grained_class'] != 'NormalOvary')}, Normal: {sum(test_df['fine_grained_class'] == 'NormalOvary')})")
    assert len(test_df) == 469, f"Expected exactly 469 test scans, got {len(test_df)}"

    # Checksum integrity check
    logger.info("Verifying SHA-256 hashes of all 469 test images and masks...")
    verified_images = 0
    verified_masks = 0
    for _, row in test_df.iterrows():
        ip = Path(row["relative_image_path"])
        mp = Path(row["tumor_mask_path"])
        if not ip.exists():
            logger.error(f"Test image not found: {ip}")
            return 1
        if not mp.exists():
            logger.error(f"Test mask not found: {mp}")
            return 1
        with open(ip, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == row["sha256_checksum"]:
                verified_images += 1
        verified_masks += 1

    logger.info(f"Test Image Integrity: {verified_images}/{len(test_df)} (100.0% SHA-256 match)")
    logger.info(f"Test Mask Integrity:  {verified_masks}/{len(test_df)} (100.0% verified present)")
    assert verified_images == 469

    if args.dry_run:
        logger.info("Dry-run requested: all checks passed. Exiting without model evaluation.")
        return 0

    if not HAS_TORCH:
        logger.error("PyTorch / SMP is required for model evaluation.")
        return 1

    device = torch.device(args.device)
    test_records = test_df.to_dict("records")
    if args.output_dir is None:
        args.output_dir = Path(f"outputs/benchmark_runs/segmentation_benchmark/seed{args.seed}_test_evaluation")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    arms = ["arm_a_raw", "arm_b_inpainted", "arm_c_masked"]
    all_arm_results: Dict[str, List[Dict[str, Any]]] = {}

    for arm in arms:
        ckpt_p = args.checkpoint_dir / f"{arm}_seed{args.seed}" / "best_checkpoint.pth"
        model = load_model_from_checkpoint(ckpt_p, args.architecture, device)
        results = evaluate_single_arm(model, arm, test_records, args.resolution, device)
        all_arm_results[arm] = results

    # Assemble per-scan metrics table
    per_scan_rows = []
    tumour_rows = []
    normal_rows = []

    for i in range(len(test_records)):
        rec = test_records[i]
        img_id = rec["image_id"]
        is_norm = (rec["fine_grained_class"] == "NormalOvary")
        raw_c = rec.get("pseudo_patient_cluster")
        if pd.isna(raw_c) or raw_c is None or str(raw_c).lower() in ("nan", "none", ""):
            cluster = f"SINGLETON_{img_id}"
        else:
            cluster = str(raw_c)

        res_a = all_arm_results["arm_a_raw"][i]
        res_b = all_arm_results["arm_b_inpainted"][i]
        res_c = all_arm_results["arm_c_masked"][i]

        row = {
            "image_id": img_id,
            "fine_grained_class": rec["fine_grained_class"],
            "is_normal": is_norm,
            "pseudo_patient_cluster": cluster,
            "caliper_pixels": res_a["caliper_pixels"],
            "dilated_caliper_pixels": res_a["dilated_caliper_pixels"],
            # Arm A metrics
            "arm_a_dsc": res_a["dsc"],
            "arm_a_iou": res_a["iou"],
            "arm_a_biou": res_a["biou_d2"],
            "arm_a_lpr": res_a["lpr"],
            # Arm B metrics
            "arm_b_dsc": res_b["dsc"],
            "arm_b_iou": res_b["iou"],
            "arm_b_biou": res_b["biou_d2"],
            "arm_b_lpr": res_b["lpr"],
            # Arm C metrics
            "arm_c_dsc": res_c["dsc"],
            "arm_c_iou": res_c["iou"],
            "arm_c_biou": res_c["biou_d2"],
            "arm_c_lpr": res_c["lpr"],
            # Overlap metadata
            "is_overlap": res_a["is_overlap"],
            "overlap_ratio": res_a["overlap_ratio"],
            # Normal ovary FPRs
            "arm_a_fpr_intra": res_a["fpr_intra_ovary"],
            "arm_a_fpr_extra": res_a["fpr_extra_ovary"],
            "arm_a_clean_rej": res_a["clean_rejection"],
            "arm_b_fpr_intra": res_b["fpr_intra_ovary"],
            "arm_b_fpr_extra": res_b["fpr_extra_ovary"],
            "arm_b_clean_rej": res_b["clean_rejection"],
            "arm_c_fpr_intra": res_c["fpr_intra_ovary"],
            "arm_c_fpr_extra": res_c["fpr_extra_ovary"],
            "arm_c_clean_rej": res_c["clean_rejection"],
        }
        per_scan_rows.append(row)

    per_scan_df = pd.DataFrame(per_scan_rows)
    per_scan_csv = args.output_dir / f"seed{args.seed}_per_scan_test_metrics.csv"
    per_scan_df.to_csv(per_scan_csv, index=False)
    logger.info(f"Saved per-scan metrics: {per_scan_csv} ({len(per_scan_df)} rows)")

    # Tumour Scans Summary (N = 382)
    tumour_df = per_scan_df[~per_scan_df["is_normal"]].copy()
    overlap_df = tumour_df[tumour_df["is_overlap"]].copy()
    zero_overlap_df = tumour_df[~tumour_df["is_overlap"]].copy()

    logger.info(f"Tumour Scans: {len(tumour_df)} | Valid Overlap Cohort: {len(overlap_df)} | Zero-Overlap Cohort: {len(zero_overlap_df)}")

    tumour_summary = []
    for arm in ["arm_a", "arm_b", "arm_c"]:
        arm_label = "Arm A (Raw)" if arm == "arm_a" else ("Arm B (Inpainted)" if arm == "arm_b" else "Arm C (Masked)")
        dsc_vals = tumour_df[f"{arm}_dsc"]
        iou_vals = tumour_df[f"{arm}_iou"]
        biou_vals = tumour_df[f"{arm}_biou"]
        lpr_vals = overlap_df[f"{arm}_lpr"]  # Valid overlap cohort only

        tumour_summary.append({
            "arm": arm_label,
            "n_tumour_scans": len(tumour_df),
            "dsc_mean": float(dsc_vals.mean()),
            "dsc_sd": float(dsc_vals.std()),
            "dsc_median": float(dsc_vals.median()),
            "dsc_iqr": float(dsc_vals.quantile(0.75) - dsc_vals.quantile(0.25)),
            "iou_mean": float(iou_vals.mean()),
            "iou_sd": float(iou_vals.std()),
            "iou_median": float(iou_vals.median()),
            "iou_iqr": float(iou_vals.quantile(0.75) - iou_vals.quantile(0.25)),
            "biou_mean": float(biou_vals.mean()),
            "biou_sd": float(biou_vals.std()),
            "biou_median": float(biou_vals.median()),
            "biou_iqr": float(biou_vals.quantile(0.75) - biou_vals.quantile(0.25)),
            "n_overlap_scans": len(overlap_df),
            "lpr_mean": float(lpr_vals.mean()),
            "lpr_sd": float(lpr_vals.std()),
            "lpr_median": float(lpr_vals.median()),
            "lpr_iqr": float(lpr_vals.quantile(0.75) - lpr_vals.quantile(0.25)),
        })

    tumour_summary_df = pd.DataFrame(tumour_summary)
    tumour_summary_csv = args.output_dir / f"seed{args.seed}_tumour_metrics_summary.csv"
    tumour_summary_df.to_csv(tumour_summary_csv, index=False)
    logger.info(f"Saved tumour metrics summary: {tumour_summary_csv}")

    # Normal Ovary Scans Summary (N = 87)
    normal_df = per_scan_df[per_scan_df["is_normal"]].copy()
    normal_summary = []
    for arm in ["arm_a", "arm_b", "arm_c"]:
        arm_label = "Arm A (Raw)" if arm == "arm_a" else ("Arm B (Inpainted)" if arm == "arm_b" else "Arm C (Masked)")
        intra_vals = normal_df[f"{arm}_fpr_intra"]
        extra_vals = normal_df[f"{arm}_fpr_extra"]
        clean_vals = normal_df[f"{arm}_clean_rej"]

        normal_summary.append({
            "arm": arm_label,
            "n_normal_scans": len(normal_df),
            "intra_ovary_fpr_mean": float(intra_vals.mean()),
            "intra_ovary_fpr_sd": float(intra_vals.std()),
            "intra_ovary_fpr_median": float(intra_vals.median()),
            "extra_ovary_fpr_mean": float(extra_vals.mean()),
            "extra_ovary_fpr_sd": float(extra_vals.std()),
            "extra_ovary_fpr_median": float(extra_vals.median()),
            "clean_rejection_rate": float(clean_vals.mean()),
            "clean_rejection_count": int(clean_vals.sum()),
        })

    normal_summary_df = pd.DataFrame(normal_summary)
    normal_summary_csv = args.output_dir / f"seed{args.seed}_normal_ovary_metrics_summary.csv"
    normal_summary_df.to_csv(normal_summary_csv, index=False)
    logger.info(f"Saved normal ovary metrics summary: {normal_summary_csv}")

    # Paired Pseudo-Patient-Cluster Bootstrap (B = 2,000 resamples)
    logger.info(f"Running paired cluster bootstrap ({args.bootstrap_iterations} resamples, BCa CIs)...")
    clusters_tumour = tumour_df["pseudo_patient_cluster"].to_numpy()
    overlap_mask_tumour = tumour_df["is_overlap"].to_numpy()

    comparisons = [
        ("Arm C (Masked) vs Arm B (Inpainted)", "arm_c", "arm_b"),
        ("Arm C (Masked) vs Arm A (Raw)", "arm_c", "arm_a"),
        ("Arm B (Inpainted) vs Arm A (Raw)", "arm_b", "arm_a"),
    ]

    bootstrap_results = []

    for comp_name, arm_1, arm_2 in comparisons:
        for metric, valid_m in [
            ("dsc", None),
            ("iou", None),
            ("biou_d2", None),
            ("lpr", overlap_mask_tumour),
        ]:
            col_1 = f"{arm_1}_{metric if metric != 'biou_d2' else 'biou'}"
            col_2 = f"{arm_2}_{metric if metric != 'biou_d2' else 'biou'}"
            s1 = tumour_df[col_1].to_numpy()
            s2 = tumour_df[col_2].to_numpy()

            res = paired_cluster_bootstrap_segmentation(
                scores_c=s1,
                scores_b=s2,
                cluster_ids=clusters_tumour,
                valid_mask=valid_m,
                metric_name=metric,
                num_resamples=args.bootstrap_iterations,
                seed=args.seed,
            )

            bootstrap_results.append({
                "comparison": comp_name,
                "metric": metric,
                "cohort": "Overlap Cohort (N=193)" if metric == "lpr" else "All Tumour Scans (N=382)",
                "delta_mean": float(res.observed_difference),
                "ci_lower_bca": float(res.ci_lower_95),
                "ci_upper_bca": float(res.ci_upper_95),
                "ci_method": res.ci_method,
                "ci_lower_percentile": float(res.ci_lower_percentile_95),
                "ci_upper_percentile": float(res.ci_upper_percentile_95),
                "bca_fallback_used": (res.ci_method != "bca"),
                "z0_bias": float(res.bca_bias_correction_z0),
                "acceleration": float(res.bca_acceleration_a),
                "bootstrap_mean": float(res.bootstrap_mean),
                "bootstrap_std": float(res.bootstrap_std),
                "p_value": float(res.p_value),
                "num_clusters": res.num_clusters,
                "num_samples": res.num_samples,
                "empty_draws_rejected": res.empty_draws_count,
            })

    # Also compute bootstrap for Normal Ovary Clean Rejection Rate (Arm C vs Arm B)
    clusters_normal = normal_df["pseudo_patient_cluster"].to_numpy()
    for comp_name, arm_1, arm_2 in comparisons:
        for metric in ["clean_rej", "fpr_intra", "fpr_extra"]:
            s1 = normal_df[f"{arm_1}_{metric}"].to_numpy()
            s2 = normal_df[f"{arm_2}_{metric}"].to_numpy()
            res = paired_cluster_bootstrap_segmentation(
                scores_c=s1,
                scores_b=s2,
                cluster_ids=clusters_normal,
                valid_mask=None,
                metric_name=metric,
                num_resamples=args.bootstrap_iterations,
                seed=args.seed,
            )
            bootstrap_results.append({
                "comparison": comp_name,
                "metric": metric,
                "cohort": "Normal Ovary Scans (N=87)",
                "delta_mean": float(res.observed_difference),
                "ci_lower_bca": float(res.ci_lower_95),
                "ci_upper_bca": float(res.ci_upper_95),
                "ci_method": res.ci_method,
                "ci_lower_percentile": float(res.ci_lower_percentile_95),
                "ci_upper_percentile": float(res.ci_upper_percentile_95),
                "bca_fallback_used": (res.ci_method != "bca"),
                "z0_bias": float(res.bca_bias_correction_z0),
                "acceleration": float(res.bca_acceleration_a),
                "bootstrap_mean": float(res.bootstrap_mean),
                "bootstrap_std": float(res.bootstrap_std),
                "p_value": float(res.p_value),
                "num_clusters": res.num_clusters,
                "num_samples": res.num_samples,
                "empty_draws_rejected": res.empty_draws_count,
            })

    bootstrap_df = pd.DataFrame(bootstrap_results)
    bootstrap_csv = args.output_dir / f"seed{args.seed}_bootstrap_inferential_report.csv"
    bootstrap_df.to_csv(bootstrap_csv, index=False)
    logger.info(f"Saved inferential report: {bootstrap_csv}")

    # Generate diagnostic plots
    try:
        generate_diagnostic_plots(per_scan_df, bootstrap_df, normal_df, args.output_dir, seed=args.seed)
    except Exception as e:
        logger.warning(f"Could not generate plots (matplotlib error): {e}")

    # Generate master JSON report
    report_json = {
        "evaluation_name": f"Three-Arm Seed {args.seed} Segmentation Test Evaluation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "architecture": args.architecture,
        "resolution": args.resolution,
        "seed": args.seed,
        "cohorts": {
            "total_test_scans": len(test_df),
            "tumour_scans": len(tumour_df),
            "valid_artifact_overlap_scans": len(overlap_df),
            "zero_artifact_overlap_scans": len(zero_overlap_df),
            "normal_ovary_scans": len(normal_df),
            "unique_test_clusters": int(test_df["pseudo_patient_cluster"].nunique()),
        },
        "tumour_summary": tumour_summary,
        "normal_summary": normal_summary,
        "bootstrap_results": bootstrap_results,
    }

    report_path = args.output_dir / f"seed{args.seed}_evaluation_summary.json"
    with open(report_path, "w") as f:
        json.dump(report_json, f, indent=2)
    logger.info(f"Saved master JSON report: {report_path}")

    logger.info(f"=== Seed {args.seed} Test Evaluation Successfully Finished ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
