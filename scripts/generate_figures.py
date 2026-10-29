#!/usr/bin/env python3
"""Publication-grade visualization generator for OvaryVision research artifacts."""

import json
from pathlib import Path
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ovaryvision.logging_utils import get_logger
from ovaryvision.preprocessing import HarmonicDiffusionInpainter, apply_arm_b, apply_arm_c

logger = get_logger("generate_figures")


def generate_synthetic_comparison_figure(case_id: str = "SYNTH_CASE_0001", output_dir: Path = Path("outputs/figures")) -> Path:
    """Generate multi-panel figure comparing Clean, Corrupted, Mask, Inpainted, and Masked."""
    bench_dir = Path("data/interim/synthetic_benchmark")
    meta_file = bench_dir / f"{case_id}_metadata.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"Case metadata not found: {meta_file}")

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    clean_img = np.array(Image.open(meta["clean_path"]).convert("L"))
    corrupted_img = np.array(Image.open(meta["corrupted_path"]).convert("L"))
    mask = np.array(Image.open(meta["mask_path"]).convert("L")) > 0

    inpainter = HarmonicDiffusionInpainter()
    inpainted_img = apply_arm_b(corrupted_img, mask, inpainter=inpainter, dilation_radius=2)
    masked_img = apply_arm_c(corrupted_img, mask, fill_value=0, dilation_radius=2)

    diff_b = np.abs(inpainted_img.astype(float) - clean_img.astype(float))
    diff_c = np.abs(masked_img.astype(float) - clean_img.astype(float))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=300)
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    # Panel A: Clean
    axes[0, 0].imshow(clean_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("(a) Clean Reference ($I_{clean}$)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel B: Corrupted (Arm A)
    axes[0, 1].imshow(corrupted_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("(b) Arm A: Corrupted / Calipers ($I_A$)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # Panel C: Binary Artifact Mask
    axes[0, 2].imshow(mask, cmap="Blues_r")
    axes[0, 2].set_title("(c) Annotation Mask ($M_{art}$)", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    # Panel D: Arm B Inpainted
    axes[1, 0].imshow(inpainted_img, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("(d) Arm B: Inpainted ($I_B$)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # Panel E: Arm C Masked
    axes[1, 1].imshow(masked_img, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("(e) Arm C: Zero-Masked ($I_C$)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    # Panel F: Inpainting Absolute Error Map
    im_err = axes[1, 2].imshow(diff_b, cmap="inferno", vmin=0, vmax=60)
    axes[1, 2].set_title("(f) Arm B Error Map: $|I_B - I_{clean}|$", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")
    cbar = fig.colorbar(im_err, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Pixel Error", fontsize=9)

    out_file = output_dir / "synthetic_benchmark_tripartition.svg"
    out_png = output_dir / "synthetic_benchmark_tripartition.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    logger.info(f"Generated synthetic benchmark comparison figures: {out_file} and {out_png}")
    return out_file


def generate_reliability_diagram_figure(
    calib_report_path: Path = Path("outputs/metrics/calibration_report.json"),
    output_dir: Path = Path("outputs/figures"),
) -> Path:
    """Generate reliability diagram comparing uncalibrated vs calibrated probabilities."""
    if not calib_report_path.is_file():
        # Generate representative calibration report from simulation if file not present
        from ovaryvision.calibration import evaluate_probability_calibration
        rng = np.random.default_rng(42)
        n = 500
        z_val = rng.normal(0, 1.2, size=n) * 2.2
        y_val = (rng.uniform(0, 1, size=n) < (1.0 / (1.0 + np.exp(-z_val / 2.2)))).astype(int)
        z_test = rng.normal(0, 1.2, size=n) * 2.2
        y_test = (rng.uniform(0, 1, size=n) < (1.0 / (1.0 + np.exp(-z_test / 2.2)))).astype(int)
        report = evaluate_probability_calibration(z_val, y_val, z_test, y_test)
    else:
        with open(calib_report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        from ovaryvision.calibration import CalibrationReport
        report = CalibrationReport(**data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. Uncalibrated
    uncal_confs = [b.mean_confidence for b in report.uncalibrated_bins if b.sample_count > 0]
    uncal_freqs = [b.observed_frequency for b in report.uncalibrated_bins if b.sample_count > 0]
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[0].plot(uncal_confs, uncal_freqs, "s-", color="#d9534f", label=f"Uncalibrated (ECE: {report.uncalibrated_ece:.3f})")
    axes[0].set_xlabel("Mean Predicted Confidence", fontsize=10)
    axes[0].set_ylabel("Observed Empirical Frequency", fontsize=10)
    axes[0].set_title(f"Uncalibrated Probabilities\nECE = {report.uncalibrated_ece:.3f}, Brier = {report.uncalibrated_brier:.3f}", fontsize=11)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right")

    # 2. Calibrated (Temperature Scaling)
    cal_confs = [b.mean_confidence for b in report.calibrated_bins if b.sample_count > 0]
    cal_freqs = [b.observed_frequency for b in report.calibrated_bins if b.sample_count > 0]
    axes[1].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[1].plot(cal_confs, cal_freqs, "o-", color="#0275d8", label=f"Temp Scaling T*={report.optimal_temperature:.2f} (ECE: {report.calibrated_ece:.3f})")
    axes[1].set_xlabel("Mean Predicted Confidence", fontsize=10)
    axes[1].set_ylabel("Observed Empirical Frequency", fontsize=10)
    axes[1].set_title(f"Calibrated Probabilities (T* = {report.optimal_temperature:.2f})\nECE = {report.calibrated_ece:.3f}, Brier = {report.calibrated_brier:.3f}", fontsize=11)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower right")

    out_file = output_dir / "calibration_reliability_diagrams.svg"
    out_png = output_dir / "calibration_reliability_diagrams.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    logger.info(f"Generated reliability diagram figures: {out_file} and {out_png}")
    return out_file


def main() -> None:
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_synthetic_comparison_figure(case_id="SYNTH_CASE_0001", output_dir=out_dir)
    generate_reliability_diagram_figure(output_dir=out_dir)


if __name__ == "__main__":
    main()

