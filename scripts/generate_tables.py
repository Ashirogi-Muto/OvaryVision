#!/usr/bin/env python3
"""Format evaluation metrics into publication-ready Markdown and LaTeX manuscript tables."""

import csv
import json
from pathlib import Path
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ovaryvision.logging_utils import get_logger

logger = get_logger("generate_tables")


def format_synthetic_benchmark_latex(csv_path: Path, output_tex: Path) -> None:
    """Read synthetic benchmark CSV and format into academic LaTeX table."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Group by arm and compute means
    arms = ["Arm A (Raw)", "Arm B (Inpainted)", "Arm C (Masked)"]
    metrics = {
        arm: {
            "global_mae": [],
            "global_psnr": [],
            "global_ssim": [],
            "in_mask_mae": [],
            "in_mask_psnr": [],
            "exterior_mae": [],
            "exterior_psnr": [],
            "exterior_ssim": [],
            "ltv_diff": [],
        }
        for arm in arms
    }

    for r in rows:
        arm = r["arm"]
        if arm in metrics:
            metrics[arm]["global_mae"].append(float(r["global_mae"]))
            if r["global_psnr_db"] != "inf":
                metrics[arm]["global_psnr"].append(float(r["global_psnr_db"]))
            metrics[arm]["global_ssim"].append(float(r["global_ssim"]))
            metrics[arm]["in_mask_mae"].append(float(r["in_mask_mae"]))
            if r["in_mask_psnr_db"] != "inf":
                metrics[arm]["in_mask_psnr"].append(float(r["in_mask_psnr_db"]))
            metrics[arm]["exterior_mae"].append(float(r["exterior_mae"]))
            if r["exterior_psnr_db"] != "inf":
                metrics[arm]["exterior_psnr"].append(float(r["exterior_psnr_db"]))
            metrics[arm]["exterior_ssim"].append(float(r["exterior_ssim"]))
            metrics[arm]["ltv_diff"].append(float(r["local_texture_variance_diff"]))

    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controlled Synthetic-Artifact Benchmark: Tri-Partition Image Integrity Comparison across Preprocessing Arms. Values are reported as mean (standard deviation) across synthesized test scans.}",
        r"\label{tab:synthetic_integrity}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Preprocessing Arm} & \textbf{In-Mask MAE} $\downarrow$ & \textbf{In-Mask PSNR} $\uparrow$ & \textbf{Ext. MAE} $\downarrow$ & \textbf{Ext. PSNR} $\uparrow$ & \textbf{Global SSIM} $\uparrow$ & $\Delta$\textbf{LTV} $\downarrow$ \\",
        r"\midrule",
    ]

    for arm in arms:
        m = metrics[arm]
        im_mae_str = f"{np_mean(m['in_mask_mae']):.2f} ({np_std(m['in_mask_mae']):.2f})"
        im_psnr_str = f"{np_mean(m['in_mask_psnr']):.2f} ({np_std(m['in_mask_psnr']):.2f})"
        ext_mae_str = f"{np_mean(m['exterior_mae']):.2f} ({np_std(m['exterior_mae']):.2f})"
        ext_psnr_str = f"{np_mean(m['exterior_psnr']):.2f} ({np_std(m['exterior_psnr']):.2f})" if m['exterior_psnr'] else "$\\infty$"
        ssim_str = f"{np_mean(m['global_ssim']):.4f} ({np_std(m['global_ssim']):.4f})"
        ltv_str = f"{np_mean(m['ltv_diff']):.1f} ({np_std(m['ltv_diff']):.1f})"

        # Clean arm label for LaTeX
        clean_arm = arm.replace("&", r"\&")
        tex_lines.append(f"{clean_arm} & {im_mae_str} & {im_psnr_str} & {ext_mae_str} & {ext_psnr_str} & {ssim_str} & {ltv_str} \\\\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")

    logger.info(f"Exported LaTeX table to {output_tex}")


def np_mean(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def np_std(lst: list) -> float:
    if len(lst) <= 1:
        return 0.0
    mean = np_mean(lst)
    variance = sum((x - mean) ** 2 for x in lst) / (len(lst) - 1)
    return variance ** 0.5


def main() -> None:
    csv_path = Path("outputs/metrics/synthetic_benchmark_comparative_table.csv")
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tex = out_dir / "table1_synthetic_integrity.tex"

    if csv_path.is_file():
        format_synthetic_benchmark_latex(csv_path, out_tex)
    else:
        logger.warning(f"CSV table not found at {csv_path}. Run evaluate_synthetic_benchmark.py first.")


if __name__ == "__main__":
    main()

