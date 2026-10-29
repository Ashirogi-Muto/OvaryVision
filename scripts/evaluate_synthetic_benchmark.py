#!/usr/bin/env python3
"""Execute tri-partition comparative integrity evaluation on synthetic benchmark cases."""

import csv
import json
from pathlib import Path
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image

from ovaryvision.logging_utils import get_logger
from ovaryvision.preprocessing import HarmonicDiffusionInpainter, apply_arm_b, apply_arm_c
from ovaryvision.synthetic_artifacts import evaluate_reconstruction_integrity

logger = get_logger("evaluate_synthetic_benchmark")


def main() -> None:
    bench_dir = Path("data/interim/synthetic_benchmark")
    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_cases = sorted(list(bench_dir.glob("SYNTH_CASE_*_metadata.json")))
    if not json_cases:
        logger.error(f"No synthetic cases found in {bench_dir}. Run 'ovaryvision generate-synthetic-benchmark' first.")
        sys.exit(1)

    logger.info(f"Evaluating {len(json_cases)} synthetic benchmark cases...")
    inpainter = HarmonicDiffusionInpainter()

    results = []
    csv_rows = []

    for meta_file in json_cases:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        case_id = meta["case_id"]
        clean_p = Path(meta["clean_path"])
        corrupted_p = Path(meta["corrupted_path"])
        mask_p = Path(meta["mask_path"])

        clean_img = np.array(Image.open(clean_p).convert("L"))
        corrupted_img = np.array(Image.open(corrupted_p).convert("L"))
        mask = np.array(Image.open(mask_p).convert("L")) > 0

        # Arm A: Corrupted / Raw
        rep_a = evaluate_reconstruction_integrity(clean_img, corrupted_img, mask)

        # Arm B: Inpainted with strict exterior conservation
        inpainted_img = apply_arm_b(corrupted_img, mask, inpainter=inpainter, dilation_radius=2)
        rep_b = evaluate_reconstruction_integrity(clean_img, inpainted_img, mask)

        # Arm C: Zero-masked
        masked_img = apply_arm_c(corrupted_img, mask, fill_value=0, dilation_radius=2)
        rep_c = evaluate_reconstruction_integrity(clean_img, masked_img, mask)

        case_entry = {
            "case_id": case_id,
            "mask_pixel_count": int(rep_a.local_texture_variance.mask_pixel_count),
            "mask_pixel_ratio": float(rep_a.mask_pixel_ratio),
            "arm_a_raw": rep_a.model_dump(),
            "arm_b_inpainted": rep_b.model_dump(),
            "arm_c_masked": rep_c.model_dump(),
        }
        results.append(case_entry)

        # CSV reporting rows
        for arm_name, rep in [("Arm A (Raw)", rep_a), ("Arm B (Inpainted)", rep_b), ("Arm C (Masked)", rep_c)]:
            csv_rows.append({
                "case_id": case_id,
                "arm": arm_name,
                "global_mae": f"{rep.global_mae:.4f}",
                "global_psnr_db": f"{rep.global_psnr:.2f}" if rep.global_psnr != float("inf") else "inf",
                "global_ssim": f"{rep.global_ssim:.4f}",
                "in_mask_mae": f"{rep.in_mask_mae:.4f}",
                "in_mask_psnr_db": f"{rep.in_mask_psnr:.2f}" if rep.in_mask_psnr != float("inf") else "inf",
                "exterior_mae": f"{rep.exterior_mae:.4f}",
                "exterior_psnr_db": f"{rep.exterior_psnr:.2f}" if rep.exterior_psnr != float("inf") else "inf",
                "exterior_ssim": f"{rep.exterior_ssim:.4f}",
                "local_texture_variance_diff": f"{rep.local_texture_variance.delta_ltv:.2f}",
            })

        logger.info(
            f"Case {case_id}: "
            f"Arm B in-mask MAE={rep_b.in_mask_mae:.2f}, Exterior PSNR={rep_b.exterior_psnr} dB | "
            f"Arm C in-mask MAE={rep_c.in_mask_mae:.2f}, Exterior PSNR={rep_c.exterior_psnr} dB"
        )

    # Save JSON report
    json_path = out_dir / "synthetic_benchmark_comparative_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"total_cases": len(results), "cases": results}, f, indent=2)

    # Save CSV table
    csv_path = out_dir / "synthetic_benchmark_comparative_table.csv"
    fieldnames = [
        "case_id", "arm", "global_mae", "global_psnr_db", "global_ssim",
        "in_mask_mae", "in_mask_psnr_db", "exterior_mae", "exterior_psnr_db",
        "exterior_ssim", "local_texture_variance_diff"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info(f"Comparative benchmark reports exported:")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
