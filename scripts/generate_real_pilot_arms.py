#!/usr/bin/env python3
"""Generate Arms A, B, and C on the 140 QC-passed real MMOTU pilot images and compute integrity metrics."""

import csv
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ovaryvision.preprocessing import (
    apply_arm_a,
    apply_arm_b,
    apply_arm_c,
    dilate_binary_mask,
    HarmonicDiffusionInpainter,
)
from ovaryvision.synthetic_artifacts import compute_local_texture_variance


def compute_ssim_distortion(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute global structural similarity between two images (distortion relative to raw original)."""
    # Simple standardized SSIM implementation
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    var1 = np.var(img1)
    var2 = np.var(img2)
    cov = np.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2.0 * mu1 * mu2 + c1) * (2.0 * cov + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2))
    return float(ssim)


def main() -> None:
    manifest_p = Path("data/manifests/real_artifact_manifest_pilot.csv")
    if not manifest_p.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")

    out_base = Path("data/processed/real_pilot")
    dir_a = out_base / "arm_a_raw"
    dir_b = out_base / "arm_b_inpainted"
    dir_c = out_base / "arm_c_masked"

    for d in [dir_a, dir_b, dir_c]:
        d.mkdir(parents=True, exist_ok=True)

    metrics_csv_path = Path("outputs/metrics/real_pilot_integrity_metrics.csv")
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(manifest_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["qc_status"] == "PASSED":
                rows.append(r)

    print(f"Generating 3 arms for {len(rows)} QC-passed pilot images...")
    inpainter = HarmonicDiffusionInpainter(max_iterations=50)

    integrity_records = []

    for idx, r in enumerate(rows, 1):
        img_id = r["image_id"]
        raw_p = Path(r["relative_image_path"])
        mask_p = Path(r["artifact_mask_path"])

        raw_pil = Image.open(raw_p).convert("RGB")
        raw_np = np.array(raw_pil)
        mask_np = (np.array(Image.open(mask_p).convert("L")) > 0).astype(np.uint8)

        # 2px morphological dilation
        dilated_mask = dilate_binary_mask(mask_np, radius=2)
        dilated_area = int(np.sum(dilated_mask))
        dilated_ratio = dilated_area / (raw_np.shape[0] * raw_np.shape[1])

        # 1. Arm A (Raw Identity)
        arm_a_np = apply_arm_a(raw_np)
        Image.fromarray(arm_a_np).save(dir_a / f"{img_id}.png")

        # 2. Arm B (Harmonic Inpainting with 2px dilation)
        arm_b_np = apply_arm_b(raw_np, mask_np, inpainter=inpainter, dilation_radius=2)
        Image.fromarray(arm_b_np).save(dir_b / f"{img_id}.png")

        # 3. Arm C (Zero-Masking with 2px dilation)
        arm_c_np = apply_arm_c(raw_np, mask_np, fill_value=0, dilation_radius=2)
        Image.fromarray(arm_c_np).save(dir_c / f"{img_id}.png")

        # --- Verification of Exterior Invariance ---
        exterior_mask = ~dilated_mask
        
        # Arm B Exterior
        ext_diff_b = np.abs(arm_b_np[exterior_mask].astype(float) - raw_np[exterior_mask].astype(float))
        ext_mae_b = float(np.mean(ext_diff_b))
        ext_psnr_b = "inf" if ext_mae_b == 0.0 else f"{20.0 * math.log10(255.0 / max(float(np.sqrt(np.mean(ext_diff_b**2))), 1e-10)):.2f}"

        # Arm C Exterior
        ext_diff_c = np.abs(arm_c_np[exterior_mask].astype(float) - raw_np[exterior_mask].astype(float))
        ext_mae_c = float(np.mean(ext_diff_c))
        ext_psnr_c = "inf" if ext_mae_c == 0.0 else f"{20.0 * math.log10(255.0 / max(float(np.sqrt(np.mean(ext_diff_c**2))), 1e-10)):.2f}"

        # --- In-Mask Metrics ---
        # Interior shift relative to raw artifact pixels
        int_diff_b = np.abs(arm_b_np[dilated_mask].astype(float) - raw_np[dilated_mask].astype(float))
        int_mae_b = float(np.mean(int_diff_b))
        
        int_diff_c = np.abs(arm_c_np[dilated_mask].astype(float) - raw_np[dilated_mask].astype(float))
        int_mae_c = float(np.mean(int_diff_c))

        # Local Texture Variation (LTV) on grayscale
        raw_gray = np.array(raw_pil.convert("L"))
        b_gray = np.array(Image.fromarray(arm_b_np).convert("L"))
        c_gray = np.array(Image.fromarray(arm_c_np).convert("L"))

        res_b = compute_local_texture_variance(b_gray, dilated_mask)
        res_c = compute_local_texture_variance(c_gray, dilated_mask)
        delta_ltv_b = res_b.delta_ltv
        delta_ltv_c = res_c.delta_ltv

        # Global Distortion relative to raw scan
        ssim_b = compute_ssim_distortion(raw_gray, b_gray)
        ssim_c = compute_ssim_distortion(raw_gray, c_gray)

        integrity_records.append({
            "image_id": img_id,
            "raw_mask_pixels": r["pixel_count"],
            "dilated_mask_pixels": dilated_area,
            "dilated_area_ratio": f"{dilated_ratio:.5f}",
            "arm_b_exterior_mae": f"{ext_mae_b:.6f}",
            "arm_b_exterior_psnr": ext_psnr_b,
            "arm_b_in_mask_mae": f"{int_mae_b:.2f}",
            "arm_b_delta_ltv": f"{delta_ltv_b:.2f}",
            "arm_b_ssim_distortion_to_raw": f"{ssim_b:.4f}",
            "arm_c_exterior_mae": f"{ext_mae_c:.6f}",
            "arm_c_exterior_psnr": ext_psnr_c,
            "arm_c_in_mask_mae": f"{int_mae_c:.2f}",
            "arm_c_delta_ltv": f"{delta_ltv_c:.2f}",
            "arm_c_ssim_distortion_to_raw": f"{ssim_c:.4f}",
        })

        if idx % 25 == 0 or idx == len(rows):
            print(f"  Processed {idx}/{len(rows)} images... (Exterior MAE: B={ext_mae_b:.1e}, C={ext_mae_c:.1e})")

    # Write integrity metrics CSV
    fieldnames = list(integrity_records[0].keys())
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(integrity_records)

    print(f"\n[✓] Finished generating 3 arms for {len(rows)} images.")
    print(f"Transformed images saved in: {out_base}")
    print(f"Integrity metrics saved in: {metrics_csv_path}")

    # Aggregate integrity summary
    b_ext_maes = [float(rec["arm_b_exterior_mae"]) for rec in integrity_records]
    c_ext_maes = [float(rec["arm_c_exterior_mae"]) for rec in integrity_records]
    b_dltvs = [float(rec["arm_b_delta_ltv"]) for rec in integrity_records]
    c_dltvs = [float(rec["arm_c_delta_ltv"]) for rec in integrity_records]
    b_ssims = [float(rec["arm_b_ssim_distortion_to_raw"]) for rec in integrity_records]
    c_ssims = [float(rec["arm_c_ssim_distortion_to_raw"]) for rec in integrity_records]

    print("\nAggregate Image Integrity Summary:")
    print(f"  Arm B Exterior MAE : {np.mean(b_ext_maes):.6f} (Tolerance: 0.0, PSNR=inf)")
    print(f"  Arm C Exterior MAE : {np.mean(c_ext_maes):.6f} (Tolerance: 0.0, PSNR=inf)")
    print(f"  Arm B In-Mask Delta-LTV : {np.mean(b_dltvs):.2f} ± {np.std(b_dltvs):.2f}")
    print(f"  Arm C In-Mask Delta-LTV : {np.mean(c_dltvs):.2f} ± {np.std(c_dltvs):.2f}")
    print(f"  Arm B SSIM Distortion   : {np.mean(b_ssims):.4f} ± {np.std(b_ssims):.4f}")
    print(f"  Arm C SSIM Distortion   : {np.mean(c_ssims):.4f} ± {np.std(c_ssims):.4f}")


if __name__ == "__main__":
    main()
