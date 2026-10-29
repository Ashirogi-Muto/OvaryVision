#!/usr/bin/env python3
"""Audit artifact masks and generate visual contact sheets for malignant and benign cases."""

import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.ndimage as ndi
import pandas as pd

def extract_mask(gray: np.ndarray) -> np.ndarray:
    bright = gray >= 240
    h_kernel = np.ones((1, 12), dtype=bool)
    v_kernel = np.ones((12, 1), dtype=bool)
    lines = ndi.binary_opening(bright, structure=h_kernel) | ndi.binary_opening(bright, structure=v_kernel)
    labeled, num_features = ndi.label(bright)
    text = np.zeros_like(bright)
    for f in range(1, num_features + 1):
        f_mask = labeled == f
        sz = np.sum(f_mask)
        if 15 <= sz <= 450:
            text |= f_mask
    return (lines | text).astype(np.uint8)

def create_overlay(raw_rgb: np.ndarray, art_mask: np.ndarray, tumor_mask: np.ndarray | None) -> Image.Image:
    # Yellow for artifact mask (255, 255, 0)
    # Cyan for tumor contour (0, 255, 255)
    vis = raw_rgb.copy()
    h, w = vis.shape[:2]

    # Yellow overlay on artifact mask
    if np.any(art_mask):
        vis[art_mask > 0] = [255, 230, 0]

    vis_pil = Image.fromarray(vis)
    draw = ImageDraw.Draw(vis_pil)

    # Draw cyan contour for tumor
    if tumor_mask is not None and np.any(tumor_mask):
        eroded = ndi.binary_erosion(tumor_mask, iterations=2)
        contour = tumor_mask ^ eroded
        for y, x in zip(*np.where(contour)):
            vis_pil.putpixel((x, y), (0, 255, 255))

    return vis_pil

def main():
    data_dir = Path("data/raw/mmotu/images")
    tumor_dir = Path("data/raw/mmotu/tumor_masks")
    manifest_df = pd.read_csv("data/manifests/mmotu_split_seed42.csv").set_index("image_id")

    out_dir = Path("outputs/qc/contact_sheets")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_images = sorted(list(data_dir.glob("*.JPG")), key=lambda p: int(p.stem))

    records = []
    overlays_mal = []

    print(f"Auditing {len(raw_images)} images and generating visual overlays...")

    for idx, p in enumerate(raw_images):
        stem = int(p.stem)
        lbl = manifest_df.loc[stem, "diagnostic_label"] if stem in manifest_df.index else 0
        role = manifest_df.loc[stem, "split_role"] if stem in manifest_df.index else "train"

        img_pil = Image.open(p).convert("RGB")
        raw_rgb = np.array(img_pil)
        gray = np.array(img_pil.convert("L"))

        art_mask = extract_mask(gray)
        n_art = int(np.sum(art_mask))
        total_px = gray.shape[0] * gray.shape[1]
        area_ratio = float(n_art / total_px)

        tumor_p = tumor_dir / f"{stem}_binary.PNG"
        tumor_mask = np.array(Image.open(tumor_p).convert("L")) > 0 if tumor_p.is_file() else None

        tumor_overlap = 0.0
        if tumor_mask is not None and n_art > 0:
            tumor_overlap = float(np.sum(art_mask & tumor_mask) / n_art)

        # Determine QC status
        if n_art == 0:
            qc_status = "FLAGGED_EMPTY"
        elif area_ratio > 0.05:
            qc_status = "FLAGGED_EXCESSIVE_AREA"
        elif tumor_overlap > 0.50:
            qc_status = "FLAGGED_TUMOR_OVERLAP"
        else:
            qc_status = "PASSED"

        records.append({
            "image_id": stem,
            "diagnostic_label": lbl,
            "split_role_seed42": role,
            "has_artifact": bool(n_art > 0),
            "pixel_count": n_art,
            "area_ratio": round(area_ratio, 5),
            "tumor_overlap": round(tumor_overlap, 4),
            "qc_status": qc_status
        })

        # Save overlay for all 53 malignant cases
        if lbl == 1:
            ov = create_overlay(raw_rgb, art_mask, tumor_mask)
            # Add text annotation
            draw = ImageDraw.Draw(ov)
            caption = f"ID:{stem} | QC:{qc_status} | Ovlp:{tumor_overlap*100:.1f}%"
            draw.rectangle([0, 0, len(caption)*7 + 4, 16], fill=(0, 0, 0))
            draw.text((2, 2), caption, fill=(255, 255, 255))
            overlays_mal.append((stem, ov))

    df = pd.DataFrame(records)
    df.to_csv("outputs/metrics/full_cohort_mask_qc_summary.csv", index=False)
    print("Saved outputs/metrics/full_cohort_mask_qc_summary.csv")

    # 1. Generate Malignant Contact Sheet (53 images: 6 rows x 9 cols)
    print(f"Generating malignant contact sheet for {len(overlays_mal)} cases...")
    cell_w, cell_h = 180, 135
    cols, rows = 9, 6
    sheet_w = cols * cell_w
    sheet_h = rows * cell_h
    mal_sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))

    for i, (stem, ov) in enumerate(overlays_mal):
        r = i // cols
        c = i % cols
        thumb = ov.resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        mal_sheet.paste(thumb, (c * cell_w, r * cell_h))

    mal_sheet.save(out_dir / "malignant_cases_artifact_contact_sheet.png")
    print(f"Saved: {out_dir / 'malignant_cases_artifact_contact_sheet.png'}")

    # 2. Generate Benign Sample Contact Sheet (36 images: 6 rows x 6 cols)
    benign_sample_ids = [1, 10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400,
                         450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
                         1050, 1100, 1150, 1200, 1250, 1300, 1320, 1340, 1360, 1380, 1400, 1410]
    benign_sheet = Image.new("RGB", (6 * cell_w, 6 * cell_h), (20, 20, 20))
    for i, b_id in enumerate(benign_sample_ids):
        p = data_dir / f"{b_id}.JPG"
        if not p.is_file():
            continue
        img_pil = Image.open(p).convert("RGB")
        raw_rgb = np.array(img_pil)
        gray = np.array(img_pil.convert("L"))
        art_mask = extract_mask(gray)
        tumor_p = tumor_dir / f"{b_id}_binary.PNG"
        tumor_mask = np.array(Image.open(tumor_p).convert("L")) > 0 if tumor_p.is_file() else None
        ov = create_overlay(raw_rgb, art_mask, tumor_mask)
        draw = ImageDraw.Draw(ov)
        n_art = int(np.sum(art_mask))
        cap = f"ID:{b_id} (Benign) | CaliperPx:{n_art}"
        draw.rectangle([0, 0, len(cap)*7 + 4, 16], fill=(0, 0, 0))
        draw.text((2, 2), cap, fill=(255, 255, 255))
        thumb = ov.resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        r = i // 6
        c = i % 6
        benign_sheet.paste(thumb, (c * cell_w, r * cell_h))

    benign_sheet.save(out_dir / "benign_sample_artifact_contact_sheet.png")
    print(f"Saved: {out_dir / 'benign_sample_artifact_contact_sheet.png'}")

if __name__ == "__main__":
    main()
