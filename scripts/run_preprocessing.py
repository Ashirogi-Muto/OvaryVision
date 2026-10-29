#!/usr/bin/env python3
"""Batch preprocessing script executing Arms A, B, and C across an image manifest."""

import argparse
from pathlib import Path
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from typing import List

from ovaryvision.logging_utils import get_logger
from ovaryvision.manifests import ImageRecord, load_image_manifest
from ovaryvision.preprocessing import (
    HarmonicDiffusionInpainter,
    process_single_image,
)

logger = get_logger("run_preprocessing")


def run_batch_preprocessing(
    manifest_path: Path,
    output_base_dir: Path,
    arms: List[str],
    dilation_radius: int = 2,
    fill_value: int = 0,
    limit: int = 0,
) -> None:
    """Run batch image preprocessing across specified arms."""
    logger.info(f"Loading image records from {manifest_path}...")
    records = load_image_manifest(manifest_path)
    if limit > 0:
        records = records[:limit]
        logger.info(f"Limited processing to first {limit} records.")

    logger.info(f"Processing {len(records)} records for arms: {arms}")
    inpainter = HarmonicDiffusionInpainter()

    success_counts = {arm: 0 for arm in arms}
    error_counts = {arm: 0 for arm in arms}

    for idx, rec in enumerate(records, start=1):
        img_path = Path(rec.relative_image_path)
        # Check if relative path exists from current directory or relative to project root
        if not img_path.is_file():
            # Try searching under data/
            candidates = list(Path("data").glob(f"**/{rec.image_id}.*"))
            if candidates:
                img_path = candidates[0]
            else:
                logger.warning(f"[{idx}/{len(records)}] Skipping {rec.image_id}: image file not found at {rec.relative_image_path}")
                continue

        mask_path = Path(rec.artifact_mask_path) if rec.artifact_mask_path else None
        if mask_path and not mask_path.is_file():
            logger.debug(f"Mask file not found at {mask_path}, treating as unmasked.")
            mask_path = None

        for arm in arms:
            arm_dir = output_base_dir / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            out_file = arm_dir / f"{rec.image_id}.png"

            try:
                process_single_image(
                    image_path=img_path,
                    mask_path=mask_path,
                    arm=arm,
                    output_path=out_file,
                    inpainter=inpainter,
                    dilation_radius=dilation_radius,
                    fill_value=fill_value,
                )
                success_counts[arm] += 1
            except Exception as e:
                logger.error(f"Failed processing {rec.image_id} for {arm}: {e}")
                error_counts[arm] += 1

        if idx % 50 == 0 or idx == len(records):
            logger.info(f"Progress: [{idx}/{len(records)}] records processed.")

    logger.info("Batch preprocessing completed.")
    for arm in arms:
        logger.info(f"Arm '{arm}': {success_counts[arm]} succeeded, {error_counts[arm]} failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch preprocess ultrasound scans across experimental arms.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to image manifest CSV/JSON")
    parser.add_argument("--output-base-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument(
        "--arm",
        type=str,
        default="all",
        choices=["all", "arm_a_raw", "arm_b_inpainted", "arm_c_masked"],
        help="Target preprocessing arm or 'all'",
    )
    parser.add_argument("--dilation-radius", type=int, default=2, help="Mask dilation radius in pixels")
    parser.add_argument("--fill-value", type=int, default=0, help="Arm C neutral fill value")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of images to process")
    args = parser.parse_args()

    manifest_p = Path(args.manifest).resolve()
    if not manifest_p.is_file():
        logger.error(f"Manifest file not found: {manifest_p}")
        sys.exit(1)

    if args.arm == "all":
        target_arms = ["arm_a_raw", "arm_b_inpainted", "arm_c_masked"]
    else:
        target_arms = [args.arm]

    out_base = Path(args.output_base_dir).resolve()
    run_batch_preprocessing(
        manifest_path=manifest_p,
        output_base_dir=out_base,
        arms=target_arms,
        dilation_radius=args.dilation_radius,
        fill_value=args.fill_value,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
