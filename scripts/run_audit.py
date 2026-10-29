#!/usr/bin/env python3
"""Convenience script to run data auditing on a target directory."""

import argparse
from pathlib import Path
import sys

# Ensure src/ is on Python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ovaryvision.data_audit import audit_image_dataset
from ovaryvision.logging_utils import setup_logger

logger = setup_logger("ovaryvision.audit_script")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ultrasound dataset directory.")
    parser.add_argument("--data-dir", "-d", required=True, help="Directory containing ultrasound images.")
    parser.add_argument("--output-dir", "-o", default="outputs/metrics", help="Directory to save audit outputs.")
    parser.add_argument("--dataset-id", default="MMOTU_2D", help="Dataset identifier.")
    parser.add_argument("--hamming-threshold", type=int, default=4, help="Hamming distance for near-duplicates.")
    args = parser.parse_args()

    try:
        summary, records = audit_image_dataset(
            target_dir=args.data_dir,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            hamming_threshold=args.hamming_threshold,
        )
        print(f"\nAudit complete for {args.dataset_id}:")
        print(f"  Valid images: {summary.valid_images_found}")
        print(f"  Corrupt files: {summary.corrupt_or_unreadable_files}")
        print(f"  Exact duplicates: {summary.exact_duplicate_clusters_count}")
        print(f"  Near duplicates: {summary.near_duplicate_clusters_count}")
        print(f"  Status: {summary.status}")
    except Exception as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

