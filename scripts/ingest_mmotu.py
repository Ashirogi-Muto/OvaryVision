#!/usr/bin/env python3
"""Opt-in ingestion and post-ingestion audit helper for the MMOTU benchmark dataset.

IMPORTANT: This script NEVER downloads datasets automatically. Ingestion must be explicitly
authorized and pointed to a locally placed archive or downloaded dataset folder.
"""

import argparse
from pathlib import Path
import shutil
import sys
import zipfile
import tarfile

# Ensure src/ is on python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ovaryvision.data_audit import audit_image_dataset
from ovaryvision.logging_utils import setup_logger

logger = setup_logger("ovaryvision.mmotu_ingest")

AUTHORITATIVE_SOURCES = {
    "primary_paper": "Zhao et al., 'A Multi-Modality Ovarian Tumor Ultrasound Image Dataset for Unsupervised Cross-Domain Semantic Segmentation', IEEE ISBI / arXiv:2207.06799 (2022).",
    "zenodo_record": "https://zenodo.org/records/21128657 (DOI: 10.5281/zenodo.21128657)",
    "figshare_record": "https://figshare.com/articles/dataset/_zip/25058690 (DOI: 10.6084/m9.figshare.25058690)",
    "official_code_repo": "https://github.com/cv516Buaa/MMOTU_DS2Net",
    "license": "Creative Commons Attribution 4.0 International / Apache 2.0",
}


def print_ingestion_guide() -> None:
    """Print authoritative acquisition and placement instructions."""
    print("\n=======================================================")
    print("     MMOTU DATASET INGESTION & AUDIT INSTRUCTIONS     ")
    print("=======================================================")
    print(f"  Reference Citation : {AUTHORITATIVE_SOURCES['primary_paper']}")
    print(f"  Zenodo Deposit     : {AUTHORITATIVE_SOURCES['zenodo_record']}")
    print(f"  Figshare Deposit   : {AUTHORITATIVE_SOURCES['figshare_record']}")
    print(f"  Official GitHub    : {AUTHORITATIVE_SOURCES['official_code_repo']}")
    print(f"  Distribution License: {AUTHORITATIVE_SOURCES['license']}")
    print("\nStep-by-step ingestion procedure:")
    print("  1. Download the MMOTU zip/tar archive from Zenodo or Figshare.")
    print("  2. Place the archive at 'data/raw/mmotu.zip' or extract directly to:")
    print("       Images      : data/raw/mmotu/images/")
    print("       Tumor Masks : data/raw/mmotu/tumor_masks/")
    print("       Labels CSV  : data/raw/mmotu/labels.csv")
    print("  3. To extract an archive with this script:")
    print("       python3 scripts/ingest_mmotu.py --archive data/raw/mmotu.zip --confirm")
    print("  4. To audit an already extracted directory:")
    print("       python3 scripts/ingest_mmotu.py --audit-only --data-dir data/raw/mmotu/images/")
    print("\n[NOTE] Public MMOTU contains tumor ROI masks, NOT clinician caliper masks.")
    print("       Artifact masks must be placed in data/interim/artifact_masks/.\n")


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    """Safely extract zip or tar archive to target directory."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive file not found: {archive_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s to %s ...", archive_path, target_dir)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            # Check for path traversal vulnerabilities before extracting
            for member in zf.namelist():
                if member.startswith("/") or ".." in member:
                    raise ValueError(f"Illegal path traversal detected in archive: {member}")
            zf.extractall(target_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Illegal path traversal detected in archive: {member.name}")
            tf.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}. Expected .zip, .tar, or .tar.gz")

    logger.info("Extraction completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MMOTU Ingestion and Forensic Audit Tool.")
    parser.add_argument("--guide", action="store_true", help="Print authoritative acquisition instructions.")
    parser.add_argument("--archive", type=str, help="Path to downloaded MMOTU zip/tar archive.")
    parser.add_argument("--target-dir", type=str, default="data/raw/mmotu", help="Extraction destination directory.")
    parser.add_argument("--confirm", action="store_true", help="Explicit confirmation flag required for extraction.")
    parser.add_argument("--audit-only", action="store_true", help="Skip extraction and run forensic data audit.")
    parser.add_argument("--data-dir", type=str, default="data/raw/mmotu/images", help="Image directory to audit.")
    parser.add_argument("--output-dir", type=str, default="outputs/metrics", help="Directory for audit outputs.")
    parser.add_argument("--hamming-threshold", type=int, default=4, help="Hamming distance for near-duplicate clustering.")
    args = parser.parse_args()

    if args.guide or (not args.archive and not args.audit_only):
        print_ingestion_guide()
        if not args.archive and not args.audit_only:
            return

    if args.archive:
        archive_p = Path(args.archive)
        target_p = Path(args.target_dir)

        if not args.confirm:
            print("\n[SAFETY GATE] Ingestion requires explicit confirmation.")
            print(f"Target archive: {archive_p}")
            print(f"Target extraction directory: {target_p}")
            print("To proceed, re-run with '--confirm'.")
            sys.exit(1)

        try:
            extract_archive(archive_p, target_p)
        except Exception as exc:
            logger.error("Extraction failed: %s", exc)
            sys.exit(1)

    # Post-ingestion or audit-only execution
    data_dir_p = Path(args.data_dir)
    if not data_dir_p.exists():
        print(f"\n[ERROR] Image directory does not exist: {data_dir_p}")
        print("Please ensure images are extracted to this location before auditing.")
        sys.exit(1)

    try:
        summary, records = audit_image_dataset(
            target_dir=data_dir_p,
            output_dir=args.output_dir,
            dataset_id="MMOTU_2D",
            hamming_threshold=args.hamming_threshold,
        )
        print("\n=======================================================")
        print("          MMOTU FORENSIC AUDIT SUMMARY                 ")
        print("=======================================================")
        print(f"  Valid Images Found     : {summary.valid_images_found}")
        print(f"  Corrupt/Unreadable     : {summary.corrupt_or_unreadable_files}")
        print(f"  Unique SHA-256 Hashes  : {summary.unique_sha256_count}")
        print(f"  Exact Duplicate Pairs  : {summary.exact_duplicate_clusters_count}")
        print(f"  Near-Duplicate Pairs   : {summary.near_duplicate_clusters_count}")
        print(f"  Patient IDs Present    : {'YES' if summary.patient_ids_available else 'NO (Frame Leakage Risk)'}")
        print(f"  Audit Status           : {summary.status}")
        print(f"\nAudit report JSON and inventory CSV saved to: {args.output_dir}")
    except Exception as exc:
        logger.error("Audit failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

