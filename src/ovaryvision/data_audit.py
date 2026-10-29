"""Data audit, image inventory, SHA-256 deduplication, and near-duplicate hashing."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ovaryvision.logging_utils import get_logger
from ovaryvision.manifests import ImageRecord, save_image_manifest

logger = get_logger("ovaryvision.data_audit")

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DuplicateCluster(BaseModel):
    """Cluster of identical or near-duplicate images."""
    cluster_id: str
    cluster_type: str  # 'exact_duplicate' (sha256) or 'perceptual_near_duplicate' (dHash)
    image_paths: List[str]
    representative_path: str


class AuditSummary(BaseModel):
    """Structured summary output of an image dataset audit."""
    audit_timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    target_directory: str
    total_files_scanned: int
    valid_images_found: int
    corrupt_or_unreadable_files: int
    corrupt_file_paths: List[str] = Field(default_factory=list)
    unique_sha256_count: int
    exact_duplicate_clusters_count: int
    exact_duplicate_clusters: List[DuplicateCluster] = Field(default_factory=list)
    near_duplicate_clusters_count: int
    near_duplicate_clusters: List[DuplicateCluster] = Field(default_factory=list)
    dimensions_distribution: Dict[str, int] = Field(default_factory=dict)
    channels_distribution: Dict[str, int] = Field(default_factory=dict)
    patient_ids_available: bool = False
    patient_ids_note: str = ""
    manifest_reconciled: bool = False
    status: str = "COMPLETED"


def compute_sha256(file_path: Path | str) -> str:
    """Compute SHA-256 digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash) for fast near-duplicate detection."""
    # Resize to (hash_size + 1, hash_size) grayscale
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized)
    # Compare adjacent pixels in each row
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Pack boolean array into an integer
    decimal_val = 0
    for bit in diff.flatten():
        decimal_val = (decimal_val << 1) | int(bit)
    return decimal_val


def hamming_distance(h1: int, h2: int) -> int:
    """Compute bitwise Hamming distance between two integer hashes."""
    return bin(h1 ^ h2).count("1")


def inspect_image_file(file_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Attempt to open and read image metadata. Returns (metadata_dict, error_string)."""
    try:
        sha = compute_sha256(file_path)
        size_bytes = file_path.stat().st_size

        with Image.open(file_path) as img:
            # Force load image data to catch corrupt truncated files
            img.verify()

        # Reopen after verify() to inspect properties and compute dhash
        with Image.open(file_path) as img:
            width, height = img.size
            bands = len(img.getbands())
            dhash_val = compute_dhash(img)

        return {
            "path": str(file_path),
            "sha256": sha,
            "width": width,
            "height": height,
            "channels": bands,
            "size_bytes": size_bytes,
            "dhash": dhash_val,
        }, None
    except Exception as exc:
        return None, f"Failed to inspect {file_path.name}: {exc}"


def cluster_near_duplicates(
    items: List[Dict[str, Any]],
    hamming_threshold: int = 4,
) -> List[DuplicateCluster]:
    """Group images by perceptual dHash Hamming distance into clusters."""
    n = len(items)
    visited: Set[int] = set()
    clusters: List[DuplicateCluster] = []

    for i in range(n):
        if i in visited:
            continue
        cluster_members = [items[i]["path"]]
        visited.add(i)

        h1 = items[i]["dhash"]
        for j in range(i + 1, n):
            if j in visited:
                continue
            h2 = items[j]["dhash"]
            if hamming_distance(h1, h2) <= hamming_threshold:
                cluster_members.append(items[j]["path"])
                visited.add(j)

        if len(cluster_members) > 1:
            clusters.append(
                DuplicateCluster(
                    cluster_id=f"NEAR_DUP_{len(clusters)+1:04d}",
                    cluster_type="perceptual_near_duplicate",
                    image_paths=cluster_members,
                    representative_path=cluster_members[0],
                )
            )

    return clusters


def audit_image_dataset(
    target_dir: Path | str,
    output_dir: Optional[Path | str] = None,
    dataset_id: str = "MMOTU_2D",
    hamming_threshold: int = 4,
    metadata_manifest_path: Optional[Path | str] = None,
) -> Tuple[AuditSummary, List[ImageRecord]]:
    """Scan and audit an image directory, reporting duplicates, dimensions, and corruptions.

    Args:
        target_dir: Root directory containing images to audit.
        output_dir: Where to save audit_report.json and inventory.csv (if specified).
        dataset_id: Identifier of the dataset being audited.
        hamming_threshold: Maximum Hamming distance for near-duplicate clustering.
        metadata_manifest_path: Optional path to existing metadata manifest for cross-checking.

    Returns:
        (AuditSummary, List[ImageRecord])
    """
    target = Path(target_dir).resolve()
    if not target.exists():
        msg = f"Target directory does not exist: {target}. Cannot audit non-existent data."
        logger.error(msg)
        raise FileNotFoundError(msg)

    if not target.is_dir():
        msg = f"Target path is not a directory: {target}."
        logger.error(msg)
        raise NotADirectoryError(msg)

    logger.info("Starting dataset audit on directory: %s", target)

    # Discover candidate files
    discovered_files: List[Path] = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        discovered_files.extend(target.rglob(f"*{ext}"))
        discovered_files.extend(target.rglob(f"*{ext.upper()}"))

    # De-duplicate file list if case-insensitive matching yielded duplicates
    discovered_files = sorted(list({p.resolve() for p in discovered_files}))

    if not discovered_files:
        msg = (
            f"No supported image files found in {target} (checked extensions: {SUPPORTED_IMAGE_EXTENSIONS}). "
            "Please ensure ultrasound images are placed in the target directory."
        )
        logger.warning(msg)
        summary = AuditSummary(
            target_directory=str(target),
            total_files_scanned=0,
            valid_images_found=0,
            corrupt_or_unreadable_files=0,
            unique_sha256_count=0,
            exact_duplicate_clusters_count=0,
            near_duplicate_clusters_count=0,
            patient_ids_available=False,
            patient_ids_note="No images scanned; patient IDs unverified.",
            status="NO_IMAGES_FOUND",
        )
        return summary, []

    valid_records_data: List[Dict[str, Any]] = []
    corrupt_paths: List[str] = []
    sha_map: Dict[str, List[str]] = {}
    dim_dist: Dict[str, int] = {}
    chan_dist: Dict[str, int] = {}

    for file_path in discovered_files:
        meta, err = inspect_image_file(file_path)
        if err or meta is None:
            corrupt_paths.append(str(file_path))
            logger.warning("Corrupt or invalid image: %s (%s)", file_path, err)
        else:
            valid_records_data.append(meta)
            sha = meta["sha256"]
            sha_map.setdefault(sha, []).append(str(file_path))

            dim_key = f"{meta['width']}x{meta['height']}"
            dim_dist[dim_key] = dim_dist.get(dim_key, 0) + 1

            chan_key = f"{meta['channels']}_channel"
            chan_dist[chan_key] = chan_dist.get(chan_key, 0) + 1

    # Exact duplicate clusters
    exact_clusters: List[DuplicateCluster] = []
    for sha, paths in sha_map.items():
        if len(paths) > 1:
            exact_clusters.append(
                DuplicateCluster(
                    cluster_id=f"EXACT_DUP_{len(exact_clusters)+1:04d}",
                    cluster_type="exact_duplicate",
                    image_paths=paths,
                    representative_path=paths[0],
                )
            )

    # Near duplicate clusters via dHash
    near_clusters = cluster_near_duplicates(valid_records_data, hamming_threshold=hamming_threshold)

    # Build image records
    image_records: List[ImageRecord] = []
    # Map path to near duplicate cluster ID if any
    path_to_cluster: Dict[str, str] = {}
    for cl in near_clusters:
        for p in cl.image_paths:
            path_to_cluster[p] = cl.cluster_id

    for idx, item in enumerate(valid_records_data):
        p = item["path"]
        img_id = Path(p).stem
        cluster_id = path_to_cluster.get(p)
        record = ImageRecord(
            image_id=img_id,
            dataset_id=dataset_id,
            relative_image_path=str(Path(p).relative_to(target.parent) if target.parent in Path(p).parents else p),
            sha256_checksum=item["sha256"],
            width=item["width"],
            height=item["height"],
            channels=item["channels"],
            file_size_bytes=item["size_bytes"],
            diagnostic_label=-1,
            pseudo_patient_cluster=cluster_id,
        )
        image_records.append(record)

    patient_note = (
        "Dataset files scanned do not contain embedded DICOM patient metadata or explicit patient IDs in filenames. "
        "Perceptual dHash near-duplicate clustering applied to form pseudo-patient groups."
    )

    summary = AuditSummary(
        target_directory=str(target),
        total_files_scanned=len(discovered_files),
        valid_images_found=len(valid_records_data),
        corrupt_or_unreadable_files=len(corrupt_paths),
        corrupt_file_paths=corrupt_paths,
        unique_sha256_count=len(sha_map),
        exact_duplicate_clusters_count=len(exact_clusters),
        exact_duplicate_clusters=exact_clusters,
        near_duplicate_clusters_count=len(near_clusters),
        near_duplicate_clusters=near_clusters,
        dimensions_distribution=dim_dist,
        channels_distribution=chan_dist,
        patient_ids_available=False,
        patient_ids_note=patient_note,
        manifest_reconciled=False,
        status="AUDIT_SUCCESS",
    )

    logger.info(
        "Audit complete: %d valid images, %d corrupt, %d exact duplicate clusters, %d near-duplicate clusters.",
        len(valid_records_data),
        len(corrupt_paths),
        len(exact_clusters),
        len(near_clusters),
    )

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_file = out_path / "data_audit_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary.model_dump(), f, indent=2)

        inventory_file = out_path / "data_inventory.csv"
        save_image_manifest(image_records, inventory_file)
        logger.info("Saved audit report to %s and inventory to %s", report_file, inventory_file)

    return summary, image_records

