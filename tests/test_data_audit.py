"""Unit tests for dataset auditing, deduplication, and near-duplicate detection."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from ovaryvision.data_audit import (
    audit_image_dataset,
    compute_sha256,
    compute_dhash,
    hamming_distance,
)


def _create_synthetic_image(path: Path, width: int = 100, height: int = 100, fill: int = 128) -> None:
    """Helper to write a simple grayscale PNG image."""
    arr = np.full((height, width), fill, dtype=np.uint8)
    # Add a simple geometric pattern
    arr[20:40, 20:40] = 255
    arr[60:80, 60:80] = 0
    img = Image.fromarray(arr)
    img.save(path)


def test_dhash_and_hamming():
    """Verify dHash produces consistent values and expected Hamming distances."""
    img1 = Image.new("L", (100, 100), color=100)
    img2 = Image.new("L", (100, 100), color=100)
    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    assert h1 == h2
    assert hamming_distance(h1, h2) == 0


def test_audit_with_synthetic_images(tmp_path: Path):
    """Test auditing directory containing valid, duplicate, near-duplicate, and corrupt files."""
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    out_dir = tmp_path / "metrics"

    img1 = data_dir / "img1.png"
    _create_synthetic_image(img1, 100, 100, 100)

    # Exact duplicate (copy img1)
    img1_dup = data_dir / "img1_dup.png"
    img1_dup.write_bytes(img1.read_bytes())

    # Near-duplicate: slight variation
    img2 = data_dir / "img2.png"
    arr = np.full((100, 100), 100, dtype=np.uint8)
    arr[20:40, 20:40] = 255
    arr[60:80, 60:80] = 10  # Slight 10-value difference from 0
    Image.fromarray(arr).save(img2)

    # Completely different image
    img3 = data_dir / "img3.png"
    arr_diff = np.random.randint(0, 255, (120, 120), dtype=np.uint8)
    Image.fromarray(arr_diff).save(img3)

    # Corrupt file
    corrupt = data_dir / "corrupt.png"
    corrupt.write_bytes(b"not_a_valid_image_header_bytes")

    summary, records = audit_image_dataset(
        target_dir=data_dir,
        output_dir=out_dir,
        dataset_id="MMOTU_TEST",
        hamming_threshold=4,
    )

    assert summary.status == "AUDIT_SUCCESS"
    assert summary.total_files_scanned == 5
    assert summary.valid_images_found == 4
    assert summary.corrupt_or_unreadable_files == 1
    assert summary.exact_duplicate_clusters_count == 1
    assert summary.exact_duplicate_clusters[0].cluster_type == "exact_duplicate"
    assert len(summary.exact_duplicate_clusters[0].image_paths) == 2
    assert (out_dir / "data_audit_report.json").is_file()
    assert (out_dir / "data_inventory.csv").is_file()


def test_audit_non_existent_directory(tmp_path: Path):
    """Verify auditing a non-existent directory raises FileNotFoundError."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        audit_image_dataset(target_dir=non_existent)


def test_audit_empty_directory(tmp_path: Path):
    """Verify auditing an empty directory returns NO_IMAGES_FOUND summary gracefully."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    summary, records = audit_image_dataset(target_dir=empty_dir)
    assert summary.status == "NO_IMAGES_FOUND"
    assert summary.valid_images_found == 0
    assert len(records) == 0

