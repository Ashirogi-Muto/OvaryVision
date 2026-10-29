"""Unit tests for image manifests, split schemas, and leakage detection."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from ovaryvision.manifests import (
    ImageRecord,
    SplitRecord,
    load_image_manifest,
    save_image_manifest,
    load_split_manifest,
    save_split_manifest,
    validate_manifest_leakage,
)

SAMPLE_SHA256 = "a" * 64


def test_image_record_validation():
    """Verify ImageRecord enforces sha256 format and label constraints."""
    # Valid record
    rec = ImageRecord(
        image_id="IMG_001",
        relative_image_path="data/raw/sample.png",
        sha256_checksum=SAMPLE_SHA256,
        width=800,
        height=400,
        file_size_bytes=1000,
        diagnostic_label=1,
    )
    assert rec.image_id == "IMG_001"
    assert rec.diagnostic_label == 1

    # Invalid sha256 (not 64 characters)
    with pytest.raises(ValidationError):
        ImageRecord(
            image_id="IMG_002",
            relative_image_path="data/raw/sample.png",
            sha256_checksum="invalid_hash",
            width=800,
            height=400,
            file_size_bytes=1000,
            diagnostic_label=0,
        )

    # Invalid diagnostic label (not -1, 0, or 1)
    with pytest.raises(ValidationError):
        ImageRecord(
            image_id="IMG_003",
            relative_image_path="data/raw/sample.png",
            sha256_checksum=SAMPLE_SHA256,
            width=800,
            height=400,
            file_size_bytes=1000,
            diagnostic_label=5,
        )


def test_conflating_masks_is_prohibited():
    """Verify that pointing artifact_mask_path and tumor_mask_path to the same file raises an error."""
    with pytest.raises(ValidationError, match="Severe integrity violation"):
        ImageRecord(
            image_id="IMG_BAD_MASKS",
            relative_image_path="data/raw/sample.png",
            sha256_checksum=SAMPLE_SHA256,
            width=800,
            height=400,
            file_size_bytes=1000,
            artifact_mask_path="data/masks/mask_01.png",
            tumor_mask_path="data/masks/mask_01.png",  # Same file!
        )


def test_manifest_csv_roundtrip(tmp_path: Path):
    """Verify saving and loading ImageRecords to CSV produces identical data."""
    csv_file = tmp_path / "test_inventory.csv"
    records = [
        ImageRecord(
            image_id="IMG_101",
            dataset_id="MMOTU_2D",
            relative_image_path="mmotu/img101.jpg",
            sha256_checksum=SAMPLE_SHA256,
            width=640,
            height=480,
            channels=1,
            file_size_bytes=54321,
            diagnostic_label=0,
            has_visible_artifacts=True,
            artifact_mask_path="masks/art_101.png",
            tumor_mask_path="masks/tumor_101.png",
        ),
        ImageRecord(
            image_id="IMG_102",
            dataset_id="MMOTU_2D",
            relative_image_path="mmotu/img102.jpg",
            sha256_checksum="b" * 64,
            width=800,
            height=600,
            channels=1,
            file_size_bytes=65432,
            diagnostic_label=1,
            has_visible_artifacts=False,
        ),
    ]

    save_image_manifest(records, csv_file)
    loaded = load_image_manifest(csv_file)
    assert len(loaded) == 2
    assert loaded[0].image_id == "IMG_101"
    assert loaded[0].has_visible_artifacts is True
    assert loaded[0].artifact_mask_path == "masks/art_101.png"
    assert loaded[1].image_id == "IMG_102"
    assert loaded[1].has_visible_artifacts is False


def test_split_leakage_detection():
    """Verify that validate_manifest_leakage flags pseudo-patient clusters spanning multiple folds."""
    # Leak-free split
    valid_splits = [
        SplitRecord(image_id="IMG_1", split_role="train", split_seed=42, pseudo_patient_cluster="C1", diagnostic_label=0),
        SplitRecord(image_id="IMG_2", split_role="train", split_seed=42, pseudo_patient_cluster="C1", diagnostic_label=0),
        SplitRecord(image_id="IMG_3", split_role="val", split_seed=42, pseudo_patient_cluster="C2", diagnostic_label=1),
        SplitRecord(image_id="IMG_4", split_role="test_internal", split_seed=42, pseudo_patient_cluster="C3", diagnostic_label=1),
    ]
    assert validate_manifest_leakage(valid_splits) is True

    # Leaking split (C1 in train AND val)
    leaking_splits = [
        SplitRecord(image_id="IMG_1", split_role="train", split_seed=42, pseudo_patient_cluster="C1", diagnostic_label=0),
        SplitRecord(image_id="IMG_2", split_role="val", split_seed=42, pseudo_patient_cluster="C1", diagnostic_label=0),
    ]
    with pytest.raises(ValueError, match="Cluster data leakage detected"):
        validate_manifest_leakage(leaking_splits)

    # Direct image leakage (same image_id in train AND val)
    direct_leaking = [
        SplitRecord(image_id="IMG_SHARED", split_role="train", split_seed=42, diagnostic_label=0),
        SplitRecord(image_id="IMG_SHARED", split_role="val", split_seed=42, diagnostic_label=0),
    ]
    with pytest.raises(ValueError, match="Direct image leakage detected"):
        validate_manifest_leakage(direct_leaking)

    # Duplicate image in same role
    duplicate_image = [
        SplitRecord(image_id="IMG_DUP", split_role="train", split_seed=42, diagnostic_label=0),
        SplitRecord(image_id="IMG_DUP", split_role="train", split_seed=42, diagnostic_label=0),
    ]
    with pytest.raises(ValueError, match="Duplicate image entry detected"):
        validate_manifest_leakage(duplicate_image)

