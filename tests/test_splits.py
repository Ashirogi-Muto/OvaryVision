"""Tests for group-stratified dataset splitting and leakage verification."""

from pathlib import Path
import pytest

from ovaryvision.manifests import ImageRecord, load_split_manifest, validate_manifest_leakage
from ovaryvision.splits import (
    SplitRatio,
    compute_split_summary,
    generate_and_save_splits,
    generate_group_stratified_split,
)


def _make_dummy_records(num_records: int = 40, num_clusters: int = 10) -> list[ImageRecord]:
    """Helper to create dummy ImageRecord entries with cluster and label assignments."""
    records = []
    for i in range(num_records):
        cluster_id = f"CLUSTER_{i % num_clusters:04d}"
        # Cluster 0..4 benign, 5..9 malignant
        label = 0 if (i % num_clusters) < (num_clusters // 2) else 1
        rec = ImageRecord(
            image_id=f"IMG_{i:04d}",
            dataset_id="MMOTU_2D",
            relative_image_path=f"data/raw/mmotu/images/img_{i:04d}.jpg",
            sha256_checksum="a" * 64,
            width=512,
            height=512,
            channels=1,
            file_size_bytes=2048,
            diagnostic_label=label,
            pseudo_patient_cluster=cluster_id,
            has_visible_artifacts=False,
        )
        records.append(rec)
    return records


def test_split_ratio_validation():
    """Verify SplitRatio enforces summation to 1.0."""
    valid = SplitRatio(train=0.70, val=0.15, test=0.15)
    assert valid.train == 0.70

    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        SplitRatio(train=0.80, val=0.20, test=0.20)


def test_group_stratified_split_zero_leakage():
    """Verify generated splits have zero image and zero cluster leakage."""
    records = _make_dummy_records(num_records=50, num_clusters=10)
    splits = generate_group_stratified_split(records, seed=42)

    assert len(splits) == 50
    # Must pass validate_manifest_leakage without raising ValueError
    assert validate_manifest_leakage(splits) is True

    # Manually verify no cluster spans multiple roles
    cluster_to_role = {}
    for s in splits:
        if s.pseudo_patient_cluster in cluster_to_role:
            assert cluster_to_role[s.pseudo_patient_cluster] == s.split_role
        else:
            cluster_to_role[s.pseudo_patient_cluster] = s.split_role


def test_stratification_and_summary():
    """Verify summary correctly counts class distributions and roles."""
    records = _make_dummy_records(num_records=60, num_clusters=12)
    splits = generate_group_stratified_split(records, seed=42)
    summary = compute_split_summary(splits)

    assert summary.total_images == 60
    assert summary.total_clusters == 12
    assert summary.train.total > summary.val.total
    assert summary.train.total > summary.test_internal.total
    assert summary.train.benign > 0
    assert summary.train.malignant > 0


def test_seed_determinism():
    """Verify same seed produces identical split and different seed produces different split."""
    records = _make_dummy_records(num_records=40, num_clusters=10)
    split_a = generate_group_stratified_split(records, seed=42)
    split_b = generate_group_stratified_split(records, seed=42)
    split_c = generate_group_stratified_split(records, seed=1337)

    assignments_a = [s.split_role for s in split_a]
    assignments_b = [s.split_role for s in split_b]
    assignments_c = [s.split_role for s in split_c]

    assert assignments_a == assignments_b
    assert assignments_a != assignments_c


def test_external_records_assigned():
    """Verify external records are assigned to 'test_external'."""
    internal = _make_dummy_records(num_records=30, num_clusters=6)
    external = [
        ImageRecord(
            image_id=f"EXT_{i:04d}",
            dataset_id="OvaTUS_V1",
            relative_image_path=f"data/raw/ovatus/img_{i}.jpg",
            sha256_checksum="b" * 64,
            width=512,
            height=512,
            channels=1,
            file_size_bytes=2048,
            diagnostic_label=0,
            has_visible_artifacts=False,
        )
        for i in range(5)
    ]

    splits = generate_group_stratified_split(internal, seed=42, external_records=external)
    assert len(splits) == 35

    external_splits = [s for s in splits if s.split_role == "test_external"]
    assert len(external_splits) == 5
    for s in external_splits:
        assert s.image_id.startswith("EXT_")


def test_generate_and_save_splits(tmp_path: Path):
    """Verify batch split generation across multiple seeds and CSV export."""
    records = _make_dummy_records(num_records=40, num_clusters=8)
    seeds = [42, 2026]
    saved = generate_and_save_splits(records, output_dir=tmp_path, seeds=seeds)

    assert len(saved) == 2
    for seed, path in saved.items():
        assert path.exists()
        loaded = load_split_manifest(path)
        assert len(loaded) == 40
        assert all(s.split_seed == seed for s in loaded)
        validate_manifest_leakage(loaded)

