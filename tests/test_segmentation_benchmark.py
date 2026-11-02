"""Unit tests for the three-arm segmentation benchmark script and metrics."""

from pathlib import Path
import numpy as np
import pytest

from scripts.run_three_arm_segmentation_benchmark import (
    SegmentationDataset,
    collate_seg_fn,
    compute_boundary_iou,
    compute_segmentation_metrics,
    parse_args,
    validate_environment_and_manifests,
)


def test_parse_args_defaults():
    """Verify that default CLI arguments match the approved benchmark protocol."""
    args = parse_args([])
    assert args.split_manifest == Path("data/manifests/mmotu_2d_segmentation_split.csv")
    assert args.cdfi_manifest == Path("data/manifests/mmotu_2d_cdfi_classification.csv")
    assert args.architecture == "resnet34"
    assert args.resolution == 512
    assert args.batch_size == 8
    assert args.epochs == 40
    assert args.seeds == [42, 1337, 2026]
    assert args.bootstrap_iterations == 2000
    assert "arm_a_raw" in args.arms
    assert "arm_b_inpainted" in args.arms
    assert "arm_c_masked" in args.arms
    assert args.amp is True
    assert not args.dry_run


def test_parse_args_custom_values():
    """Verify custom CLI flag overrides."""
    custom = [
        "--architecture", "convnext_tiny",
        "--resolution", "256",
        "--batch-size", "16",
        "--epochs", "20",
        "--seeds", "100", "200",
        "--bootstrap-iterations", "500",
        "--dry-run",
    ]
    args = parse_args(custom)
    assert args.architecture == "convnext_tiny"
    assert args.resolution == 256
    assert args.batch_size == 16
    assert args.epochs == 20
    assert args.seeds == [100, 200]
    assert args.bootstrap_iterations == 500
    assert args.dry_run is True


def test_validate_manifests_integrity():
    """Verify that the official split and CDFI manifests pass programmatic validation."""
    split_p = Path("data/manifests/mmotu_2d_segmentation_split.csv")
    cdfi_p = Path("data/manifests/mmotu_2d_cdfi_classification.csv")

    split_df, cdfi_df = validate_environment_and_manifests(split_p, cdfi_p)

    # 850/150/469 partition verification
    assert (split_df["segmentation_split"] == "internal_train").sum() == 850
    assert (split_df["segmentation_split"] == "internal_val").sum() == 150
    assert (split_df["segmentation_split"] == "held_out_test").sum() == 469

    # CDFI 57 ambiguous tier verification
    assert (cdfi_df["confidence_tier"] == "LOW_AMBIGUOUS").sum() == 57


def test_compute_segmentation_metrics_identical_masks():
    """Verify metrics when prediction perfectly matches ground truth."""
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[20:40, 20:40] = 1.0  # 20x20 square = 400 px
    caliper = np.zeros((64, 64), dtype=np.uint8)
    caliper[25:35, 25:35] = 1  # 10x10 square overlap = 100 px

    metrics = compute_segmentation_metrics(mask, mask, caliper)

    assert np.isclose(metrics["dsc"], 1.0)
    assert np.isclose(metrics["iou"], 1.0)
    assert np.isclose(metrics["biou_d2"], 1.0)
    assert metrics["is_overlap"] is True
    assert np.isclose(metrics["lpr"], 1.0)
    assert np.isclose(metrics["overlap_ratio"], 100.0 / 400.0)


def test_compute_segmentation_metrics_zero_overlap():
    """Verify metrics when calipers do not intersect the lesion (undefined LPR)."""
    target = np.zeros((64, 64), dtype=np.float32)
    target[10:20, 10:20] = 1.0

    pred = np.zeros((64, 64), dtype=np.float32)
    pred[10:20, 10:20] = 1.0

    caliper = np.zeros((64, 64), dtype=np.uint8)
    caliper[50:60, 50:60] = 1  # Distant caliper, 0 overlap!

    metrics = compute_segmentation_metrics(pred, target, caliper)

    assert np.isclose(metrics["dsc"], 1.0)
    assert metrics["is_overlap"] is False
    assert np.isnan(metrics["lpr"])
    assert metrics["overlap_ratio"] == 0.0


def test_compute_boundary_iou():
    """Verify Boundary IoU behavior with boundary shifts."""
    t = np.zeros((64, 64), dtype=bool)
    t[20:40, 20:40] = True

    # Prediction with boundary shift of 1px
    p = np.zeros((64, 64), dtype=bool)
    p[21:41, 21:41] = True

    biou = compute_boundary_iou(p, t, d=2)
    assert 0.0 < biou < 1.0


def test_paired_spatial_augmentation_invariance_within_epoch():
    """Verify that within the same epoch, Arm A, Arm B, and Arm C receive IDENTICAL spatial transformations."""
    records = [
        {
            "image_id": 1,
            "relative_image_path": "data/raw/mmotu/images/1.JPG",
            "tumor_mask_path": "data/raw/mmotu/tumor_masks/1_binary.PNG",
            "fine_grained_class": "SimpleCyst",
        }
    ]

    ds_a = SegmentationDataset(records, arm="arm_a_raw", resolution=128, is_training=True, seed=42, epoch=5)
    ds_b = SegmentationDataset(records, arm="arm_b_inpainted", resolution=128, is_training=True, seed=42, epoch=5)
    ds_c = SegmentationDataset(records, arm="arm_c_masked", resolution=128, is_training=True, seed=42, epoch=5)

    _, mask_a, meta_a = ds_a[0]
    _, mask_b, meta_b = ds_b[0]
    _, mask_c, meta_c = ds_c[0]

    # In lockstep: the ground-truth mask after identical spatial transformation must be identical
    arr_a = np.asarray(mask_a)
    arr_b = np.asarray(mask_b)
    arr_c = np.asarray(mask_c)
    assert np.array_equal(arr_a, arr_b)
    assert np.array_equal(arr_a, arr_c)
    assert meta_a["epoch"] == 5
    assert meta_b["epoch"] == 5
    assert meta_c["epoch"] == 5


def test_epoch_varying_spatial_augmentations():
    """Verify that spatial transformations vary deterministically across different epochs for the same image."""
    records = [
        {
            "image_id": 10,
            "relative_image_path": "data/raw/mmotu/images/10.JPG",
            "tumor_mask_path": "data/raw/mmotu/tumor_masks/10_binary.PNG",
            "fine_grained_class": "Chocolate_Cyst",
        }
    ]

    ds = SegmentationDataset(records, arm="arm_a_raw", resolution=128, is_training=True, seed=42, epoch=0)

    # Collect masks across 10 epochs
    masks = []
    for ep in range(10):
        ds.set_epoch(ep)
        _, mask, _ = ds[0]
        masks.append(np.asarray(mask))

    # Verify that transformations vary (at least two distinct masks across 10 epochs due to flips/rotations)
    distinct_orientations = 0
    base_mask = masks[0]
    for m in masks[1:]:
        if not np.array_equal(base_mask, m):
            distinct_orientations += 1

    assert distinct_orientations > 0, "Transformations failed to vary across epochs!"


def test_collate_seg_fn():
    """Verify that custom collator stacks tensors and preserves metadata dicts without type errors."""
    mock_batch = [
        (np.zeros((3, 64, 64), dtype=np.float32), np.zeros((1, 64, 64), dtype=np.float32), {"image_id": "1", "fine_grained_class": "SimpleCyst"}),
        (np.ones((3, 64, 64), dtype=np.float32), np.ones((1, 64, 64), dtype=np.float32), {"image_id": "2", "fine_grained_class": "NormalOvary"}),
    ]
    imgs, masks, metas = collate_seg_fn(mock_batch)
    assert imgs.shape == (2, 3, 64, 64)
    assert masks.shape == (2, 1, 64, 64)
    assert len(metas) == 2
    assert metas[0]["image_id"] == "1"
    assert metas[1]["fine_grained_class"] == "NormalOvary"


def test_evaluate_seed42_cli_parser():
    """Verify that evaluate script arguments parse correctly."""
    from scripts.evaluate_seed42_test_cohort import parse_args as parse_eval_args
    args = parse_eval_args(["--resolution", "256", "--seed", "42", "--bootstrap-iterations", "500", "--dry-run"])
    assert args.resolution == 256
    assert args.seed == 42
    assert args.bootstrap_iterations == 500
    assert args.dry_run is True


def test_aggregate_three_seed_benchmark_cli_parser():
    """Verify that aggregate script arguments parse correctly."""
    from scripts.aggregate_three_seed_benchmark import parse_args as parse_agg_args
    args = parse_agg_args(["--seeds", "42", "1337", "2026", "--bootstrap-iterations", "1000"])
    assert args.seeds == [42, 1337, 2026]
    assert args.bootstrap_iterations == 1000


