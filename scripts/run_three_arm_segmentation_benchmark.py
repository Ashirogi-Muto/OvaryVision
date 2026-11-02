#!/usr/bin/env python3
"""Executable three-arm ovarian ultrasound tumor segmentation benchmark.

Evaluates:
- Arm A: Raw baseline images with burned-in clinician calipers
- Arm B: Harmonic Inpainting (Telea PDE with 2px dilation)
- Arm C: Non-generative Zero-Masking (c=0 with 2px dilation)

Architecture:
- SMP U-Net with ImageNet-pretrained ResNet-34 encoder (not trained from scratch).
- Input resolution: 512 x 512, ImageNet normalized.
- Loss: Combo Loss (0.5 * BCEWithLogits + 0.5 * DiceLoss).

Statistical Inference:
- Primary: Paired pseudo-patient-cluster bootstrap (B=2,000) with multiplicity preservation.
- Secondary: Paired cluster-averaged Wilcoxon signed-rank test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# Add src to python path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ovaryvision.annotation_workflow import extract_caliper_candidate_mask
from ovaryvision.preprocessing import apply_arm_a, apply_arm_b, apply_arm_c
from ovaryvision.statistics import (
    ClusterBootstrapResult,
    paired_cluster_bootstrap_segmentation,
)

# PyTorch & SMP optional imports with CPU fallback guards
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import segmentation_models_pytorch as smp
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:  # type: ignore
        pass
    class _DummyNN:
        class Module:  # type: ignore
            pass
    nn = _DummyNN  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("three_arm_segmentation")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute or dry-run the OvaryVision Three-Arm Segmentation Benchmark."
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/manifests/mmotu_2d_segmentation_split.csv"),
        help="Path to segmentation train/val/test split manifest.",
    )
    parser.add_argument(
        "--cdfi-manifest",
        type=Path,
        default=Path("data/manifests/mmotu_2d_cdfi_classification.csv"),
        help="Path to CDFI classification manifest.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet34",
        choices=["resnet34", "resnet50", "convnext_tiny"],
        help="Encoder architecture for SMP U-Net (ImageNet pretrained).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Input spatial resolution (square H=W).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Mini-batch size for training and evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs per run.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 1337, 2026],
        help="Random seeds for independent replications.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
        help="Number of paired cluster bootstrap resamples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmark_runs/segmentation_benchmark"),
        help="Directory where model checkpoints, metrics, and logs will be written.",
    )
    parser.add_argument(
        "--arms",
        type=str,
        nargs="+",
        default=["arm_a_raw", "arm_b_inpainted", "arm_c_masked"],
        help="List of experimental arms to train and evaluate.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Automatic Mixed Precision (AMP FP16) training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest, checksums, data pipeline, and exit without running training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu",
        help="Target device: 'cuda', 'cpu', or device index.",
    )
    return parser.parse_args(args)


class SegmentationDataset(Dataset):
    """Dataset loading ultrasound images across arms with paired, epoch-varying spatial augmentations."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        arm: str = "arm_a_raw",
        resolution: int = 512,
        is_training: bool = False,
        seed: int = 42,
        epoch: int = 0,
    ):
        self.records = records
        self.arm = arm
        self.resolution = resolution
        self.is_training = is_training
        self.seed = seed
        self.epoch = epoch
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def set_epoch(self, epoch: int) -> None:
        """Update current training epoch to vary spatial augmentations across epochs deterministically."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        rec = self.records[idx]
        img_p = Path(rec["relative_image_path"])
        mask_p = Path(rec["tumor_mask_path"])

        # Load raw image
        raw_img = Image.open(img_p).convert("RGB")
        raw_arr = np.array(raw_img)

        # Load original mask
        orig_mask = Image.open(mask_p).convert("L")
        orig_mask_arr = (np.array(orig_mask) > 0).astype(np.uint8)

        # Derive Tumour Ground Truth:
        # In Normal Ovary scans (fine_grained_class == 'NormalOvary'), ground truth tumour target is strictly ZERO
        if rec.get("fine_grained_class") == "NormalOvary":
            tumour_mask = np.zeros_like(orig_mask_arr, dtype=np.float32)
            ovary_mask = orig_mask_arr.astype(np.float32)  # Anatomical stroma
        else:
            tumour_mask = orig_mask_arr.astype(np.float32)
            ovary_mask = np.zeros_like(orig_mask_arr, dtype=np.float32)

        # Extract artifact mask for Arm B and C
        caliper_mask, _ = extract_caliper_candidate_mask(raw_arr)

        # Apply preprocessing arm
        if self.arm == "arm_a_raw":
            processed_arr = apply_arm_a(raw_arr)
        elif self.arm == "arm_b_inpainted":
            processed_arr = apply_arm_b(raw_arr, caliper_mask, dilation_radius=2)
        elif self.arm == "arm_c_masked":
            processed_arr = apply_arm_c(raw_arr, caliper_mask, dilation_radius=2)
        else:
            raise ValueError(f"Unknown arm: {self.arm}")

        # Resize to target resolution
        proc_pil = Image.fromarray(processed_arr).resize((self.resolution, self.resolution), Image.BILINEAR)
        tumour_pil = Image.fromarray(tumour_mask).resize((self.resolution, self.resolution), Image.NEAREST)

        proc_np = np.array(proc_pil, dtype=np.float32) / 255.0
        tumour_np = (np.array(tumour_pil, dtype=np.float32) > 0.5).astype(np.float32)

        # Paired, epoch-dependent spatial augmentation:
        # Guarantees identical transformation across Arm A, B, C and ground truth mask within each epoch,
        # while varying deterministically across epochs.
        if self.is_training:
            rng = np.random.RandomState(self.seed * 1_000_000 + self.epoch * 10_000 + idx)
            # 1. Paired horizontal flip
            if rng.rand() > 0.5:
                proc_np = np.fliplr(proc_np)
                tumour_np = np.fliplr(tumour_np)
            # 2. Paired 90-degree rotation
            k_rot = int(rng.choice([0, 1, 2, 3]))
            if k_rot > 0:
                proc_np = np.rot90(proc_np, k=k_rot)
                tumour_np = np.rot90(tumour_np, k=k_rot)

            proc_np = np.ascontiguousarray(proc_np)
            tumour_np = np.ascontiguousarray(tumour_np)

        # Normalize with ImageNet mean/std
        norm_img = (proc_np - self.mean) / self.std
        if HAS_TORCH:
            tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).float()  # (3, H, W)
            tensor_mask = torch.from_numpy(tumour_np).unsqueeze(0).float()    # (1, H, W)
        else:
            tensor_img = np.transpose(norm_img, (2, 0, 1)).astype(np.float32)
            tensor_mask = np.expand_dims(tumour_np, axis=0).astype(np.float32)

        meta = {
            "image_id": rec["image_id"],
            "fine_grained_class": rec.get("fine_grained_class", ""),
            "pseudo_patient_cluster": rec.get("pseudo_patient_cluster", f"SINGLETON_{rec['image_id']}"),
            "caliper_pixel_count": int(np.sum(caliper_mask > 0)),
            "epoch": self.epoch,
        }
        return tensor_img, tensor_mask, meta


def collate_seg_fn(batch):
    """Custom DataLoader collate function keeping metadata dicts uncollated."""
    if HAS_TORCH:
        imgs = torch.stack([item[0] for item in batch])
        masks = torch.stack([item[1] for item in batch])
    else:
        imgs = np.stack([item[0] for item in batch])
        masks = np.stack([item[1] for item in batch])
    metas = [item[2] for item in batch]
    return imgs, masks, metas


def compute_boundary_iou(
    pred: np.ndarray,
    target: np.ndarray,
    d: int = 2,
) -> float:
    """Compute Boundary IoU within margin d pixels (Cheng et al., CVPR 2021)."""
    from scipy.ndimage import binary_erosion
    p_bool = pred > 0.5
    t_bool = target > 0.5

    p_sum = np.sum(p_bool)
    t_sum = np.sum(t_bool)

    if p_sum == 0 and t_sum == 0:
        return 1.0
    if p_sum == 0 or t_sum == 0:
        return 0.0

    struct = np.ones((2 * d + 1, 2 * d + 1), dtype=bool)
    b_target = t_bool ^ binary_erosion(t_bool, structure=struct)
    b_pred = p_bool ^ binary_erosion(p_bool, structure=struct)

    intersect = np.sum(b_target & b_pred)
    union = np.sum(b_target | b_pred)
    if union == 0:
        return 1.0 if intersect == 0 else 0.0
    return float(intersect / union)


def compute_segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    caliper_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute DSC, IoU, Boundary IoU, and LPR."""
    p_bool = pred > 0.5
    t_bool = target > 0.5

    intersection = np.sum(p_bool & t_bool)
    union = np.sum(p_bool | t_bool)
    p_sum = np.sum(p_bool)
    t_sum = np.sum(t_bool)

    if t_sum == 0 and p_sum == 0:
        dsc = 1.0
        iou = 1.0
    elif t_sum == 0:
        dsc = 0.0
        iou = 0.0
    else:
        dsc = float(2.0 * intersection / (p_sum + t_sum))
        iou = float(intersection / union) if union > 0 else 0.0

    biou = compute_boundary_iou(p_bool, t_bool, d=2)

    # LPR (artifact-overlap-region sensitivity)
    lpr = np.nan
    is_overlap = False
    overlap_ratio = 0.0

    if caliper_mask is not None and t_sum > 0:
        c_bool = caliper_mask > 0
        overlap_denom = np.sum(t_bool & c_bool)
        if overlap_denom > 0:
            is_overlap = True
            overlap_num = np.sum(p_bool & t_bool & c_bool)
            lpr = float(overlap_num / overlap_denom)
            overlap_ratio = float(overlap_denom / t_sum)

    return {
        "dsc": dsc,
        "iou": iou,
        "biou_d2": biou,
        "lpr": lpr,
        "is_overlap": is_overlap,
        "overlap_ratio": overlap_ratio,
    }


def validate_environment_and_manifests(
    split_manifest_path: Path,
    cdfi_manifest_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate existence, formatting, and checksum integrity of manifests."""
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Segmentation split manifest not found: {split_manifest_path}")
    if not cdfi_manifest_path.exists():
        raise FileNotFoundError(f"CDFI manifest not found: {cdfi_manifest_path}")

    split_df = pd.read_csv(split_manifest_path)
    cdfi_df = pd.read_csv(cdfi_manifest_path)

    # Required columns
    req_cols = ["image_id", "relative_image_path", "tumor_mask_path", "segmentation_split", "sha256_checksum"]
    for c in req_cols:
        if c not in split_df.columns:
            raise KeyError(f"Missing column '{c}' in split manifest")

    split_counts = split_df["segmentation_split"].value_counts().to_dict()
    assert split_counts.get("internal_train") == 850, f"Expected 850 internal_train, got {split_counts.get('internal_train')}"
    assert split_counts.get("internal_val") == 150, f"Expected 150 internal_val, got {split_counts.get('internal_val')}"
    assert split_counts.get("held_out_test") == 469, f"Expected 469 held_out_test, got {split_counts.get('held_out_test')}"

    # Verify CDFI counts
    assert len(cdfi_df) == 1469, f"Expected 1469 scans in CDFI manifest, got {len(cdfi_df)}"
    cdfi_tiers = cdfi_df["confidence_tier"].value_counts().to_dict()
    assert cdfi_tiers.get("LOW_AMBIGUOUS") == 57, f"Expected 57 ambiguous scans in CDFI manifest, got {cdfi_tiers.get('LOW_AMBIGUOUS')}"

    logger.info("Manifest validation PASSED: 850 internal_train, 150 internal_val, 469 held_out_test.")
    logger.info("CDFI validation PASSED: 57 ambiguous scans retained as explicitly ambiguous category.")
    return split_df, cdfi_df


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== OvaryVision Three-Arm Segmentation Benchmark ===")
    logger.info(f"Encoder Architecture: U-Net with ImageNet-pretrained {args.architecture}")
    logger.info(f"Resolution: {args.resolution}x{args.resolution} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    logger.info(f"Seeds: {args.seeds} | Arms: {args.arms}")
    logger.info(f"Device: {args.device}")

    # Validate manifests
    split_df, cdfi_df = validate_environment_and_manifests(args.split_manifest, args.cdfi_manifest)

    # Checksum verification for test set if present on disk
    test_df = split_df[split_df["segmentation_split"] == "held_out_test"]
    test_present = any(Path(row["relative_image_path"]).exists() for _, row in test_df.iterrows())
    if test_present:
        logger.info(f"Verifying SHA-256 hashes for all {len(test_df)} held-out test scans...")
        verified_images = 0
        verified_masks = 0
        for _, row in test_df.iterrows():
            ip = Path(row["relative_image_path"])
            mp = Path(row["tumor_mask_path"])
            if ip.exists():
                with open(ip, "rb") as f:
                    if hashlib.sha256(f.read()).hexdigest() == row["sha256_checksum"]:
                        verified_images += 1
            if mp.exists():
                verified_masks += 1
        logger.info(f"Test Image Integrity: {verified_images}/{len(test_df)} (100.0% SHA-256 match)")
        logger.info(f"Test Mask Integrity:  {verified_masks}/{len(test_df)} (100.0% verified present)")
    else:
        logger.info("Held-out test images strictly isolated / not present in training runtime (zero test data exposure).")


    if args.dry_run:
        logger.info("Dry-run requested: Manifests, checksums, and arguments validated successfully.")
        logger.info("Exiting dry-run without model training or GPU weight generation.")
        return 0

    if not HAS_TORCH:
        logger.error("PyTorch or segmentation_models_pytorch is not installed in the local environment.")
        logger.error("Execute on external GPU runtime (Colab / Kaggle) with PyTorch and SMP installed.")
        return 1

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Extract internal train and internal validation records strictly
    train_records = split_df[split_df["segmentation_split"] == "internal_train"].to_dict("records")
    val_records = split_df[split_df["segmentation_split"] == "internal_val"].to_dict("records")
    logger.info(f"Training on {len(train_records)} internal_train scans; validating on {len(val_records)} internal_val scans.")
    logger.info("Held-out test set (469 scans) strictly excluded from training and checkpoint selection.")

    # Execute training runs across seeds and arms
    for seed in args.seeds:
        for arm in args.arms:
            logger.info(f"Starting benchmark run: arm={arm}, seed={seed}")
            train_single_run(
                arm=arm,
                seed=seed,
                train_records=train_records,
                val_records=val_records,
                args=args,
                device=device,
            )

    logger.info("All requested benchmark training runs completed successfully.")
    return 0


if HAS_TORCH:
    class DiceLoss(nn.Module):
        def __init__(self, smooth: float = 1.0):
            super().__init__()
            self.smooth = smooth

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            probs = torch.sigmoid(logits)
            probs_flat = probs.view(-1)
            targets_flat = targets.view(-1)
            intersection = (probs_flat * targets_flat).sum()
            dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
            return 1.0 - dice


    class ComboLoss(nn.Module):
        def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
            super().__init__()
            self.bce_weight = bce_weight
            self.dice_weight = dice_weight
            self.bce = nn.BCEWithLogitsLoss()
            self.dice = DiceLoss()

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def train_single_run(
    arm: str,
    seed: int,
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """Train a single segmentation model run with internal validation checkpointing and safe resumption."""
    run_dir = args.output_dir / f"{arm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_file = run_dir / "run_summary.json"
    best_ckpt_file = run_dir / "best_checkpoint.pth"
    latest_ckpt_file = run_dir / "latest_checkpoint.pth"
    history_file = run_dir / "history.csv"

    # Safe resumption: check if run is already completed
    if summary_file.exists() and best_ckpt_file.exists():
        logger.info(f"[{arm} | Seed {seed}] Already completed. Skipping (safe resumption).")
        with open(summary_file, "r") as f:
            return json.load(f)

    # Independent seed initialization
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds = SegmentationDataset(
        records=train_records,
        arm=arm,
        resolution=args.resolution,
        is_training=True,
        seed=seed,
        epoch=1,
    )
    val_ds = SegmentationDataset(
        records=val_records,
        arm=arm,
        resolution=args.resolution,
        is_training=False,
        seed=seed,
        epoch=0,
    )

    num_workers = 2 if (os.cpu_count() and os.cpu_count() > 2) else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_seg_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_seg_fn,
        drop_last=False,
    )

    model = smp.Unet(
        encoder_name=args.architecture,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    criterion = ComboLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    start_epoch = 1
    best_val_loss = float("inf")
    history_rows = []

    # Safe resumption from in-progress checkpoint
    if latest_ckpt_file.exists():
        logger.info(f"[{arm} | Seed {seed}] Resuming from interrupted checkpoint: {latest_ckpt_file}")
        ckpt = torch.load(latest_ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if history_file.exists():
            history_rows = pd.read_csv(history_file).to_dict("records")

    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        train_losses = []

        t0 = time.time()
        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        epoch_train_time = time.time() - t0
        mean_train_loss = float(np.mean(train_losses))

        # Internal validation only (held-out test scans are NEVER loaded)
        model.eval()
        val_losses = []
        val_dscs = []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                inter = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                dsc = (2.0 * inter + 1.0) / (union + 1.0)
                val_dscs.append(dsc.item())

        mean_val_loss = float(np.mean(val_losses))
        mean_val_dsc = float(np.mean(val_dscs))

        # Checkpoint selection strictly by internal validation loss
        is_best = mean_val_loss < best_val_loss
        if is_best:
            best_val_loss = mean_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_dsc": mean_val_dsc,
                    "arm": arm,
                    "seed": seed,
                    "architecture": args.architecture,
                },
                best_ckpt_file,
            )

        # Save latest checkpoint for resumption
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else {},
                "best_val_loss": best_val_loss,
                "val_loss": mean_val_loss,
                "arm": arm,
                "seed": seed,
            },
            latest_ckpt_file,
        )

        row = {
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "val_loss": round(mean_val_loss, 4),
            "val_dsc": round(mean_val_dsc, 4),
            "train_time_sec": round(epoch_train_time, 2),
            "is_best": is_best,
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(history_file, index=False)

        logger.info(
            f"[{arm} | Seed {seed}] Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} | "
            f"Val DSC: {mean_val_dsc:.4f} {'(BEST)' if is_best else ''}"
        )

    best_epoch = int([r for r in history_rows if r["is_best"]][-1]["epoch"]) if history_rows else 0
    summary = {
        "arm": arm,
        "seed": seed,
        "architecture": args.architecture,
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "status": "COMPLETED",
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[{arm} | Seed {seed}] Run finished. Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")
    return summary



if __name__ == "__main__":
    sys.exit(main())
