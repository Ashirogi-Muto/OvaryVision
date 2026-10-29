#!/usr/bin/env python3
from __future__ import annotations
"""Self-contained GPU Training & Evaluation Pipeline for OvaryVision.

This script executes deep CNN model training across the three experimental arms
(Arm A: Raw, Arm B: Inpainted, Arm C: Masked) on external GPU runtimes (Google Colab, Kaggle).

Key Features:
- Models: ConvNeXt-Tiny and EfficientNet-B0 with ImageNet-1K pretrained weights.
- Loss: Class-Weighted Focal Loss (gamma=2.0) with automated inverse class weights.
- Optimization: AdamW with Cosine Annealing learning rate schedule.
- Validation Monitoring: Early stopping based on validation AUROC.
- Post-Hoc Calibration: Fits Temperature Scaling (T*) strictly on validation logits.
- Output Artifacts: Checkpoint weights, prediction CSV, and evaluation JSON report.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Verify PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:  # type: ignore
        pass
    class _DummyNN:
        class Module:  # type: ignore
            pass
    nn = _DummyNN  # type: ignore


def set_seed(seed: int = 42) -> None:
    """Enforce complete determinism across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class UltrasoundDataset(Dataset):
    """PyTorch Dataset loading preprocessed ovarian ultrasound scans."""

    def __init__(
        self,
        records: List[Dict[str, str]],
        data_dir: Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.records = records
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        image_id = rec["image_id"]
        rel_path = rec.get("relative_image_path") or f"{image_id}.png"
        img_full_path = self.data_dir / rel_path

        # Fallback to direct image_id search if relative path is missing
        if not img_full_path.is_file():
            candidates = list(self.data_dir.glob(f"**/{image_id}.*"))
            if candidates:
                img_full_path = candidates[0]
            else:
                raise FileNotFoundError(f"Ultrasound image not found for image_id '{image_id}' in {self.data_dir}")

        image = Image.open(img_full_path).convert("RGB")
        label = int(rec["diagnostic_label"])

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, label, image_id


class FocalLoss(nn.Module):
    """Class-weighted binary/multiclass Focal Loss with focusing parameter gamma."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits shape: (N, C), targets shape: (N,)
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def build_model(backbone: str, num_classes: int = 2) -> nn.Module:
    """Instantiate backbone CNN with ImageNet pre-trained weights.

    Approved Benchmark Architectures (Section 15.1):
    - convnext_small: Inverted bottleneck CNN with 7x7 depthwise convolutions.
    - efficientnet_b3: Compound-scaled CNN with MBConv inverted residual blocks.

    Smoke-Test / Lightweight Options:
    - convnext_tiny: Fast smoke-test variant.
    - efficientnet_b0: Fast smoke-test variant.
    """
    backbone_clean = backbone.lower().strip()
    if backbone_clean in ("convnext_small", "convnext"):
        try:
            weights = models.ConvNeXt_Small_Weights.DEFAULT
            model = models.convnext_small(weights=weights)
        except AttributeError:
            model = models.convnext_small(pretrained=True)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif backbone_clean in ("efficientnet_b3", "efficientnet"):
        try:
            weights = models.EfficientNet_B3_Weights.DEFAULT
            model = models.efficientnet_b3(weights=weights)
        except AttributeError:
            model = models.efficientnet_b3(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif backbone_clean == "convnext_tiny":
        try:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT
            model = models.convnext_tiny(weights=weights)
        except AttributeError:
            model = models.convnext_tiny(pretrained=True)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif backbone_clean == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except AttributeError:
            model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Must be 'convnext_small', 'efficientnet_b3', "
            "'convnext_tiny', or 'efficientnet_b0'."
        )

    return model


def compute_roc_auc_numpy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute binary ROC AUC from continuous predictions via Mann-Whitney U."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ties
    unique, idx, counts = np.unique(scores[order], return_index=True, return_counts=True)
    for i, c in zip(idx, counts):
        if c > 1:
            ranks[order[i : i + c]] = np.mean(np.arange(i + 1, i + c + 1))
    u_stat = np.sum(ranks[labels == 1]) - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Fit optimal temperature scalar T* on validation fold logits via bounded grid/scalar minimization."""
    z = val_logits
    y = val_labels

    def nll(t: float) -> float:
        scaled = z / max(t, 1e-4)
        max_val = np.maximum(scaled, 0.0)
        loss = max_val - scaled * y + np.log1p(np.exp(-np.abs(scaled)))
        return float(np.mean(loss))

    best_t = 1.0
    best_loss = float("inf")
    # Search grid from 0.05 to 5.0 in 0.01 increments
    for t_cand in np.arange(0.05, 5.0, 0.01):
        loss_val = nll(t_cand)
        if loss_val < best_loss:
            best_loss = loss_val
            best_t = float(t_cand)

    return best_t


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[int] = []
    all_ids: List[str] = []

    with torch.no_grad():
        for images, labels, ids in loader:
            images = images.to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            all_ids.extend(ids)

    logits_arr = np.concatenate(all_logits, axis=0)  # shape (N, 2)
    labels_arr = np.array(all_labels, dtype=int)
    # Logit difference z = logit_1 - logit_0 represents log-odds of malignant class
    binary_logits = logits_arr[:, 1] - logits_arr[:, 0]
    return binary_logits, logits_arr, labels_arr, all_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="OvaryVision GPU Training & Benchmark Pipeline")
    parser.add_argument("--manifest", type=str, required=True, help="Path to split manifest CSV")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--arm", type=str, default="arm_a_raw", choices=["arm_a_raw", "arm_b_inpainted", "arm_c_masked"])
    parser.add_argument(
        "--backbone",
        type=str,
        default="convnext_small",
        choices=["convnext_small", "efficientnet_b3", "convnext_tiny", "efficientnet_b0"],
        help="CNN backbone architecture (convnext_small and efficientnet_b3 are approved benchmark defaults)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark_runs", help="Output directory")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch and torchvision are required to run gpu_train.py.", file=sys.stderr)
        print("Install via: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
        sys.exit(1)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware device: {device} | CUDA Available: {torch.cuda.is_available()}")

    out_dir = Path(args.output_dir) / f"{args.arm}_{args.backbone}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Split Manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    train_recs: List[Dict[str, str]] = []
    val_recs: List[Dict[str, str]] = []
    test_recs: List[Dict[str, str]] = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row["split_role"].lower().strip()
            if role == "train":
                train_recs.append(row)
            elif role == "val":
                val_recs.append(row)
            elif role in ("test_internal", "test", "test_external"):
                test_recs.append(row)

    print(f"Loaded manifest partitions: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")

    # 2. Compute Inverse Class Weights for Focal Loss
    train_labels = [int(r["diagnostic_label"]) for r in train_recs]
    n_train = len(train_labels)
    n_pos = sum(train_labels)
    n_neg = n_train - n_pos
    alpha_0 = n_train / (2.0 * max(n_neg, 1))
    alpha_1 = n_train / (2.0 * max(n_pos, 1))
    alpha_tensor = torch.tensor([alpha_0, alpha_1], dtype=torch.float32).to(device)
    print(f"Class weighting: Benign={alpha_0:.3f}, Malignant={alpha_1:.3f}")

    # 3. Data Transformations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir = Path(args.data_dir)
    train_ds = UltrasoundDataset(train_recs, data_dir, transform=train_transform)
    val_ds = UltrasoundDataset(val_recs, data_dir, transform=eval_transform)
    test_ds = UltrasoundDataset(test_recs, data_dir, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 4. Model, Loss, Optimizer
    model = build_model(args.backbone, num_classes=2).to(device)
    criterion = FocalLoss(alpha=alpha_tensor, gamma=args.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Training Loop with Early Stopping on Validation AUROC
    best_val_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = out_dir / "best_model.pth"

    print("Beginning training across epochs...")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        loss_val = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        val_bin_logits, _, val_y, _ = evaluate(model, val_loader, device)
        val_probs = 1.0 / (1.0 + np.exp(-val_bin_logits))
        val_auc = compute_roc_auc_numpy(val_probs, val_y)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {loss_val:.4f} | Val AUROC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with AUROC={best_val_auc:.4f}")
                break

    training_time_sec = time.time() - start_time
    print(f"Training completed in {training_time_sec:.1f}s. Loading best weights from epoch {best_epoch}...")

    # 6. Evaluation & Post-Hoc Temperature Scaling
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Validation logits for fitting T*
    val_bin_logits, _, val_y, _ = evaluate(model, val_loader, device)
    optimal_t = fit_temperature(val_bin_logits, val_y)
    print(f"Fitted Validation Temperature Scaling: T* = {optimal_t:.4f}")

    # Test set evaluation
    test_bin_logits, test_raw_logits, test_y, test_ids = evaluate(model, test_loader, device)
    uncal_probs = 1.0 / (1.0 + np.exp(-test_bin_logits))
    cal_probs = 1.0 / (1.0 + np.exp(-(test_bin_logits / optimal_t)))
    test_auc = compute_roc_auc_numpy(cal_probs, test_y)
    test_brier = float(np.mean((cal_probs - test_y) ** 2))
    predicted_labels = (cal_probs >= 0.5).astype(int)

    print(f"Test Set AUROC: {test_auc:.4f} | Calibrated Brier Score: {test_brier:.4f}")

    # 7. Export Predictions CSV
    pred_csv_path = out_dir / "predictions.csv"
    with open(pred_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id", "true_label", "raw_logit_0", "raw_logit_1", "binary_logit",
            "uncalibrated_prob", "calibrated_prob", "predicted_class",
            "arm", "backbone", "seed"
        ])
        for img_id, y_true, (logit_0, logit_1), bin_logit, p_uncal, p_cal, p_class in zip(
            test_ids, test_y, test_raw_logits, test_bin_logits, uncal_probs, cal_probs, predicted_labels
        ):
            writer.writerow([
                img_id, y_true, f"{logit_0:.5f}", f"{logit_1:.5f}", f"{bin_logit:.5f}",
                f"{p_uncal:.5f}", f"{p_cal:.5f}", p_class,
                args.arm, args.backbone, args.seed
            ])

    # 8. Export Metrics JSON
    metrics_summary = {
        "arm": args.arm,
        "backbone": args.backbone,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_auroc": float(best_val_auc),
        "optimal_temperature": float(optimal_t),
        "test_auroc": float(test_auc),
        "test_brier_calibrated": float(test_brier),
        "training_time_seconds": float(training_time_sec),
        "predictions_file": str(pred_csv_path),
        "checkpoint_file": str(best_model_path),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"Artifacts successfully written to {out_dir}")


if __name__ == "__main__":
    main()
