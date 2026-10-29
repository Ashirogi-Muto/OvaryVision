#!/usr/bin/env python3
from __future__ import annotations
"""OvaryVision Kaggle GPU Benchmark Runner.

Executes ConvNeXt-Small and EfficientNet-B3 model training on the MMOTU dataset
under Arm A (Raw baseline) with leak-resistant group-stratified splits.
"""

import argparse
import base64
import csv
import gzip
import io
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
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
    class Dataset:
        pass
    class _DummyNN:
        class Module:
            pass
    nn = _DummyNN

# Embedded group-stratified split manifests (gzipped base64)
EMBEDDED_SPLITS = {
    42: """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""",
    1337: """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""",
    2026: """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""",
}

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def unpack_split_manifest(seed: int) -> List[Dict[str, str]]:
    if seed not in EMBEDDED_SPLITS:
        raise ValueError(f"No embedded split manifest for seed {seed}")
    compressed = base64.b64decode(EMBEDDED_SPLITS[seed])
    raw_csv = gzip.decompress(compressed).decode("utf-8")
    reader = csv.DictReader(io.StringIO(raw_csv))
    return list(reader)

def locate_dataset_dir() -> Path:
    # 1. Search specifically for OTU_2d/images directory
    for p in Path("/kaggle/input").rglob("images"):
        if "otu_2d" in str(p).lower() and p.is_dir():
            jpg_count = len(list(p.glob("*.JPG")))
            if jpg_count > 0:
                print(f"Located OTU_2d images directory directly ({jpg_count} images): {p}")
                return p

    # 2. Search for any directory containing 1.JPG inside an otu_2d path
    for p in Path("/kaggle/input").rglob("1.JPG"):
        if "otu_2d" in str(p).lower():
            parent = p.parent
            print(f"Located OTU_2d parent containing 1.JPG: {parent}")
            return parent

    # 3. Local fallback for development/testing
    local_path = Path("data/raw/mmotu/images")
    if local_path.is_dir() and len(list(local_path.glob("*.JPG"))) > 0:
        print(f"Located local images directory: {local_path}")
        return local_path

    raise FileNotFoundError("Could not locate valid OTU_2d images directory with JPG files.")

class UltrasoundDataset(Dataset):
    def __init__(self, records: List[Dict[str, str]], data_dir: Path, transform: Optional[transforms.Compose] = None):
        self.records = records
        self.data_dir = data_dir
        self.transform = transform
        # Pre-index images by stem for fast lookup
        self.image_index: Dict[str, Path] = {}
        for p in self.data_dir.glob("*.JPG"):
            self.image_index[p.stem] = p

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        image_id = str(rec["image_id"])
        if image_id in self.image_index:
            img_path = self.image_index[image_id]
        else:
            candidates = list(self.data_dir.glob(f"**/{image_id}.*"))
            if not candidates:
                raise FileNotFoundError(f"Image {image_id} not found in {self.data_dir}")
            img_path = candidates[0]
            self.image_index[image_id] = img_path

        img = Image.open(img_path).convert("RGB")
        label = int(rec["diagnostic_label"])
        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)
        return tensor, label, image_id

class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        return (((1.0 - pt) ** self.gamma) * ce_loss).mean()

def build_model(backbone: str, num_classes: int = 2) -> nn.Module:
    b = backbone.lower().strip()
    if b in ("convnext_small", "convnext"):
        try:
            weights = models.ConvNeXt_Small_Weights.DEFAULT
            m = models.convnext_small(weights=weights)
        except AttributeError:
            m = models.convnext_small(pretrained=True)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_classes)
    elif b in ("efficientnet_b3", "efficientnet"):
        try:
            weights = models.EfficientNet_B3_Weights.DEFAULT
            m = models.efficientnet_b3(weights=weights)
        except AttributeError:
            m = models.efficientnet_b3(pretrained=True)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
    elif b == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_classes)
    elif b == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return m

def compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    unique, idx, counts = np.unique(scores[order], return_index=True, return_counts=True)
    for i, c in zip(idx, counts):
        if c > 1:
            ranks[order[i : i + c]] = np.mean(np.arange(i + 1, i + c + 1))
    u_stat = np.sum(ranks[labels == 1]) - (len(pos) * (len(pos) + 1)) / 2.0
    return float(u_stat / (len(pos) * len(neg)))

def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    z = val_logits
    y = val_labels
    def nll(t: float) -> float:
        scaled = z / max(t, 1e-4)
        max_val = np.maximum(scaled, 0.0)
        return float(np.mean(max_val - scaled * y + np.log1p(np.exp(-np.abs(scaled)))))

    best_t, best_loss = 1.0, float("inf")
    for t_cand in np.arange(0.05, 5.0, 0.01):
        loss_val = nll(t_cand)
        if loss_val < best_loss:
            best_loss = loss_val
            best_t = float(t_cand)
    return best_t

def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    n = len(probs)
    total_ece = 0.0
    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs >= low) & (probs <= high) if i == num_bins - 1 else (probs >= low) & (probs < high)
        count = int(np.sum(in_bin))
        if count > 0:
            mean_conf = float(np.mean(probs[in_bin]))
            obs_freq = float(np.mean(labels[in_bin]))
            total_ece += (count / n) * abs(obs_freq - mean_conf)
    return float(total_ece)

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for imgs, lbls, ids in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            all_logits.append(outputs.cpu().numpy())
            all_labels.extend(lbls.numpy().tolist())
            all_ids.extend(ids)
    logits_arr = np.concatenate(all_logits, axis=0)
    labels_arr = np.array(all_labels, dtype=int)
    bin_logits = logits_arr[:, 1] - logits_arr[:, 0]
    return bin_logits, logits_arr, labels_arr, all_ids

def main():
    parser = argparse.ArgumentParser(description="OvaryVision GPU Training")
    parser.add_argument("--backbone", default="convnext_small", choices=["convnext_small", "efficientnet_b3", "convnext_tiny", "efficientnet_b0"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    out_dir = Path(args.output_dir) / f"arm_a_{args.backbone}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = locate_dataset_dir()
    manifest_records = unpack_split_manifest(args.seed)
    print(f"Unpacked {len(manifest_records)} records for seed {args.seed}")

    train_recs = [r for r in manifest_records if r["split_role"].lower() == "train"]
    val_recs = [r for r in manifest_records if r["split_role"].lower() == "val"]
    test_recs = [r for r in manifest_records if r["split_role"].lower() in ("test_internal", "test")]
    print(f"Cohorts: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")

    # Inverse class weights
    y_train = [int(r["diagnostic_label"]) for r in train_recs]
    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    alpha_0 = len(y_train) / (2.0 * max(n_neg, 1))
    alpha_1 = len(y_train) / (2.0 * max(n_pos, 1))
    alpha_tensor = torch.tensor([alpha_0, alpha_1], dtype=torch.float32).to(device)

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(UltrasoundDataset(train_recs, data_dir, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(UltrasoundDataset(val_recs, data_dir, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(UltrasoundDataset(test_recs, data_dir, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.backbone).to(device)
    criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auc, best_epoch, patience_cnt = -1.0, 0, 0
    best_pth = out_dir / "best_model.pth"

    print(f"Training {args.backbone} on Arm A (Raw baseline) across {args.epochs} epochs...")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        for imgs, lbls, _ in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * len(lbls)
        train_loss = tot_loss / len(train_recs)
        scheduler.step()

        val_bin_logits, _, val_y, _ = evaluate(model, val_loader, device)
        val_probs = 1.0 / (1.0 + np.exp(-val_bin_logits))
        val_auc = compute_roc_auc(val_probs, val_y)
        print(f"Epoch {ep:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val AUROC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = ep
            patience_cnt = 0
            torch.save(model.state_dict(), best_pth)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping triggered at epoch {ep}. Best epoch {best_epoch} with Val AUROC {best_val_auc:.4f}")
                break

    train_time = time.time() - t0
    print(f"Training finished in {train_time:.1f}s. Evaluating best model...")
    model.load_state_dict(torch.load(best_pth, map_location=device))

    # Validation temperature scaling
    val_bin_logits, _, val_y, _ = evaluate(model, val_loader, device)
    optimal_t = fit_temperature(val_bin_logits, val_y)
    print(f"Fitted Validation Temperature Scaling: T* = {optimal_t:.4f}")

    # Test set evaluation
    test_bin_logits, test_raw_logits, test_y, test_ids = evaluate(model, test_loader, device)
    uncal_probs = 1.0 / (1.0 + np.exp(-test_bin_logits))
    cal_probs = 1.0 / (1.0 + np.exp(-(test_bin_logits / optimal_t)))
    test_auc = compute_roc_auc(cal_probs, test_y)
    uncal_ece = compute_ece(uncal_probs, test_y)
    cal_ece = compute_ece(cal_probs, test_y)
    uncal_brier = float(np.mean((uncal_probs - test_y) ** 2))
    cal_brier = float(np.mean((cal_probs - test_y) ** 2))
    preds = (cal_probs >= 0.5).astype(int)

    print(f"Test Set Evaluation:")
    print(f"  AUROC             : {test_auc:.4f}")
    print(f"  Uncalibrated ECE  : {uncal_ece:.4f}")
    print(f"  Calibrated ECE    : {cal_ece:.4f}")
    print(f"  Uncalibrated Brier: {uncal_brier:.4f}")
    print(f"  Calibrated Brier  : {cal_brier:.4f}")

    # Write predictions CSV
    pred_path = out_dir / "predictions.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id", "true_label", "raw_logit_0", "raw_logit_1", "binary_logit",
            "uncalibrated_prob", "calibrated_prob", "predicted_class",
            "arm", "backbone", "seed"
        ])
        for i_id, y_t, (l0, l1), b_l, p_u, p_c, p_cls in zip(
            test_ids, test_y, test_raw_logits, test_bin_logits, uncal_probs, cal_probs, preds
        ):
            writer.writerow([i_id, y_t, f"{l0:.5f}", f"{l1:.5f}", f"{b_l:.5f}", f"{p_u:.5f}", f"{p_c:.5f}", p_cls, "arm_a_raw", args.backbone, args.seed])

    # Write metrics JSON
    summary = {
        "arm": "arm_a_raw",
        "backbone": args.backbone,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_auroc": float(best_val_auc),
        "optimal_temperature": float(optimal_t),
        "test_auroc": float(test_auc),
        "test_ece_uncalibrated": float(uncal_ece),
        "test_ece_calibrated": float(cal_ece),
        "test_brier_uncalibrated": float(uncal_brier),
        "test_brier_calibrated": float(cal_brier),
        "training_time_seconds": float(train_time),
        "predictions_file": str(pred_path),
        "checkpoint_file": str(best_pth),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Benchmark completed. Artifacts saved in {out_dir}")

if __name__ == "__main__":
    main()
