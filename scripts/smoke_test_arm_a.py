"""Two-epoch smoke test for Arm A segmentation on GPU.

Verifies:
- Architecture: SMP U-Net with ResNet-34 encoder
- Input Resolution: 512 x 512
- Batch Size: 8
- Data Pipeline: Real Arm A ultrasound scans & real tumor masks
- Loss: Combo BCE + Dice Loss
- Metrics: Epoch runtime, peak VRAM, and validation loss across 2 epochs
"""

import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp


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


class UltrasoundSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=(512, 512)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize(self.img_size, resample=Image.BILINEAR)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1) # (3, H, W)
        img_t = (img_t - self.mean) / self.std

        # Load mask
        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = mask.resize(self.img_size, resample=Image.NEAREST)
        mask_arr = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0) # (1, H, W)

        return img_t, mask_t


def run_smoke_test():
    # 1. Unpack data
    zip_path = Path("/content/smoke_data.zip")
    data_dir = Path("/content/data")
    if not data_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)

    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"

    img_files = sorted(list(img_dir.glob("*.JPG")))
    matched_imgs = []
    matched_masks = []
    for ip in img_files:
        tid = ip.stem
        mp = mask_dir / f"{tid}_binary.PNG"
        if mp.exists():
            matched_imgs.append(ip)
            matched_masks.append(mp)

    print(f"Total matched pairs loaded: {len(matched_imgs)}")
    assert len(matched_imgs) >= 80, f"Expected at least 80 scans, found {len(matched_imgs)}"

    # 64 train, 16 val
    train_imgs, val_imgs = matched_imgs[:64], matched_imgs[64:80]
    train_masks, val_masks = matched_masks[:64], matched_masks[64:80]

    train_ds = UltrasoundSegDataset(train_imgs, train_masks, img_size=(512, 512))
    val_ds = UltrasoundSegDataset(val_imgs, val_masks, img_size=(512, 512))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device} ({gpu_name})")

    # 2. Instantiate Model
    torch.manual_seed(42)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    criterion = ComboLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Initial validation loss
    model.eval()
    init_val_losses = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            loss = criterion(out, masks)
            init_val_losses.append(loss.item())
    initial_val_loss = float(np.mean(init_val_losses))
    print(f"Initial Validation Loss (Pre-Training): {initial_val_loss:.4f}")

    # 3. Two-epoch training
    t_start = time.time()
    epoch_logs = []

    for epoch in range(1, 3):
        t0 = time.time()
        model.train()
        train_losses = []
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_time = time.time() - t0
        mean_train_loss = float(np.mean(train_losses))

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = criterion(preds, masks)
                val_losses.append(loss.item())
        mean_val_loss = float(np.mean(val_losses))

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "val_loss": round(mean_val_loss, 4),
            "train_time_seconds": round(train_time, 2),
            "batch_count": len(train_losses),
        })
        print(f"Epoch {epoch}/2 | Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} | Time: {train_time:.2f}s")

    total_duration = time.time() - t_start

    # Peak VRAM
    peak_vram_mb = 0.0
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_mb = peak_vram_bytes / (1024 * 1024)
        peak_vram_gb = peak_vram_bytes / (1024 * 1024 * 1024)

    results = {
        "status": "SUCCESS",
        "device": str(device),
        "gpu_name": gpu_name,
        "encoder": "resnet34",
        "input_resolution": "512x512",
        "batch_size": 8,
        "train_samples": len(train_imgs),
        "val_samples": len(val_imgs),
        "initial_val_loss": round(initial_val_loss, 4),
        "epochs": epoch_logs,
        "total_runtime_seconds": round(total_duration, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
    }

    print("\n=== SMOKE TEST RESULTS ===")
    print(json.dumps(results, indent=2))

    with open("/content/smoke_test_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_smoke_test()

