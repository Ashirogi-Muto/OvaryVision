"""G4 Compute Benchmark for OvaryVision Segmentation.

Runs on NVIDIA RTX PRO 6000 Blackwell Server Edition.
Measures:
- Environment & compatibility (PyTorch, CUDA, cuDNN, Device, Compute Capability)
- Training step latency & throughput (imgs/sec)
- Validation throughput (imgs/sec)
- Peak GPU VRAM (allocated and reserved)
- GPU utilization & execution breakdown (pure GPU kernel time vs wall-clock with data loading)
- Numerical stability (NaN/Inf checks, AMP GradScaler stability)
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

# Ensure dependencies
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Installing segmentation-models-pytorch...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"])
    import segmentation_models_pytorch as smp

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize(self.img_size, resample=Image.BILINEAR)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1)
        img_t = (img_t - self.mean) / self.std

        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = mask.resize(self.img_size, resample=Image.NEAREST)
        mask_arr = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)

        return img_t, mask_t


def run_g4_benchmark():
    print("=" * 60)
    print("STARTING G4 COMPUTE BENCHMARK FOR OVARYVISION")
    print("=" * 60)

    # 1. Environment and Hardware Verification
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    device_count = torch.cuda.device_count() if cuda_available else 0

    major, minor = 0, 0
    total_vram_gb = 0.0
    if cuda_available:
        prop = torch.cuda.get_device_properties(0)
        major, minor = prop.major, prop.minor
        total_vram_gb = prop.total_memory / (1024 ** 3)

    env_info = {
        "cuda_available": cuda_available,
        "device_name": gpu_name,
        "device_count": device_count,
        "compute_capability": f"{major}.{minor}",
        "total_vram_gb": round(total_vram_gb, 2),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A",
        "cpu_count": os.cpu_count(),
    }
    print("Environment Info:")
    print(json.dumps(env_info, indent=2))

    if not cuda_available:
        print("ERROR: CUDA is not available. Aborting benchmark.")
        return

    # 2. Extract Data
    zip_path = Path("/content/smoke_data.zip")
    data_dir = Path("/content/data")
    if not data_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)

    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    img_files = sorted(list(img_dir.glob("*.JPG")))
    matched_imgs, matched_masks = [], []
    for ip in img_files:
        tid = ip.stem
        mp = mask_dir / f"{tid}_binary.PNG"
        if mp.exists():
            matched_imgs.append(ip)
            matched_masks.append(mp)

    print(f"Loaded {len(matched_imgs)} paired images and masks.")
    assert len(matched_imgs) >= 80

    train_imgs, val_imgs = matched_imgs[:64], matched_imgs[64:80]
    train_masks, val_masks = matched_masks[:64], matched_masks[64:80]

    train_ds = UltrasoundSegDataset(train_imgs, train_masks, img_size=(512, 512))
    val_ds = UltrasoundSegDataset(val_imgs, val_masks, img_size=(512, 512))

    batch_size = 8
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model, Criterion, Optimizer, Scaler
    torch.manual_seed(42)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    criterion = ComboLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # 4. Warm-up Step
    print("\nExecuting warm-up step to initialize CUDA kernels and autograd engine...")
    model.train()
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=True):
            out = model(imgs)
            loss = criterion(out, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        break
    torch.cuda.synchronize()
    print("Warm-up complete.")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # 5. Measure Pre-Training Validation Throughput
    model.eval()
    val_step_times = []
    val_gpu_times = []
    val_losses = []
    val_samples_processed = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for imgs, masks in val_loader:
            t_wall_0 = time.perf_counter()
            start_event.record()
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast("cuda", enabled=True):
                preds = model(imgs)
                loss = criterion(preds, masks)
            end_event.record()
            torch.cuda.synchronize()
            t_wall = time.perf_counter() - t_wall_0
            gpu_time_ms = start_event.elapsed_time(end_event)

            val_step_times.append(t_wall)
            val_gpu_times.append(gpu_time_ms / 1000.0)
            val_losses.append(loss.item())
            val_samples_processed += imgs.size(0)

    avg_val_step_wall = float(np.mean(val_step_times))
    avg_val_step_gpu = float(np.mean(val_gpu_times))
    val_throughput_imgs_per_sec = val_samples_processed / sum(val_step_times)
    initial_val_loss = float(np.mean(val_losses))

    print(f"\nInitial Validation Loss: {initial_val_loss:.4f}")
    print(f"Validation Step Latency (Wall-clock): {avg_val_step_wall*1000:.2f} ms/step")
    print(f"Validation Step Latency (GPU Kernel): {avg_val_step_gpu*1000:.2f} ms/step")
    print(f"Validation Throughput: {val_throughput_imgs_per_sec:.2f} images/sec")

    # 6. Training Benchmark Across 2 Full Epochs (16 training batches)
    epoch_results = []
    all_train_step_wall_times = []
    all_train_step_gpu_times = []
    has_nan_inf = False

    t_bench_start = time.perf_counter()

    for epoch in range(1, 3):
        model.train()
        epoch_train_losses = []
        epoch_step_wall_times = []
        epoch_step_gpu_times = []
        t_epoch_start = time.perf_counter()

        for imgs, masks in train_loader:
            t_step_wall_0 = time.perf_counter()
            start_event.record()

            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=True):
                preds = model(imgs)
                loss = criterion(preds, masks)

            # Check for NaN / Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                has_nan_inf = True
                print("WARNING: NaN or Inf detected in loss!")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            end_event.record()
            torch.cuda.synchronize()

            t_step_wall = time.perf_counter() - t_step_wall_0
            gpu_step_ms = start_event.elapsed_time(end_event)

            # Check weights for NaN / Inf
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_inf = True

            epoch_step_wall_times.append(t_step_wall)
            epoch_step_gpu_times.append(gpu_step_ms / 1000.0)
            epoch_train_losses.append(loss.item())

        epoch_duration = time.perf_counter() - t_epoch_start
        all_train_step_wall_times.extend(epoch_step_wall_times)
        all_train_step_gpu_times.extend(epoch_step_gpu_times)

        # Validation at epoch end
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.amp.autocast("cuda", enabled=True):
                    preds = model(imgs)
                    loss = criterion(preds, masks)
                epoch_val_losses.append(loss.item())

        mean_train_loss = float(np.mean(epoch_train_losses))
        mean_val_loss = float(np.mean(epoch_val_losses))
        current_scale = scaler.get_scale()

        print(
            f"Epoch {epoch}/2 | Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} | "
            f"Epoch Time: {epoch_duration:.2f}s | Avg Step Latency: {np.mean(epoch_step_wall_times)*1000:.1f}ms | "
            f"Scaler: {current_scale}"
        )

        epoch_results.append({
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "val_loss": round(mean_val_loss, 4),
            "epoch_duration_seconds": round(epoch_duration, 3),
            "step_wall_latency_ms": round(float(np.mean(epoch_step_wall_times)) * 1000, 2),
            "step_gpu_latency_ms": round(float(np.mean(epoch_step_gpu_times)) * 1000, 2),
            "scaler_scale": float(current_scale),
        })

    total_bench_time = time.perf_counter() - t_bench_start

    # 7. Memory & Hardware Metrics
    peak_vram_alloc_bytes = torch.cuda.max_memory_allocated()
    peak_vram_alloc_mb = peak_vram_alloc_bytes / (1024 * 1024)
    peak_vram_alloc_gb = peak_vram_alloc_bytes / (1024 ** 3)

    peak_vram_res_bytes = torch.cuda.max_memory_reserved()
    peak_vram_res_mb = peak_vram_res_bytes / (1024 * 1024)
    peak_vram_res_gb = peak_vram_res_bytes / (1024 ** 3)

    avg_train_step_wall = float(np.mean(all_train_step_wall_times))
    avg_train_step_gpu = float(np.mean(all_train_step_gpu_times))
    train_throughput_imgs_per_sec = batch_size / avg_train_step_wall

    # GPU compute utilization ratio (GPU kernel active time / wall-clock time)
    gpu_utilization_pct = (avg_train_step_gpu / avg_train_step_wall) * 100.0

    # 8. Runtime Extrapolations
    # 850 training scans = 107 batches of 8 (with drop_last=False) or 106 batches
    # 150 validation scans = 19 batches of 8
    n_train_batches = int(np.ceil(850 / batch_size))
    n_val_batches = int(np.ceil(150 / batch_size))

    single_epoch_train_sec = n_train_batches * avg_train_step_wall
    single_epoch_val_sec = n_val_batches * avg_val_step_wall
    single_epoch_total_sec = single_epoch_train_sec + single_epoch_val_sec

    # 40 epochs per run
    single_run_sec = single_epoch_total_sec * 40
    single_run_min = single_run_sec / 60.0

    # 3-arm Seed 42 pilot = 3 runs
    pilot_3_arms_min = single_run_min * 3
    pilot_3_arms_hr = pilot_3_arms_min / 60.0

    # 9 runs = 3 arms x 3 seeds
    all_9_runs_min = single_run_min * 9
    all_9_runs_hr = all_9_runs_min / 60.0

    benchmark_summary = {
        "status": "SUCCESS",
        "hardware": {
            "device_name": gpu_name,
            "architecture": "Blackwell Server Edition",
            "compute_capability": f"{major}.{minor}",
            "total_vram_gb": round(total_vram_gb, 2),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A",
        },
        "configuration": {
            "resolution": "512x512",
            "batch_size": batch_size,
            "precision": "AMP_FP16",
            "model": "SMP U-Net (ResNet-34 ImageNet pretrained)",
            "optimizer": "AdamW (lr=1e-4, wd=1e-2)",
            "loss": "ComboLoss (0.5 BCE + 0.5 Dice)",
        },
        "measurements": {
            "training_step_wall_latency_ms": round(avg_train_step_wall * 1000, 2),
            "training_step_gpu_latency_ms": round(avg_train_step_gpu * 1000, 2),
            "training_throughput_images_per_sec": round(train_throughput_imgs_per_sec, 2),
            "validation_step_wall_latency_ms": round(avg_val_step_wall * 1000, 2),
            "validation_step_gpu_latency_ms": round(avg_val_step_gpu * 1000, 2),
            "validation_throughput_images_per_sec": round(val_throughput_imgs_per_sec, 2),
            "peak_vram_allocated_mb": round(peak_vram_alloc_mb, 2),
            "peak_vram_allocated_gb": round(peak_vram_alloc_gb, 3),
            "peak_vram_reserved_mb": round(peak_vram_res_mb, 2),
            "peak_vram_reserved_gb": round(peak_vram_res_gb, 3),
            "gpu_kernel_utilization_pct": round(gpu_utilization_pct, 1),
            "numerical_stability": "STABLE_NO_NANS_NO_INFS" if not has_nan_inf else "INSTABILITY_DETECTED",
            "gradscaler_scale": float(scaler.get_scale()),
            "initial_val_loss": round(initial_val_loss, 4),
            "epochs": epoch_results,
        },
        "runtime_extrapolations": {
            "batches_per_epoch": {"train": n_train_batches, "val": n_val_batches},
            "single_epoch_seconds": round(single_epoch_total_sec, 2),
            "single_run_40_epochs_minutes": round(single_run_min, 2),
            "three_arm_pilot_minutes": round(pilot_3_arms_min, 2),
            "three_arm_pilot_hours": round(pilot_3_arms_hr, 3),
            "all_nine_runs_minutes": round(all_9_runs_min, 2),
            "all_nine_runs_hours": round(all_9_runs_hr, 3),
        },
    }

    print("\n" + "=" * 60)
    print("FINAL G4 BENCHMARK SUMMARY (JSON)")
    print("=" * 60)
    print(json.dumps(benchmark_summary, indent=2))

    with open("/content/g4_benchmark_summary.json", "w") as f:
        json.dump(benchmark_summary, f, indent=2)


if __name__ == "__main__":
    run_g4_benchmark()

