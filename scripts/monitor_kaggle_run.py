#!/usr/bin/env python3
"""Autonomous monitor for Kaggle GPU benchmark runs.

Polls Kaggle kernel status, automatically downloads outputs or error logs upon completion,
and exits with code 0 on success or code 1 on failure.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

KAGGLE_BIN = shutil.which("kaggle") or "/home/ubuntu/.local/bin/kaggle"

def main():
    kernel_slug = sys.argv[1] if len(sys.argv) > 1 else "tanishaqbansal/ovaryvision-benchmark"
    out_dir = Path("outputs/benchmark_runs/kaggle")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting autonomous monitoring for Kaggle kernel: {kernel_slug}")
    start_time = time.time()
    last_status = ""

    while True:
        res = subprocess.run(
            [KAGGLE_BIN, "kernels", "status", kernel_slug],
            capture_output=True,
            text=True,
        )
        output = res.stdout.strip()
        
        # Extract status string
        if "KernelWorkerStatus.COMPLETE" in output:
            elapsed = time.time() - start_time
            print(f"\n[✓] Kaggle Kernel COMPLETED in {elapsed:.1f}s!")
            # Download outputs
            subprocess.run(
                [KAGGLE_BIN, "kernels", "output", kernel_slug, "-p", str(out_dir)],
                capture_output=True,
            )
            # Check metrics
            metrics_file = list(out_dir.rglob("metrics.json"))
            if metrics_file:
                with open(metrics_file[0], "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                print("=======================================================")
                print("            KAGGLE GPU BENCHMARK METRICS               ")
                print("=======================================================")
                print(f"  Model Architecture  : {metrics.get('backbone')}")
                print(f"  Random Seed         : {metrics.get('seed')}")
                print(f"  Best Epoch          : {metrics.get('best_epoch')}")
                print(f"  Val AUROC           : {metrics.get('best_val_auroc'):.4f}")
                print(f"  Optimal Temperature : {metrics.get('optimal_temperature'):.4f}")
                print(f"  Test AUROC          : {metrics.get('test_auroc'):.4f}")
                print(f"  Test ECE (Uncal)    : {metrics.get('test_ece_uncalibrated'):.4f}")
                print(f"  Test ECE (Cal)      : {metrics.get('test_ece_calibrated'):.4f}")
                print(f"  Test Brier (Cal)    : {metrics.get('test_brier_calibrated'):.4f}")
                print(f"  Training Time (sec) : {metrics.get('training_time_seconds'):.1f}")
                print("=======================================================")
            else:
                print(f"Outputs downloaded to: {out_dir}")
            sys.exit(0)

        elif "KernelWorkerStatus.ERROR" in output:
            elapsed = time.time() - start_time
            print(f"\n[✗] Kaggle Kernel FAILED after {elapsed:.1f}s.")
            subprocess.run(
                [KAGGLE_BIN, "kernels", "output", kernel_slug, "-p", str(out_dir)],
                capture_output=True,
            )
            log_files = list(out_dir.glob("*.log"))
            if log_files:
                print("\n--- Recent Error Logs ---")
                with open(log_files[0], "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines[-25:]:
                        print(line, end="")
            sys.exit(1)

        elif "KernelWorkerStatus.RUNNING" in output or "KernelWorkerStatus.QUEUED" in output:
            if output != last_status:
                print(f"Status update: {output} (elapsed: {time.time() - start_time:.0f}s)")
                last_status = output
            time.sleep(15)
        else:
            print(f"Unknown status output: {output}")
            time.sleep(15)

if __name__ == "__main__":
    main()

