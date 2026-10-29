# EXPERIMENT_TRACKER.md — Central Experiment Registry: OvaryVision

This document is the persistent registry of all planned, executing, and completed experimental runs in the OvaryVision benchmark.

---

## 1. Registry Format & Rules

Every experimental run must be logged here before execution with status `PLANNED`. Upon completion, the record must be updated with final validation and test metrics, optimal temperature $T^*$, output checkpoint path, and any runtime deviations.

- **Status Values**: `PLANNED`, `RUNNING`, `COMPLETED`, `FAILED`, `ABORTED`
- **Zero Fabrication**: Metrics must only be entered when genuine GPU runs have completed and outputs have been verified against prediction CSVs.

---

## 2. Benchmark Experiment Registry

### Arm A: Raw Scans (Shortcut Baseline) — Completed & Audited

| Experiment ID | Arm | Backbone | Seed | Platform | Status | Val AUROC | Test AUROC | Test ECE (Raw) | Test ECE ($T^*$) | Brier ($T^*$) | $T^*$ | Checkpoint / Artifact Path |
|---|---|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `EXP_A_EFF3_S42` | Arm A | `efficientnet_b3` | 42 | Colab G4 | **COMPLETED** | 0.8868 | 0.9251 | 0.2960 | 0.1885 | 0.0899 | 0.4600 | `outputs/benchmark_runs/colab/arm_a_efficientnet_b3_seed42/` |
| `EXP_A_EFF3_S1337` | Arm A | `efficientnet_b3` | 1337 | Colab G4 | **COMPLETED** | 0.7754 | 0.9116 | 0.2815 | 0.1513 | 0.0697 | 0.4200 | `outputs/benchmark_runs/colab/arm_a_efficientnet_b3_seed1337/` |
| `EXP_A_EFF3_S2026` | Arm A | `efficientnet_b3` | 2026 | Colab G4 | **COMPLETED** | 0.9664 | 0.8426 | 0.2781 | 0.1643 | 0.0853 | 0.4700 | `outputs/benchmark_runs/colab/arm_a_efficientnet_b3_seed2026/` |
| `EXP_A_CONV_S42` | Arm A | `convnext_small` | 42 | Colab G4 | **COMPLETED** | 0.8827 | 0.8367 | 0.5543 | 0.4822 | 0.2668 | 4.9900 | `outputs/benchmark_runs/colab/arm_a_convnext_small_seed42/` |
| `EXP_A_CONV_S1337` | Arm A | `convnext_small` | 1337 | Colab G4 | **COMPLETED** | 0.8408 | 0.9593 | 0.3402 | 0.3925 | 0.1892 | 1.9000 | `outputs/benchmark_runs/colab/arm_a_convnext_small_seed1337/` |
| `EXP_A_CONV_S2026` | Arm A | `convnext_small` | 2026 | Colab G4 | **COMPLETED** | 0.9458 | 0.8449 | 0.0260 | 0.0233 | 0.0334 | 0.9500 | `outputs/benchmark_runs/colab/arm_a_convnext_small_seed2026/` |
| `EXP_A_CONV_S42_KAG`| Arm A | `convnext_small` | 42 | Kaggle T4 | **COMPLETED** | 0.9629 | 0.8797 | 0.0287 | 0.0117 | 0.0231 | 0.7200 | `outputs/benchmark_runs/kaggle/output/arm_a_convnext_small_seed42/`|

---

### Arm B & Arm C: Real-Image Pilot Study (Gate B Approved & Executed) — Completed & Audited

The 140 QC-passed pilot images were transformed under Arms B (Harmonic Inpainting with 2px dilation) and C (Zero-Masking with 2px dilation), guaranteeing $MAE_{ext} = 0.000000$ and $PSNR_{ext} = \infty$. Zero-shot held-out test inference was executed across all 6 baseline checkpoints:

| Experiment ID | Evaluation Subset | Backbone | Seed | $N$ | $\Delta p (B - A)$ [95% CI] | $\Delta p (C - A)$ [95% CI] | $\Delta p (C - B)$ [95% CI] | Direction Agreement | Status | Artifact Output CSV |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `EXP_PILOT_CONV_S42` | Held-Out Test | `convnext_small` | 42 | 15 | +0.0019 [-0.0001, +0.0041] | **+0.0328** [+0.0164, +0.0528] | +0.0309 [+0.0155, +0.0495] | 60.0% | **COMPLETED** | `outputs/benchmark_runs/real_pilot_eval/convnext_small_seed42_pilot_predictions.csv` |
| `EXP_PILOT_CONV_S1337`| Held-Out Test | `convnext_small` | 1337 | 24 | -0.0008 [-0.0019, +0.0001] | **+0.0052** [+0.0017, +0.0093] | +0.0060 [+0.0028, +0.0099] | 75.0% | **COMPLETED** | `outputs/benchmark_runs/real_pilot_eval/convnext_small_seed1337_pilot_predictions.csv` |
| `EXP_PILOT_CONV_S2026`| Held-Out Test | `convnext_small` | 2026 | 22 | +0.0000 [-0.0009, +0.0011] | **+0.0070** [+0.0019, +0.0158] | +0.0070 [+0.0020, +0.0151] | 77.3% | **COMPLETED** | `outputs/benchmark_runs/real_pilot_eval/convnext_small_seed2026_pilot_predictions.csv` |
| `EXP_PILOT_EFF3_S42` | Held-Out Test | `efficientnet_b3` | 42 | 15 | +0.0016 [-0.0027, +0.0058] | **+0.0085** [+0.0043, +0.0125] | +0.0069 [+0.0003, +0.0144] | 73.3% | **COMPLETED** | `outputs/benchmark_runs/real_pilot_eval/efficientnet_b3_seed42_pilot_predictions.csv` |
| `EXP_PILOT_EFF3_S1337`| Held-Out Test | `efficientnet_b3` | 1337 | 24 | -0.0021 [-0.0037, -0.0005] | **+0.0035** [+0.0003, +0.0072] | +0.0056 [+0.0020, +0.0099] | 62.5% | **COMPLETED** | `outputs/benchmark_runs/real_pilot_eval/efficientnet_b3_seed1337_pilot_predictions.csv` |
| `EXP_PILOT_EFF3_S2026`| Held-Out Test | `efficientnet_b3` | 2026 | 22 | -0.0005 [-0.0014, +0.0003] | **+0.0017** [-0.0013, +0.0046] | +0.0022 [-0.0010, +0.0052] | 54.5% | **COMPLETED** | `outputs/benchmark_runs/real_pilot_eval/efficientnet_b3_seed2026_pilot_predictions.csv` |

---

### Full-Cohort Arm B & C Training Benchmark (Scaled Dataset) — Planned

| Experiment ID | Arm | Backbone | Seed | Status | Notes |
|---|---|---|:---:|:---:|---|
| `EXP_B_EFF3_FULL` | Arm B | `efficientnet_b3` | Multi | PLANNED | Pending complete MMOTU annotation scaling (Malignant cohort prioritized) |
| `EXP_B_CONV_FULL` | Arm B | `convnext_small` | Multi | PLANNED | Pending complete MMOTU annotation scaling (Malignant cohort prioritized) |
| `EXP_C_EFF3_FULL` | Arm C | `efficientnet_b3` | Multi | PLANNED | Pending complete MMOTU annotation scaling (Malignant cohort prioritized) |
| `EXP_C_CONV_FULL` | Arm C | `convnext_small` | Multi | PLANNED | Pending complete MMOTU annotation scaling (Malignant cohort prioritized) |

---

## 3. Logged Deviations & Retries

1. **Backbone Harmonization**: Aligned with Section 15.1 of the approved abstract (`OvaryVision Second Deep Research Investigation and....docx`), transitioning from smoke-test models (`convnext_tiny`, `efficientnet_b0`) to approved benchmark architectures (`convnext_small`, `efficientnet_b3`).
2. **Kaggle Kernel Version 1 vs Version 2**: Version 1 encountered `FileNotFoundError` due to directory traversal entering `OTU_3d` instead of `OTU_2d`. Corrected in Version 2 to explicitly target `OTU_2d`, completing in 1,250s with Test AUROC 0.8797.
3. **Partition Typology**: Confirmed that seeds 42, 1337, and 2026 represent repeated group-stratified Monte Carlo splits (~14% test overlap), not exhaustive $K$-fold cross-validation.
4. **Calibration Protocol**: Abstract does not pre-specify temperature scaling; uncalibrated ECE and Brier are logged as primary metrics, while temperature-scaled metrics are logged as exploratory post-hoc analysis.
