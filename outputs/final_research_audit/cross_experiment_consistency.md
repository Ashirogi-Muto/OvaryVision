# Cross-Experiment Consistency Audit: Classification vs Segmentation

**Date**: 2026-11-01  
**Auditor**: Antigravity (Advanced Agentic Coding / DeepMind)  
**Scope**: Comparison of experimental designs, dataset manifests, artifact definitions, cohort boundaries, and preprocessing pipelines between:
1. **The 18-Run Classification Benchmark** (`outputs/benchmark_runs/full_benchmark/`)
2. **The 9-Run Segmentation Benchmark** (`outputs/benchmark_runs/segmentation_benchmark/`)

---

## 1. Executive Consistency Matrix

| Dimension | Classification Benchmark (18 Runs) | Segmentation Benchmark (9 Runs) | Consistency Verdict | Discrepancy & Methodological Justification |
|---|---|---|:---:|---|
| **Underlying Cohort** | 1,469 2D MMOTU Scans | 1,469 2D MMOTU Scans | **IDENTICAL** | Both draw from the exact same 1,469 authentic 2D images. |
| **Artifact Mask Protocol** | Automated Caliper Detector ($V \ge 220$, Sat $\le 30$, CC Area $\le 3500$, Dil $r=2$) | Automated Caliper Detector ($V \ge 220$, Sat $\le 30$, CC Area $\le 3500$, Dil $r=2$) | **IDENTICAL** | Exactly identical binary artifact masks used for both tasks. |
| **Preprocessing Arms** | Arm A (Raw), Arm B (Telea PDE Inpaint), Arm C (Zero-Masking) | Arm A (Raw), Arm B (Telea/Navier-Stokes), Arm C (Zero-Masking) | **CONSISTENT** | Both enforce 2px dilation and strict exterior conservation ($I_B = I_A$ outside mask). |
| **Dataset Manifests** | `data/manifests/mmotu_split_seed*.csv` | `data/manifests/mmotu_2d_segmentation_split.csv` | **DIFFERENT** | **Documented Difference 1**: Separate manifests used due to distinct benchmarking traditions. |
| **Partitioning Boundaries** | 70% Train (1,029) / 15% Val (220) / 15% Test (220) | Official MMOTU: 1,000 Train (`train.txt`) / 469 Test (`val.txt`); Train split 850 / 150 | **DIFFERENT** | **Documented Difference 2**: Segmentation adheres to official MMOTU benchmark benchmark partitions. |
| **Task Target Schema** | Binary Label $Y \in \{0, 1\}$ (Benign vs Malignant) | Pixel Mask $Y \in \{0, 1\}^{H \times W}$ (Tumour Mass vs Background) | **DIFFERENT** | **Documented Difference 3**: Classification tests malignancy risk; segmentation tests spatial delineation. |
| **Input Resolution** | $224 \times 224$ pixels | $512 \times 512$ pixels | **DIFFERENT** | **Documented Difference 4**: Standard resolution for ImageNet CNN backbones vs fine spatial lesion masks. |
| **Model Architectures** | ConvNeXt-Small & EfficientNet-B3 | SMP U-Net with ResNet-34 Encoder | **DIFFERENT** | **Documented Difference 5**: Image-level classifiers vs spatial semantic segmentation encoder-decoder. |

---

## 2. Detailed Discrepancy Documentation & Analysis

### Discrepancy 1: Split Boundaries and Test Cohorts
- **Classification Benchmark**:
  - Partitioned the 1,469 scans via custom group-stratified pseudo-patient dHash splits:
    - Train: 1,029 scans (37 malignant, 992 benign)
    - Val: 220 scans (8 malignant, 212 benign)
    - Test: 220 scans (8 malignant, 212 benign)
  - Generated separately for seeds 42, 1337, 2026.
- **Segmentation Benchmark**:
  - Adopted the official MMOTU publication split defined by Zhao et al. (2022) in `train.txt` (1,000 scans) and `val.txt` (469 scans):
    - Official `val.txt` was frozen as `held_out_test` ($N=469$: 382 tumour scans + 87 normal ovary scans).
    - Official `train.txt` was partitioned into 850 `internal_train` and 150 `internal_val` using whole dHash pseudo-patient clusters.
  - Fixed across all three seeds (42, 1337, 2026).
- **Scientific Impact**:
  - Direct comparison of test metrics between classification and segmentation cannot be performed at the individual scan level across the two benchmarks because the test cohorts comprise different subsets of the 1,469 scans.
  - However, within each benchmark, test sets are 100% frozen, leak-free, and identical across Arms A, B, and C.

### Discrepancy 2: Normal Ovary Handling
- **Classification Benchmark**:
  - Normal Ovary scans ($N=267$ across the full cohort; $N=38$ in test) are labeled as Benign ($Y=0$). The model outputs a continuous scalar probability of malignancy $p \in [0, 1]$.
- **Segmentation Benchmark**:
  - In `val.txt`, there are exactly 87 Normal Ovary scans (`class_id = 5`).
  - Normal ovaries have an anatomical parenchyma mask ($M_{\text{ovary}}$) but zero tumour ground truth ($Y_{\text{tumour}} = 0$).
  - Evaluating Dice on an all-zero ground truth mask produces an undefined $0/0$ division. Therefore:
    - Tumour segmentation metrics (DSC, IoU, Boundary IoU, LPR) are evaluated strictly on the $N=382$ confirmed tumour scans.
    - Normal ovaries ($N=87$) are evaluated via Clean Rejection Rate ($\mathbb{I}(\hat{Y} = 0)$), Intra-Ovarian FPR ($\frac{|\hat{Y} \cap M_{\text{ovary}}|}{|M_{\text{ovary}}|}$), and Extra-Ovarian FPR.
- **Scientific Impact**:
  - The segmentation design uncovers a critical spatial failure mode: zero-masking creates false-positive cystic lesions inside normal ovaries (Intra-Ovary FPR $+9.14\%$, $p = 0.0005$), whereas in classification this spatial nuance was partially masked.

### Discrepancy 3: Resolution & Augmentation Pipeline
- **Classification**:
  - Input resolution: $224 \times 224$. Standard RandomResizedCrop, RandomHorizontalFlip.
  - Seeds set independently for PyTorch, NumPy, and random.
- **Segmentation**:
  - Input resolution: $512 \times 512$ (preserves fine caliper geometries and thin cyst walls).
  - Paired spatial augmentations: The data loader implements deterministic epoch-varying spatial transformations (HorizontalFlip, ShiftScaleRotate) seeded as `epoch * 100000 + idx`. This guarantees that for any given epoch, the exact same spatial transform is applied to Arm A, Arm B, Arm C, and the ground-truth mask, eliminating augmentation variance between arms.

---

## 3. Methodological Reassurance

These differences represent task-appropriate experimental designs, not accidental implementation drift:
- Classification requires low-resolution CNN inference with probability calibration (ECE/Brier) on severe class imbalance ($3.6\%$ malignant).
- Segmentation requires high-resolution U-Net spatial feature maps with paired geometric transforms and separate benign specificity auditing.
- Both experiments maintained 100% test-set isolation, reproducible random seeds, and strict exterior conservation.

