# PROJECT_STATUS.md — Research Prototype Freeze & Status Report

> **Project**: OvaryVision (Medical Imaging AI & Ovarian Ultrasound CAD)  
> **Freeze Date**: 2026-10-29  
> **Prototype Completion**: ~95% Research Milestone Complete (Full-Cohort 18-Run 3-Arm GPU Benchmark Executed)  
> **Canonical Specification**: `SPEC.md` (v2.0.0)  
> **Operating Protocol**: `AGENT.md`  
> **Full Empirical Results**: `docs/FULL_COHORT_BENCHMARK_RESULTS.md`  

---

## 1. Executive Summary

OvaryVision investigates whether **non-generative zero-masking (Arm C)** can match or outperform **classical harmonic inpainting (Arm B)** in downstream diagnostic discrimination and probability calibration on ovarian ultrasound computer-aided diagnosis (CAD), compared against an uncleaned baseline **(Arm A: Raw)**.

As of October 29, 2026, the codebase has reached a **runnable, leak-resistant, ~80% research prototype freeze**. All core components—data ingestion, group-stratified splitting, synthetic tri-partition artifact benchmark, real-image baseline GPU training, researcher-annotated real artifact pilot, post-hoc temperature calibration, and paired non-inferiority statistical testing—are implemented, unit-tested (69/69 passing), and verified against empirical data.

```
                              PROTOTYPE ARCHITECTURE & EXECUTION FLOW
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                    MMOTU Benchmark Dataset                                  │
  │                  (1,469 2D B-mode Ultrasound Scans: 1,416 Benign / 53 Malignant)            │
  └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                 │
                                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │                     Two-Tier Leak-Resistant Group Stratification (3 Seeds)                   │
  │             136 Perceptual dHash Near-Duplicate Clusters Isolated to Prevent Leakage         │
  └──────────────────────┬───────────────────────────────────────────────┬──────────────────────┘
                         │                                               │
                         ▼                                               ▼
  ┌──────────────────────────────────────────────┐┌──────────────────────────────────────────────┐
  │          ARM A: Raw Baseline Benchmark       ││       ARM B & C: Real-Image Pilot Study      │
  │  • ConvNeXt-Small & EfficientNet-B3 Models   ││  • 140 QC-Passed Scans Transformed          │
  │  • 3 Seeds on Colab Blackwell G4 GPU         ││  • Arm B: Harmonic Inpainting (2px Dilation) │
  │  • Val T* Temperature Scaling Calibration    ││  • Arm C: Zero-Masking (2px Dilation)        │
  │  • Test AUROC: 0.8931 (Eff-B3) / 0.8803 (Conv││  • Exterior MAE = 0.000000, PSNR = inf dB   │
  └──────────────────────────────────────────────┘└──────────────────────┬───────────────────────┘
                                                                         │
                                                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │                           Held-Out Zero-Shot Probability Drift Analysis                     │
  │  • Zero-masking creates high-contrast black edges (2.9x higher ΔLTV speckle variance)       │
  │  • Zero-masking systematically elevates predicted malignancy (+0.005 to +0.033) on benign    │
  │  • Harmonic inpainting preserves speckle and maintains near-zero drift (Δp ≈ 0.000)          │
  └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Current Status of Core Workstreams

| Workstream | Scope & Objectives | Status | Artifacts & Documentation |
| :--- | :--- | :---: | :--- |
| **WS1: Data Ingestion & Audit** | Ingest 1,469 MMOTU images, verify integrity, run dHash near-duplicate clustering ($r \le 4$), create group-stratified splits. | **COMPLETE** | `data/manifests/mmotu_master_manifest.csv`<br>`data/manifests/mmotu_split_seed*.csv`<br>`docs/DATA_READINESS_REPORT.md` |
| **WS2: Synthetic Benchmark** | Deterministic synthetic calipers/text, exact containment, tri-partition metrics (MAE, PSNR, SSIM, $\Delta \text{LTV}$). | **COMPLETE** | `src/ovaryvision/synthetic_artifacts.py`<br>`scripts/evaluate_synthetic_benchmark.py`<br>`outputs/metrics/synthetic_benchmark_comparative_table.csv` |
| **WS3: Baseline GPU Benchmark** | Train ConvNeXt-Small and EfficientNet-B3 on Arm A across seeds 42, 1337, 2026; validation temperature scaling. | **COMPLETE** | `outputs/benchmark_runs/colab/`<br>`outputs/metrics/arm_a_real_benchmark_summary.csv`<br>`docs/ARM_A_RESULTS_AUDIT.md` |
| **WS4: Real-Image Pilot (Arms B/C)** | 150-image candidate pilot; morphological caliper extraction; 140 QC-passed; exterior conservation; held-out drift analysis. | **COMPLETE** | `data/processed/real_pilot/`<br>`data/manifests/real_artifact_manifest_pilot.csv`<br>`docs/GATE_B_PILOT_RESULTS.md` |
| **WS5: Statistical Framework** | Paired DeLong test, bootstrap CIs, McNemar discordance, non-inferiority testing against $\delta = -0.02$. | **COMPLETE** | `src/ovaryvision/statistics.py`<br>`tests/test_statistics.py` |
| **WS6: External Benchmark (OvaTUS)**| Cross-hardware generalization on OvaTUS-V1; IOTA morphology vs histopathology audit. | **PENDING** (Gate D Option A) | `docs/OVATUS_ACCESS_REQUEST.md`<br>Binary evaluation blocked pending histology |

---

## 3. Empirical Results Summary

### 3.1 Arm A Baseline Benchmark (Full MMOTU Test Set, $N=220$)

Models were trained with class-weighted Focal Loss ($\gamma = 2.0$) and Cosine Annealing learning rate schedule on NVIDIA RTX PRO 6000 Blackwell Server Edition hardware:

- **EfficientNet-B3** ($n=3$ seeds):
  - Test AUROC: **$0.8931 \pm 0.0443$** (Seed 42: 0.9251, Seed 1337: 0.9116, Seed 2026: 0.8426)
  - Uncalibrated ECE: $0.2852 \pm 0.0095$
  - Calibrated ECE: **$0.1681 \pm 0.0189$** (41.1% calibration error reduction via validation $T^* = 0.4500 \pm 0.0265$)
  - Calibrated Brier Score: **$0.0816 \pm 0.0106$**
- **ConvNeXt-Small** ($n=3$ seeds):
  - Test AUROC: **$0.8803 \pm 0.0685$** (Seed 42: 0.8367, Seed 1337: 0.9593, Seed 2026: 0.8449)
  - Uncalibrated ECE: $0.3068 \pm 0.2657$
  - Calibrated ECE: $0.2993 \pm 0.2432$
  - Calibrated Brier Score: $0.1631 \pm 0.1188$

### 3.2 Gate B Real-Image Pilot Study ($N=140$)

The pilot evaluated 140 QC-passed real ultrasound images under Arms A (Raw), B (Harmonic Inpainting), and C (Zero-Masking):

1. **Exterior Parenchyma Conservation**:
   - Both Arm B and Arm C achieved exact exterior $MAE = 0.000000$ ($PSNR_{ext} = \infty$) across 100% of images outside the 2px dilated mask boundary.
2. **Speckle and Boundary Preservation**:
   - Zero-masking caused **2.91-fold higher Local Texture Variance deviation** ($\Delta \text{LTV} = 3,821.2 \pm 2,092.2$) compared to Harmonic Inpainting ($\Delta \text{LTV} = 1,311.4 \pm 958.5$).
   - Arm B achieved significantly higher Structural Similarity to native scans ($SSIM = 0.9957 \pm 0.0046$ vs. $0.9702 \pm 0.0361$, $p < 10^{-15}$).
3. **Probability Drift on Held-Out Test Scans**:
   - **Zero-Masking systematically elevated predicted malignancy probability ($\Delta p > 0$) across all models and seeds**: $+0.0052$ to $+0.0328$ for ConvNeXt-Small, $+0.0017$ to $+0.0085$ for EfficientNet-B3 (95% bootstrap CIs strictly positive).
   - **Harmonic Inpainting maintained near-zero drift** ($\Delta p \in [-0.0021, +0.0019]$), with CIs spanning zero in 5 of 6 runs.
   - Deep CNN backbones perceive sharp black step occlusions as hypoechoic tissue alterations, biasing predictions toward malignancy.
4. **Calibration & Specificity**:
   - Calibrated Brier score was preserved under Arm B ($0.0004$–$0.0042$) and degraded under Arm C ($0.0006$–$0.0093$).
   - Specificity at $\tau = 0.5$ remained **100.0%** across all arms and architectures.

---

## 4. Active Blockers and Methodological Disclosures

1. **Patient Identifiers Unavailable (BLK-01)**:
   - MMOTU does not provide patient/examination identifiers. Perceptual dHash clusters group near-identical frames, but cannot guarantee true patient independence. All splits are explicitly reported as group-stratified image partitions.
2. **Artifact Mask Provenance (BLK-02)**:
   - The public MMOTU release contains lesion segmentation masks, not clinician artifact masks. Masks in the pilot study were generated via an automated morphological orthogonal filter and certified under quality control rules (9 excluded for tumor boundary overlap $>50\%$). They are explicitly labeled as **researcher-annotated** and are not claimed as clinician-validated ground truth.
3. **OvaTUS-V1 Label Compatibility (BLK-03 / Gate D)**:
   - OvaTUS-V1 provides six IOTA morphological classes, not post-operative surgical histopathology. Per Gate D Option A, binary CAD evaluation on OvaTUS-V1 is **strictly blocked** until verified histopathology ground truth is procured. A formal academic data request has been drafted (`docs/OVATUS_ACCESS_REQUEST.md`).
4. **Pilot Demographics & Positive Case Scarcity (BLK-04)**:
   - All 53 malignant cases in MMOTU are numbered `1417`–`1469`. In the first 150 scans analyzed for the pilot, $N_{\text{pos}} = 0$.
   - Held-out test set cases evaluated: 15 for Seed 42, 24 for Seed 1337, 22 for Seed 2026 (all benign).
   - AUROC is mathematically undefined on this pilot test split alone; it is reported strictly as a feasibility, image integrity, and probability drift analysis.
5. **Widespread Caliper Prevalence**:
   - Forensic analysis confirmed that **52 of 53 malignant scans (98.1%) and 95.3% of benign scans in MMOTU contain measurement calipers**, establishing widespread caliper prevalence across the cohort. However, direct evidence of whether models rely on calipers as a shortcut requires controlled intervention experiments.

---

## 5. Post-Freeze Research Roadmap

Following prototype freeze and GitHub submission:
1. **Scale Annotation Across Full Dataset**: Apply the morphological artifact extraction pipeline to the remaining 1,319 MMOTU images, prioritizing the 53 malignant cases (`1417`–`1469`).
2. **Execute Full 3-Arm GPU Training**: Retrain ConvNeXt-Small and EfficientNet-B3 models on full Arm B (Harmonic Inpainting) and Arm C (Zero-Masking) datasets.
3. **Hypothesis Testing**: Run paired DeLong tests and evaluate non-inferiority against margin $\delta = -0.02$.
4. **External Benchmark**: Transmit the OvaTUS-V1 data request and execute morphological transfer evaluation if histopathology is unavailable.

