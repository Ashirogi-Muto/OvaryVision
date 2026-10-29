# FULL_COHORT_BENCHMARK_RESULTS.md — Full 18-Run 3-Arm Empirical Benchmark Report

**Project**: OvaryVision (`/home/ubuntu/major`)  
**Evaluation Date**: October 29, 2026 (Completed 11:17 UTC / 4:47 PM IST)  
**Execution Environment**: NVIDIA RTX PRO 6000 Blackwell Server Edition (94.97 GB VRAM) via Google Colab CLI  
**Total Compute Duration**: 24.24 minutes across all 18 models  
**Hardware Verification**: PyTorch 2.x, CUDA 12.x, OpenCV 5.0.0  

---

## 1. Executive Summary

This report documents the completed, full-cohort, 3-arm empirical benchmark for OvaryVision across all 1,469 MMOTU B-mode ultrasound scans. 

Prior to model training, automated artifact extraction identified clinician measurement overlays in **1,402 of 1,469 scans (95.4%)**. All scans were preprocessed in parallel into **Arm B (Harmonic Inpainting with 2px boundary dilation and strict exterior conservation)** and **Arm C (Zero-Masking with 2px boundary dilation and strict exterior conservation)** in 90.4 seconds.

A complete $3 \times 2 \times 3$ factorial design was executed:
- **3 Arms**: Arm A (Raw baseline with calipers), Arm B (Inpainted), Arm C (Zero-Masked)
- **2 Backbones**: ConvNeXt-Small, EfficientNet-B3
- **3 Seeds**: 42, 1337, 2026 (Fixed Group-Stratified splits isolating 1,288 dHash pseudo-patient clusters)

All 18 models were trained with Class-Weighted Focal Loss ($\gamma = 2.0$), calibrated via post-hoc validation-only Temperature Scaling ($T^*$), and evaluated on internal test partitions ($N = 220$ scans, $N_{pos} = 8$ malignant, $N_{neg} = 212$ benign).

---

## 2. Master Empirical Benchmark Results Table

| Arm | Backbone | Seed | Best Epoch | Optimal $T^*$ | Test AUROC | Calibrated ECE | Calibrated Brier | Train Time (s) |
|---|---|---|---|---|---|---|---|---|
| **Arm A (Raw)** | ConvNeXt-Small | 42 | 20 | 0.90 | **0.9298** | 0.0283 | 0.0255 | 105.3 |
| **Arm A (Raw)** | ConvNeXt-Small | 1337 | 17 | 0.82 | **0.8349** | 0.0057 | 0.0319 | 102.0 |
| **Arm A (Raw)** | ConvNeXt-Small | 2026 | 19 | 0.77 | **0.7476** | 0.0162 | 0.0336 | 102.5 |
| **Arm A (Raw)** | EfficientNet-B3 | 42 | 19 | 0.72 | **0.9204** | 0.0196 | 0.0318 | 43.2 |
| **Arm A (Raw)** | EfficientNet-B3 | 1337 | 15 | 0.73 | **0.8892** | 0.0100 | 0.0329 | 42.6 |
| **Arm A (Raw)** | EfficientNet-B3 | 2026 | 18 | 0.68 | **0.7235** | 0.0069 | 0.0339 | 42.9 |
| **Arm B (Inpainted)** | ConvNeXt-Small | 42 | 9 | 0.80 | **0.9292** | 0.0155 | 0.0310 | 96.1 |
| **Arm B (Inpainted)** | ConvNeXt-Small | 1337 | 18 | 0.91 | **0.8308** | 0.0055 | 0.0317 | 113.4 |
| **Arm B (Inpainted)** | ConvNeXt-Small | 2026 | 16 | 0.71 | **0.7288** | 0.0147 | 0.0341 | 112.2 |
| **Arm B (Inpainted)** | EfficientNet-B3 | 42 | 9 | 0.71 | **0.8620** | 0.0078 | 0.0330 | 50.6 |
| **Arm B (Inpainted)** | EfficientNet-B3 | 1337 | 17 | 0.75 | **0.8402** | 0.0033 | 0.0330 | 59.0 |
| **Arm B (Inpainted)** | EfficientNet-B3 | 2026 | 19 | 0.68 | **0.6621** | 0.0080 | 0.0343 | 58.7 |
| **Arm C (Zero-Masked)** | ConvNeXt-Small | 42 | 17 | 1.05 | **0.9239** | 0.0131 | 0.0285 | 113.0 |
| **Arm C (Zero-Masked)** | ConvNeXt-Small | 1337 | 17 | 0.85 | **0.8373** | 0.0077 | 0.0320 | 113.6 |
| **Arm C (Zero-Masked)** | ConvNeXt-Small | 2026 | 9 | 0.70 | **0.7117** | 0.0080 | 0.0345 | 95.1 |
| **Arm C (Zero-Masked)** | EfficientNet-B3 | 42 | 17 | 0.72 | **0.8909** | 0.0049 | 0.0331 | 59.4 |
| **Arm C (Zero-Masked)** | EfficientNet-B3 | 1337 | 17 | 0.75 | **0.9381** | 0.0171 | 0.0316 | 59.0 |
| **Arm C (Zero-Masked)** | EfficientNet-B3 | 2026 | 19 | 0.69 | **0.6663** | 0.0070 | 0.0344 | 58.6 |

---

## 3. Macro-Average Comparative Aggregates

| Preprocessing Arm | Mean Test AUROC | AUROC Std | Mean Calibrated Brier | Brier Std | Mean Calibrated ECE | ECE Std | Mean Train Time |
|---|---|---|---|---|---|---|---|
| **Arm A (Raw Baseline)** | **0.8409** | 0.0884 | 0.0316 | 0.0031 | 0.0145 | 0.0087 | 73.1s |
| **Arm B (Harmonic Inpainting)** | **0.8089** | 0.0967 | 0.0329 | 0.0013 | 0.0091 | 0.0049 | 81.7s |
| **Arm C (Non-Generative Masking)** | **0.8280** | 0.1140 | 0.0324 | 0.0022 | 0.0096 | 0.0046 | 83.1s |

### Core Hypotheses & Findings:
1. **Zero-Masking Matches and Exceeds Inpainting**:
   Across the full 6 paired model comparisons, Non-Generative Zero-Masking (Arm C) achieved a **macro-average test AUROC of 0.8280**, exceeding Harmonic Inpainting (Arm B, **0.8089**) by **+0.0191 (+1.91% AUROC)**.
2. **Calibration Error Reduction**:
   Both artifact removal arms reduced Expected Calibration Error (ECE) substantially relative to uncleaned scans:
   - Arm A (Uncleaned): $ECE = 0.0145 \pm 0.0087$
   - Arm B (Inpainted): $ECE = 0.0091 \pm 0.0049$ (37.2% reduction in calibration error)
   - Arm C (Masked): $ECE = 0.0096 \pm 0.0046$ (33.8% reduction in calibration error)
3. **Exploratory Comparison Between Raw and Masked Models**:
   Comparing Arm A (Raw scans) to Arm C (Zero-masked scans): the observed macro-average test AUROC difference across runs was $0.0129$ ($0.8409$ vs $0.8280$). In this 8-case test set, masking detected calipers did not produce an observed collapse in test discrimination, though this observation remains exploratory due to small positive test counts ($N_{pos}=8$) and substantial variance across random seeds. Direct proof of whether models rely on calipers as a discriminative shortcut requires controlled counterfactual intervention or attribution experiments, which have not yet been performed.

---

## 4. Paired DeLong & Stratified Bootstrap Non-Inferiority Analysis

Evaluation of the primary pre-registered hypothesis:
$$H_0: \text{AUC}_{\text{Arm C}} - \text{AUC}_{\text{Arm B}} \le -0.02 \quad \text{vs} \quad H_1: \text{AUC}_{\text{Arm C}} - \text{AUC}_{\text{Arm B}} > -0.02 \quad (\alpha = 0.05)$$

The one-sided 95% lower confidence bound is $\hat{\Delta}_{\text{observed}} - 1.6449 \cdot SE_{\text{DeLong}}$. Non-inferiority is declared if and only if $\text{Lower Bound} > -0.0200$.

| Architecture | Seed | $\text{AUC}_B$ (Inpaint) | $\text{AUC}_C$ (Mask) | Difference ($\hat{\Delta}_{C-B}$) | DeLong $p$-value (2-tailed) | One-Sided Non-Inf $p_{\text{NI}}$ | 95% Bootstrap CI | Non-Inferiority Declared? |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt-Small | 42 | 0.9292 | 0.9239 | -0.0053 | 0.8337 | 0.2805 | [-0.0584, +0.0377] | **False** (Lower 95%: -0.0469) |
| ConvNeXt-Small | 1337 | 0.8308 | 0.8373 | **+0.0065** | 0.8310 | 0.1917 | [-0.0425, +0.0690] | **False** (Lower 95%: -0.0435) |
| ConvNeXt-Small | 2026 | 0.7288 | 0.7117 | -0.0171 | 0.6459 | 0.4689 | [-0.0908, +0.0495] | **False** (Lower 95%: -0.0783) |
| EfficientNet-B3 | 42 | 0.8620 | 0.8909 | **+0.0289** | 0.4702 | 0.1108 | [-0.0431, +0.1103] | **False** (Lower 95%: -0.0369) |
| EfficientNet-B3 | 1337 | 0.8402 | 0.9381 | **+0.0979** | 0.1728 | 0.0503 | [+0.0088, +0.2659] | **False** (Lower 95%: -0.0202) |
| EfficientNet-B3 | 2026 | 0.6621 | 0.6663 | **+0.0041** | 0.9377 | 0.3237 | [-0.0920, +0.1109] | **False** (Lower 95%: -0.0827) |

### Rigorous Statistical Synthesis:
- **Observed Advantage**: In **4 of the 6 runs (66.7%)**, Arm C achieved a higher observed AUROC than Arm B (mean paired difference across runs: **$+0.0192$**).
- **Formal Non-Inferiority Result**: Non-inferiority within the narrow margin $\delta = -0.02$ was **not formally established** in any individual run. This failure is a direct consequence of low statistical power on an internal test split containing only **$N_{pos} = 8$ malignant scans** ($SE_{\text{DeLong}} = 0.0253 - 0.0718$).
- **Scientific Caveat**: The fact that $p > 0.05$ across pairwise DeLong tests must **never** be interpreted as proof of equivalence or non-inferiority ("absence of evidence is not evidence of absence"). Definitive evaluation of non-inferiority requires an adequately powered independent external test cohort with verified histopathology. An unsupported scalar sample size target cannot substitute for a full paired-AUROC power model.

---

## 5. Artifact Mask Extraction & Quality Control (QC)

- **Detection Rate**: Automated morphological extraction identified measurement calipers in **1,402 of 1,469 scans (95.4%)**.
  - Malignant Scans: 52 of 53 (98.1%) contain calipers (mean tumor overlap: 5.02%, maximum 46.38%).
  - Benign Scans: 1,350 of 1,416 (95.3%) contain calipers.
- **QC Status**: 1,335 scans (90.9%) passed all QC criteria (area $\le 5\%$, tumor overlap $\le 50\%$). 67 scans were flagged empty (no caliper), 56 flagged tumor overlap, and 11 flagged excessive area.
- **Visual Contact Sheets**:
  - `outputs/qc/contact_sheets/malignant_cases_artifact_contact_sheet.png`: Complete contact sheet for all 53 malignant scans with overlaid caliper detections (yellow) and tumor boundaries (cyan contour).
  - `outputs/qc/contact_sheets/benign_sample_artifact_contact_sheet.png`: 36 representative benign scans.
- **Provenance Disclosure**: These masks were generated using automated high-intensity morphological filtering and are strictly separated from clinical tumor segmentation masks (`data/raw/mmotu/tumor_masks/`).

---

## 6. Summary of Deliverables & Artifacts Generated

1. `outputs/benchmark_runs/full_benchmark/master_benchmark_summary.csv`: Complete run-level metrics for all 18 models.
2. `outputs/benchmark_runs/full_benchmark/master_benchmark_summary.json`: Machine-readable summary for automated parsing.
3. `outputs/benchmark_runs/full_benchmark/{run_name}/predictions.csv`: 18 test prediction CSVs with individual scan logits, calibrated probabilities, and true labels.
4. `outputs/benchmark_runs/full_benchmark/{run_name}/metrics.json`: Detailed run logs with best epoch and optimal temperature parameters.
5. `outputs/metrics/full_cohort_delong_results.csv`: Paired DeLong test statistics and bootstrap confidence intervals.
6. `outputs/metrics/full_cohort_mask_qc_summary.csv`: Cohort-wide mask detection, area ratio, and tumor overlap metrics for all 1,469 scans.
7. `outputs/qc/contact_sheets/`: Visual contact sheets for all 53 malignant cases and 36 representative benign cases.
