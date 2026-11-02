# Classification Benchmark Integration Audit: OvaryVision

**Date**: 2026-11-01  
**Auditor**: Antigravity (Advanced Agentic Coding / DeepMind)  
**Target Directory**: `outputs/benchmark_runs/full_benchmark/`  
**Master Records**: `master_benchmark_summary.csv`, `master_benchmark_summary.json`  

---

## 1. Inventory & Immutability Verification

All 18 planned classification runs were verified on local disk:
- Matrix: 3 Preprocessing Arms (`arm_a`, `arm_b`, `arm_c`) $\times$ 2 Backbones (`convnext_small`, `efficientnet_b3`) $\times$ 3 Random Seeds (`42`, `1337`, `2026`) = **18 runs**.
- Every run directory contains its original `metrics.json` and `predictions.csv` (220 test scan predictions).
- **Immutability Status**: All file timestamps remain strictly between `2026-10-29 10:52 UTC` and `2026-10-29 11:15 UTC`. **Zero classification files were altered, overwritten, or modified** during the October 30 or November 1 segmentation sprints.

---

## 2. Complete 18-Run Experimental Summary

The table below reproduces the authentic, unedited classification benchmark results across all 18 runs:

| Arm | Backbone | Seed | Best Val Epoch | Best Val AUROC | Optimal $T^*$ | Test AUROC | Calibrated ECE | Calibrated Brier | Train Time (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `arm_a` | `convnext_small` | 42 | 20 | 0.8756 | 0.90 | 0.9298 | 0.0283 | 0.0255 | 105.3 |
| `arm_a` | `convnext_small` | 1337 | 17 | 0.8237 | 0.82 | 0.8349 | 0.0057 | 0.0319 | 102.0 |
| `arm_a` | `convnext_small` | 2026 | 19 | 0.9068 | 0.77 | 0.7476 | 0.0162 | 0.0336 | 102.5 |
| `arm_a` | `efficientnet_b3` | 42 | 19 | 0.7152 | 0.72 | 0.9204 | 0.0196 | 0.0318 | 43.2 |
| `arm_a` | `efficientnet_b3` | 1337 | 15 | 0.6285 | 0.73 | 0.8892 | 0.0100 | 0.0329 | 42.6 |
| `arm_a` | `efficientnet_b3` | 2026 | 18 | 0.8886 | 0.68 | 0.7235 | 0.0069 | 0.0339 | 42.9 |
| `arm_b` | `convnext_small` | 42 | 9 | 0.8821 | 0.80 | 0.9292 | 0.0155 | 0.0310 | 96.1 |
| `arm_b` | `convnext_small` | 1337 | 18 | 0.8025 | 0.91 | 0.8308 | 0.0055 | 0.0317 | 113.4 |
| `arm_b` | `convnext_small` | 2026 | 16 | 0.9216 | 0.71 | 0.7288 | 0.0147 | 0.0341 | 112.2 |
| `arm_b` | `efficientnet_b3` | 42 | 9 | 0.6928 | 0.71 | 0.8620 | 0.0078 | 0.0330 | 50.6 |
| `arm_b` | `efficientnet_b3` | 1337 | 17 | 0.6150 | 0.75 | 0.8402 | 0.0033 | 0.0330 | 59.0 |
| `arm_b` | `efficientnet_b3` | 2026 | 19 | 0.8927 | 0.68 | 0.6621 | 0.0080 | 0.0343 | 58.7 |
| `arm_c` | `convnext_small` | 42 | 17 | 0.8844 | 1.05 | 0.9239 | 0.0131 | 0.0285 | 113.0 |
| `arm_c` | `convnext_small` | 1337 | 17 | 0.8202 | 0.85 | 0.8373 | 0.0077 | 0.0320 | 113.6 |
| `arm_c` | `convnext_small` | 2026 | 9 | 0.8939 | 0.70 | 0.7117 | 0.0080 | 0.0345 | 95.1 |
| `arm_c` | `efficientnet_b3` | 42 | 17 | 0.6657 | 0.72 | 0.8909 | 0.0049 | 0.0331 | 59.4 |
| `arm_c` | `efficientnet_b3` | 1337 | 17 | 0.6527 | 0.75 | 0.9381 | 0.0171 | 0.0316 | 59.0 |
| `arm_c` | `efficientnet_b3` | 2026 | 19 | 0.8892 | 0.69 | 0.6663 | 0.0070 | 0.0344 | 58.6 |

---

## 3. Macro-Averaged Performance per Preprocessing Arm

Averaged across all 6 models per arm (2 backbones $\times$ 3 seeds):

| Metric | Arm A (Raw Baseline) | Arm B (Harmonic Inpainting) | Arm C (Zero-Masking) | Observed Difference ($\Delta_{C-B}$) |
|---|:---:|:---:|:---:|:---:|
| **Test AUROC** | $0.8409 \pm 0.0884$ | $0.8088 \pm 0.0967$ | $\mathbf{0.8280 \pm 0.1140}$ | $\mathbf{+0.0192}$ (+1.92%) |
| **Calibrated ECE** | $0.0144 \pm 0.0087$ | $\mathbf{0.0091 \pm 0.0049}$ | $0.0096 \pm 0.0046$ | $+0.0005$ |
| **Calibrated Brier** | $\mathbf{0.0316 \pm 0.0031}$ | $0.0328 \pm 0.0013$ | $0.0323 \pm 0.0022$ | $-0.0005$ |
| **Total Train Time** | **438.5 s** | 490.0 s | 498.7 s | $+8.7$ s |

---

## 4. Methodological & Statistical Limitations of the Classification Benchmark

1. **Extreme Test Set Positive Class Scarcity ($N_{\text{pos}} = 8$)**:
   - The held-out test split of 220 scans contains exactly 212 benign scans and **8 malignant scans** (3.64% prevalence).
   - This induces severe sensitivity quantization: a single classification flip alters sensitivity by **12.5%** ($\frac{1}{8}$).
   - As documented in `docs/FULL_COHORT_AUDIT_REPORT.md`, paired DeLong standard errors for $\Delta \text{AUC}$ on this cohort range from $0.025$ to $0.072$. Consequently, formal non-inferiority against $\delta = -0.02$ cannot be statistically rejected ($p > 0.05$ does *not* imply equivalence).
2. **Seed Instability Across Architectures**:
   - On EfficientNet-B3, Seed 2026 underperformed substantially (AUROC $0.6621$ on Arm B, $0.6663$ on Arm C) relative to Seed 42 ($0.8620$ and $0.8909$), illustrating high optimization variance on small sample sizes.
3. **Contrast with Segmentation Benchmark**:
   - While classification macro-average observed AUROC favored Arm C over Arm B ($0.8280$ vs $0.8088$, $+1.92\%$), the segmentation benchmark evaluates spatial pixel-level predictions on 382 tumour cases and 87 normal cases, uncovering spatial failure modes (in-caliper lesion dropout and benign cystic false alarms) that a global image-level scalar classification probability cannot capture.

