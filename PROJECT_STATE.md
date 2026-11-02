# PROJECT_STATE.md — Continuous Project State Tracker: OvaryVision

**Current Date / Timestamp**: 2026-11-01
**Current Phase**: Three-Seed, Three-Arm Segmentation Benchmark Complete & Aggregated
**Current Status**: Complete on `exp/full-cohort-benchmark` (Ready for Supervisor / User Review)
**Canonical Specification**: `SPEC.md` (v2.0.0)
**Audit Documents**: `docs/MILESTONE_1_AUDIT.md`, `docs/FULL_COHORT_AUDIT_REPORT.md`, `docs/MMOTU_DATASET_COMPLETENESS_AUDIT.md`, `docs/SEGMENTATION_PRE-EXECUTION_AUDIT.md`

---

## 1. Project Objective
Establish a reproducible, leak-resistant, and clinically grounded data-centric benchmark comparing **Harmonic Inpainting (Arm B)** against **Non-Generative Masking (Arm C: Zero-Occlusion)** and **Raw Uncleaned Baseline (Arm A)** in ovarian ultrasound computer-aided diagnosis across internal (MMOTU) and independent external (OvaTUS-V1) cohorts.

---

## 2. Implementation Inventory

| Component / File | Category | Status | Verification / Test Method |
|---|---|---|---|
| `SPEC.md` | Specification | Implemented | v2.0.0 Canonical research specification with audited assumptions |
| `AGENT.md` | Operating Protocol & Memory | Implemented | Merged canonical agent memory and integrity rules |
| `PROJECT_STATE.md` | Project Memory | Implemented | Live state file tracking deliverables and blockers |
| `DECISIONS.md` | Decision Log | Implemented | Append-only architectural & methodology log (D01–D31) |
| `IMPLEMENTATION_PLAN.md` | Roadmap | Implemented | Dependency-aware milestones and task tracking |
| `CHANGELOG.md` | History | Implemented | Foundation, Milestone 2, Core Pipeline & Completeness Audit updates |
| `README.md` | Documentation | Implemented | Project entry point, CLI guide, and architecture |
| `docs/RESEARCH_PROTOCOL.md` | Research Protocol | Implemented | Detailed 3-arm protocol, calibration & metrics |
| `docs/FULL_COHORT_BENCHMARK_RESULTS.md` | Results Report | Implemented | 18-run benchmark report with macro-averages and metrics |
| `docs/FULL_COHORT_AUDIT_REPORT.md` | Audit Report | Implemented | Forensic audit of cohort, power, non-inferiority, mask QC |
| `docs/MMOTU_DATASET_COMPLETENESS_AUDIT.md` | Completeness Audit | Implemented | Forensic audit across 6 MMOTU releases, CEUS recovery, mask provenance |
| `docs/CEUS_AND_SEGMENTATION_EXPERIMENT_PLAN.md` | Experiment Plan | Implemented | 3-arm segmentation design, CDFI verification, CEUS transfer protocol |
| `docs/SEGMENTATION_PRE-EXECUTION_AUDIT.md` | Pre-Execution Audit | Implemented | Split integrity, CDFI contact sheet, LPR definition, architecture |
| `outputs/metrics/mmotu_release_comparison.csv` | Dataset Manifest | Implemented | Machine-readable comparison table across 7 releases |
| `outputs/metrics/ceus_data_integrity_audit.csv` | Audit Output | Implemented | 170-case per-file verification with SHA-256 and area fractions |
| `data/manifests/mmotu_ceus_master_manifest.csv` | Dataset Manifest | Implemented | 170-case CEUS master manifest with dHash clusters, splits, targets |
| `data/manifests/mmotu_2d_cdfi_classification.csv` | Dataset Manifest | Implemented | 1,469-case 2D CDFI classification with confidence tiers |
| `data/manifests/mmotu_2d_segmentation_split.csv` | Dataset Manifest | Implemented | 850/150 internal train/val cluster-preserved split, 469 held-out test |
| `scripts/run_three_arm_segmentation_benchmark.py` | Benchmark Script | Implemented | 3-arm segmentation runner with AMP, epoch-pairing, custom collation |
| `scripts/evaluate_seed42_test_cohort.py` | Evaluation Script | Implemented | Test cohort evaluator with BCa paired cluster bootstrap & plotting |
| `scripts/aggregate_three_seed_benchmark.py` | Aggregator Script | Implemented | 3-seed multi-run aggregator with scan-level paired cluster BCa bootstrap |
| `outputs/benchmark_runs/segmentation_benchmark/` | Checkpoints & Runs | Implemented | All 9 completed runs (`arm_{a,b,c}_seed{42,1337,2026}`) with best checkpoints |
| `outputs/benchmark_runs/segmentation_benchmark/seed{42,1337,2026}_test_evaluation/` | Test Evaluation | Implemented | Per-seed held-out test evaluations on 469 scans with BCa bootstrap reports |
| `outputs/benchmark_runs/segmentation_benchmark/final_three_seed_benchmark/` | Aggregated Benchmark | Implemented | Pooled 3-seed tables, forest plot, tumour/normal comparison plots, master JSON |
| `data/raw/mmotu_ceus/` | Data Staging | Implemented | 170 CEUS scans, 170 binary masks, 170 RGB masks, official labels |
| `scripts/audit_masks_and_generate_contact_sheets.py` | QC Tooling | Implemented | Mask QC extraction, area/overlap metrics, contact sheet renderer |
| `outputs/qc/contact_sheets/` | Visual QC | Implemented | Contact sheets for all 53 malignant cases and 36 benign samples |
| `docs/DATA_DICTIONARY.md` | Schemas | Implemented | Data schemas for manifests, audits, records & synthetic metrics |
| `docs/EXPERIMENT_TRACKER.md` | Registry | Implemented | Reproducible experiment logging schema |
| `docs/architecture.md` | Architecture | Implemented | System components and OCI/GPU boundary |
| `docs/data_protocol.md` | Ingestion SOP | Implemented | Ingestion, mask contracts, and privacy rules |
| `docs/MMOTU_INGESTION_GUIDE.md` | Ingestion SOP | Implemented | Step-by-step authoritative MMOTU acquisition protocol |
| `docs/MILESTONE_1_AUDIT.md` | Scientific Audit | Implemented | Full critical audit of Milestone 1 claims and code |
| `configs/default.yaml` | Configuration | Implemented | Validated default project settings |
| `configs/experiment.yaml` | Configuration | Implemented | 3-arm experiment benchmark configuration |
| `pyproject.toml` | Packaging | Implemented | PEP 621 packaging with CLI entry point & pytest config |
| `src/ovaryvision/logging_utils.py` | Core Module | Implemented | 10MB x 5 rotating file logging in `logs/` |
| `src/ovaryvision/environment.py` | Core Module | Implemented | Host hardware/software diagnostic scanner |
| `src/ovaryvision/config.py` | Core Module | Implemented | Pydantic v2 typed configuration parser |
| `src/ovaryvision/manifests.py` | Core Module | Implemented | Two-tier leakage gate: image-level and cluster-level |
| `src/ovaryvision/provenance.py` | Core Module | Implemented | Deterministic execution provenance hashing |
| `src/ovaryvision/data_audit.py` | Core Module | Implemented | Image inventory, SHA256 & dHash near-duplicate auditor |
| `src/ovaryvision/synthetic_artifacts.py`| Core Module | Implemented | Synthetic calipers, exact containment, tri-partition metrics |
| `src/ovaryvision/splits.py` | Core Module | Implemented | Group-stratified train/val/test splitter with zero leakage |
| `src/ovaryvision/preprocessing.py` | Core Module | Implemented | 3-arm engine: Arm A (Raw), Arm B (Inpainted), Arm C (Masked) |
| `src/ovaryvision/calibration.py` | Core Module | Implemented | Validation Temperature Scaling, ECE, MCE, Brier score |
| `src/ovaryvision/annotation_workflow.py` | Core Module | Implemented | Caliper candidate extraction, QC rules, overlay generator |
| `src/ovaryvision/statistics.py` | Core Module | Implemented | Paired DeLong, BCa & percentile cluster bootstrap, non-inferiority |
| `src/ovaryvision/cli.py` | CLI Interface | Implemented | Click CLI with 11 core subcommands |
| `scripts/run_audit.py` | Automation | Implemented | Quick script runner for data audits |
| `scripts/ingest_mmotu.py` | Automation | Implemented | Safe, opt-in extraction and post-ingestion audit helper |
| `scripts/run_preprocessing.py` | Automation | Implemented | Batch preprocessing across image manifests |
| `scripts/gpu_train.py` | Automation | Implemented | Standalone PyTorch GPU training script for Colab/Kaggle |
| `scripts/run_artifact_annotation_pilot.py` | Automation | Implemented | 150-image real caliper mask extraction & QC pilot |
| `scripts/evaluate_synthetic_benchmark.py` | Automation | Implemented | Tri-partition comparative benchmark evaluator |
| `scripts/generate_figures.py` | Visualization | Implemented | Vector SVG/PNG figures for synthetic benchmark & calibration |
| `scripts/generate_tables.py` | Reporting | Implemented | LaTeX manuscript table generator for tri-partition integrity |
| `tests/` | Quality Gate | Implemented | Automated pytest suite (**86 tests passing in 4.1s**) |

---

## 3. Dataset Access & Audit Status

### MMOTU (Internal Benchmark)
- **Status**: **Downloaded & Fully Audited** (1,469 valid 2D B-mode ultrasound images in `data/raw/mmotu/images/`, 1,469 tumor segmentation masks in `data/raw/mmotu/tumor_masks/`).
- **Forensic Findings**:
  - 1,469 valid images, 0 corrupt, 0 exact duplicates.
  - Perceptual dHash analysis identified **136 near-duplicate clusters** spanning **290 images** ($d_H \le 4$).
  - Official 8-class labels audited: 1,416 Benign (Classes 0–6) vs 53 Malignant (Class 7; 3.61% prevalence).
  - Master manifest created at `data/manifests/mmotu_master_manifest.csv`.
  - Multi-seed leak-resistant split manifests generated for seeds 42, 1337, 2026 at `data/manifests/mmotu_split_seed*.csv`.
  - Gate B audit confirmed: public release provides tumor segmentation masks (`_binary.PNG`), **not** clinician artifact masks ($M_{art}$). Real-image Arm A baseline is active; real-image Arm B and C await author artifact masks or validated caliper detector.
- **Audit Tooling**: Complete and verified (`ovaryvision audit-data`, `docs/DATA_READINESS_REPORT.md`).

### OvaTUS-V1 (External Benchmark)
- **Status**: External multi-center dataset; restricted access under clinical data agreement.
- **Label Compatibility & Audit Finding**: Verified that OvaTUS-V1 contains **six IOTA morphological classes** (`Solid tumor`, `Multilocular cyst`, `Unilocular cyst`, `Dermoid cyst`, `Multilocular-solid cyst`, `Unilocular-solid cyst`), **not** confirmed benign/malignant histology.
- **Evaluation Status**: **Binary evaluation blocked** until patient-matched surgical pathology records are audited, or re-framed as an IOTA morphological transfer benchmark.

---

## 4. Active Blockers & Mitigation Status

| Blocker ID | Description | Impact | Current Mitigation / Status |
|---|---|---|---|
| **BLK-01** | MMOTU public release lacks patient IDs | Risk of train/test frame leakage | Implemented perceptual dHash near-duplicate clustering in `data_audit.py`. Limitations explicitly documented: cannot guarantee patient independence. |
| **BLK-02** | Absence of official annotation artifact masks in MMOTU | Cannot train MGGAN or apply Arm B/C without masks | Strict separation enforced between tumor masks and artifact masks in `manifests.py`. Author mask procurement or caliper detection required. |
| **BLK-03** | OvaTUS-V1 labels are IOTA morphology, not confirmed histology | Cannot run binary benign/malignant CAD evaluation directly | Binary evaluation blocked pending histological records; alternative morphological transfer benchmark documented. |
| **BLK-04** | OCI host is CPU-only (ARM64, no CUDA) | Cannot train deep models locally | Architecture strictly bifurcated: OCI handles orchestration, validation, and manifests; Colab/Kaggle handles GPU training (`scripts/gpu_train.py`). |

---

## 5. Recent Testing & Verification Evidence

- **Automated Tests**: Tested via `pytest tests/` (**86 of 86 tests passing** across 13 test suites in 4.09s):
  - `tests/test_annotation_workflow.py` (5 tests)
  - `tests/test_cli.py` (13 tests)
  - `tests/test_config.py` (4 tests)
  - `tests/test_data_audit.py` (4 tests)
  - `tests/test_environment.py` (3 tests)
  - `tests/test_manifests.py` (4 tests)
  - `tests/test_provenance.py` (2 tests)
  - `tests/test_synthetic_artifacts.py` (10 tests)
  - `tests/test_splits.py` (6 tests)
  - `tests/test_preprocessing.py` (6 tests)
  - `tests/test_calibration.py` (5 tests)
  - `tests/test_statistics.py` (13 tests)
  - `tests/test_segmentation_benchmark.py` (11 tests)
- **Controlled Synthetic Benchmark Results (Generated 2026-10-29)**:
  - Arm A (Raw Calipers): In-Mask MAE = 147.31 ± 10.74, PSNR = 3.78 ± 0.47 dB, $\Delta \text{LTV} = 4,867.3 \pm 331.2$.
  - Arm B (Inpainted): In-Mask MAE = 18.61 ± 1.49, PSNR = 20.75 ± 0.69 dB, Global SSIM = 0.9819, $\Delta \text{LTV} = 390.8 \pm 5.6$.
  - Arm C (Zero-Masked): In-Mask MAE = 40.93 ± 3.29, PSNR = 14.63 ± 0.56 dB, Global SSIM = 0.9711, $\Delta \text{LTV} = 749.4 \pm 14.9$.
  - Preserved in: `outputs/metrics/synthetic_benchmark_comparative_table.csv` and `outputs/figures/table1_synthetic_integrity.tex`.
- **Real MMOTU Arm A (Raw Baseline) Multi-Seed Benchmark (Executed 2026-10-29 on Colab G4 Blackwell GPU)**:
  - `efficientnet_b3` (3 seeds: 42, 1337, 2026):
    - Test AUROC: **$0.8931 \pm 0.0443$** (Seeds: 0.9251, 0.9116, 0.8426)
    - Uncalibrated ECE: $0.2852 \pm 0.0095$
    - Calibrated ECE: **$0.1681 \pm 0.0189$** (41.1% calibration error reduction via validation $T^* = 0.4500 \pm 0.0265$)
    - Calibrated Brier Score: **$0.0816 \pm 0.0106$**
  - `convnext_small` (3 seeds: 42, 1337, 2026):
    - Test AUROC: **$0.8803 \pm 0.0685$** (Seeds: 0.8367, 0.9593, 0.8449)
    - Uncalibrated ECE: $0.3068 \pm 0.2657$
    - Calibrated ECE: $0.2993 \pm 0.2432$
    - Calibrated Brier Score: $0.1631 \pm 0.1188$
  - Preserved in: `outputs/metrics/arm_a_real_benchmark_summary.csv` and `outputs/benchmark_runs/colab/`.
- **Real-Image Pilot Study (Gate B Approved & Executed 2026-10-29)**:
  - 150 candidate images evaluated: 140 passed QC (93.3%), 9 excluded for tumor overlap ($>50\%$), 1 empty mask excluded.
  - Exterior Conservation: $MAE_{ext} = 0.000000$ and $PSNR_{ext} = \infty$ across 100% of images for both Arm B and Arm C.
  - Texture Variance: Arm B In-Mask $\Delta \text{LTV} = 1,311.4 \pm 958.5$; Arm C In-Mask $\Delta \text{LTV} = 3,821.2 \pm 2,092.2$ (2.91-fold higher speckle variance deviation under zero-masking).
  - SSIM Distortion: Arm B $SSIM = 0.9957 \pm 0.0046$; Arm C $SSIM = 0.9702 \pm 0.0361$ ($p < 10^{-15}$).
  - Held-out Test Probability Drift ($\Delta p$ vs Raw):
    - Zero-Masking (Arm C) systematically elevated predicted malignancy: $+0.0052$ to $+0.0328$ for ConvNeXt-Small, $+0.0017$ to $+0.0085$ for EfficientNet-B3.
    - Harmonic Inpainting (Arm B) maintained near-zero drift: $-0.0021$ to $+0.0019$ ($95\%$ bootstrap CIs include zero in 5 of 6 runs).
    - Calibrated Brier score preserved/improved under Arm B ($0.0004$–$0.0042$) and degraded under Arm C ($0.0006$–$0.0093$).
  - Full report: `docs/GATE_B_PILOT_RESULTS.md`; metrics in `outputs/metrics/real_pilot_classification_metrics.csv` and `outputs/metrics/real_pilot_integrity_metrics.csv`.
- **Three-Arm Seed 42 Segmentation Pilot Benchmark (Executed 2026-10-30)**:
  - Architecture: SMP U-Net with ImageNet-pretrained ResNet-34 encoder ($512 \times 512$, batch size 8, AMP FP16).
  - Cohort: 850 `internal_train` scans, 150 `internal_val` scans (cluster-preserved partition); 469 test scans strictly held out (zero test exposure).
  - Runtime: Executed on Google Colab NVIDIA RTX PRO 6000 Blackwell Server Edition (94.97 GB VRAM).
  - Results across 40 epochs:
    - **Arm A (`arm_a_raw`)**: 11.78 min (mean 17.66s/epoch); best checkpoint selected at Epoch 29 (internal val loss `0.1352`, internal val DSC `0.8665`).
    - **Arm B (`arm_b_inpainted`)**: 57.15 min (mean 85.72s/epoch, reflective of on-the-fly OpenCV Navier-Stokes inpainting per scan); best checkpoint selected at Epoch 33 (internal val loss `0.1339`, internal val DSC `0.8648`).
    - **Arm C (`arm_c_masked`)**: 12.51 min (mean 18.76s/epoch); best checkpoint selected at Epoch 32 (internal val loss `0.1319`, internal val DSC `0.8623`).
  - Checkpoints and run summaries verified and saved to `outputs/benchmark_runs/segmentation_benchmark/`.
  - Colab session terminated immediately upon completion.
- **Three-Arm Seed 42 Held-Out Test Cohort Evaluation (Executed 2026-10-30)**:
  - Cohort: 469 official held-out test scans (382 tumour scans, 87 normal ovary scans, 82 multi-frame clusters + 372 singletons = 454 clusters).
  - Runtime: Executed on Google Colab NVIDIA RTX PRO 6000 Blackwell Server Edition in 146.75 seconds.
  - Verified 100% SHA-256 integrity match for all test images and masks.
  - Key Findings:
    - **Tumour DSC ($N=382$)**: Arm A $0.8209 \pm 0.2624$, Arm B $0.8179 \pm 0.2601$, Arm C $\mathbf{0.8419 \pm 0.2039}$. Arm C significantly outperforms Arm B ($\Delta_{C-B} = +0.0240$, BCa 95% CI $[+0.0060, +0.0444]$, $p = 0.013$).
    - **Lesion Preservation Ratio ($N=193$ valid overlap cohort)**: Arm A $0.4279 \pm 0.3496$, Arm B $\mathbf{0.4618 \pm 0.3854}$, Arm C $0.3876 \pm 0.3748$. Arm B significantly outperforms Arm C ($\Delta_{C-B} = -0.0742$, BCa 95% CI $[-0.1132, -0.0346]$, $p = 0.001$).
    - **Normal Ovary Specificity ($N=87$)**: Clean rejection drops severely under Arm C ($21.84\%$, 19/87) compared to Arm B ($52.87\%$, 46/87) ($\Delta_{C-B} = -0.3103$, BCa 95% CI $[-0.4379, -0.1906]$, $p = 0.0005$). Intra-ovarian FPR increases from $14.26\%$ (Arm B) to $36.17\%$ (Arm C) ($\Delta_{C-B} = +0.2191$, BCa 95% CI $[+0.1432, +0.3028]$, $p = 0.0005$). Extra-ovarian FPR remains neutral ($1.51\%$ vs $1.66\%$, $p = 0.665$).
    - **BCa Reliability**: 0% fallback to percentile intervals across all metrics; 0 empty draws rejected.
  - Deliverables: `outputs/benchmark_runs/segmentation_benchmark/seed42_test_evaluation/`.
- **Full Nine-Run Three-Seed Segmentation Benchmark (Executed & Aggregated 2026-11-01)**:
  - Architecture: SMP U-Net with ImageNet-pretrained ResNet-34 encoder ($512 \times 512$, batch size 8, AMP FP16, AdamW, ComboLoss).
  - Model Matrix: 3 Arms (Raw, Inpainted, Masked) $\times$ 3 Seeds (42, 1337, 2026) = 9 independent runs of 40 epochs each.
  - Runtime: Total GPU runtime across all 9 runs was ~3.97 hours on NVIDIA RTX PRO 6000 Blackwell Server Edition.
  - Training Summary:
    - Arm A (Raw): Seed 42 (11.78m, best ep 29), Seed 1337 (11.68m, best ep 23), Seed 2026 (11.63m, best ep 16).
    - Arm B (Inpainted): Seed 42 (57.15m, best ep 33), Seed 1337 (54.51m, best ep 29), Seed 2026 (54.37m, best ep 17).
    - Arm C (Masked): Seed 42 (12.51m, best ep 32), Seed 1337 (12.34m, best ep 40), Seed 2026 (12.36m, best ep 39).
  - Test Evaluation: Evaluated all 9 checkpoints on official held-out test cohort (382 tumour, 87 normal; 469 scans).
  - Statistical Aggregation (Scan-level 3-Seed Mean with $B=2,000$ Cluster BCa Bootstrap across 454 clusters):
    - **Tumour Delineation ($N=382$)**:
      - Arm A (Raw): DSC $0.8073 \pm 0.2490$ (median 0.9098), IoU $0.7368 \pm 0.2579$, Boundary IoU $0.0717 \pm 0.0519$.
      - Arm B (Inpainted): DSC $0.8172 \pm 0.2401$ (median 0.9117), IoU $0.7474 \pm 0.2493$, Boundary IoU $0.0724 \pm 0.0511$.
      - Arm C (Masked): DSC $0.8191 \pm 0.2168$ (median 0.9002), IoU $0.7459 \pm 0.2310$, Boundary IoU $0.0737 \pm 0.0511$.
      - Delta DSC (C vs B): $+0.0019$ (+0.19%), BCa 95% CI $[-0.0070, +0.0131]$, $p = 0.776$ (no significant difference).
      - Delta DSC (C vs A): $+0.0118$ (+1.18%), BCa 95% CI $[+0.0009, +0.0240]$, $p = 0.042$.
      - Delta DSC (B vs A): $+0.0099$ (+0.99%), BCa 95% CI $[-0.0009, +0.0215]$, $p = 0.071$.
    - **Lesion Preservation Ratio ($N=193$ valid overlap cohort, $N=189$ zero-overlap excluded without imputation)**:
      - Arm A (Raw): $0.4105 \pm 0.3253$ (median 0.3727).
      - Arm B (Inpainted): $0.4584 \pm 0.3487$ (median 0.4526).
      - Arm C (Masked): $0.3873 \pm 0.3389$ (median 0.3056).
      - Delta LPR (C vs B): $-0.0710$ (-7.10 percentage points lower lesion sensitivity under masking), BCa 95% CI $[-0.0969, -0.0460]$, $p = 0.0005$. Masking induces severe localized lesion dropout.
      - Delta LPR (B vs A): $+0.0478$ (+4.78 percentage points higher sensitivity under inpainting vs raw), BCa 95% CI $[+0.0216, +0.0763]$, $p = 0.002$.
    - **Normal Ovary Specificity ($N=87$)**:
      - Clean Rejection Rate: Arm A $57.47\%$ (50/87), Arm B $54.02\%$ (47/87), Arm C $45.98\%$ (40/87).
      - Delta Clean Rejection (C vs B): $-0.0805$ (-8.05%), BCa 95% CI $[-0.1515, -0.0152]$, $p = 0.022$.
      - Intra-Ovarian False Positive Rate: Arm A $14.14\%$, Arm B $13.23\%$, Arm C $22.38\%$.
      - Delta Intra-Ovarian FPR (C vs B): $+0.0914$ (+9.14%), BCa 95% CI $[+0.0532, +0.1382]$, $p = 0.0005$. Zero-masking creates artificial dark voids that trigger false positive cystic lesion detections on benign ovaries.
    - **Statistical Rigour**: BCa fallback frequency was 0% across all comparisons; rejected empty draws count was 0.
  - Final Deliverables: `outputs/benchmark_runs/segmentation_benchmark/final_three_seed_benchmark/` contains all 10 summary CSVs, plots, and master JSON.
- **Publication Figures Generated**:
  - `outputs/figures/synthetic_benchmark_tripartition.svg` (and `.png`)
  - `outputs/figures/calibration_reliability_diagrams.svg` (and `.png`)
  - `outputs/benchmark_runs/segmentation_benchmark/final_three_seed_benchmark/plots/three_seed_tumour_metrics_comparison.png`
  - `outputs/benchmark_runs/segmentation_benchmark/final_three_seed_benchmark/plots/three_seed_normal_ovary_comparison.png`
  - `outputs/benchmark_runs/segmentation_benchmark/final_three_seed_benchmark/plots/three_seed_bootstrap_forest_plot.png`
- **CLI Commands Verified**:
  - `ovaryvision check-env`
  - `ovaryvision init-dirs`
  - `ovaryvision validate-config configs/default.yaml`
  - `ovaryvision generate-manifest-template`
  - `ovaryvision audit-data --data-dir data/raw/mmotu/images/`
  - `ovaryvision ingest-guide`
  - `ovaryvision generate-synthetic-benchmark --num-cases 3`
  - `ovaryvision generate-splits --manifest ...`
  - `ovaryvision run-preprocessing --manifest ...`
  - `ovaryvision evaluate-calibration --val-logits-file ... --test-logits-file ...`
  - `ovaryvision evaluate-statistics --arm-c-preds ... --arm-b-preds ...`

---

## 6. Immediate Next Actions

1. Present supervisor-ready 3-seed completed segmentation benchmark report detailing the multi-seed empirical findings and clinical trade-offs.
2. Incorporate findings into academic manuscript / thesis report (LaTeX tables, forest plot, comparison violin plots).
3. User to transmit the approved formal data request letter (`docs/OVATUS_ACCESS_REQUEST.md`) to Prof. Le Thi Lan for OvaTUS-V1 surgical histopathology ground truth.
4. User to evaluate CEUS transfer segmentation experiments (`data/raw/mmotu_ceus/`).
