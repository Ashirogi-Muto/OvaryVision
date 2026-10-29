# PROJECT_STATE.md — Continuous Project State Tracker: OvaryVision

**Current Date / Timestamp**: 2026-10-29  
**Current Phase**: Full 18-Run Benchmark & Forensic Scientific Audit Complete
**Current Status**: Complete on `exp/full-cohort-benchmark` (Ready for User Review)
**Canonical Specification**: `SPEC.md` (v2.0.0)
**Audit Documents**: `docs/MILESTONE_1_AUDIT.md`, `docs/FULL_COHORT_AUDIT_REPORT.md`

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
| `DECISIONS.md` | Decision Log | Implemented | Append-only architectural & methodology log (D01–D23) |
| `IMPLEMENTATION_PLAN.md` | Roadmap | Implemented | Dependency-aware milestones and task tracking |
| `CHANGELOG.md` | History | Implemented | Foundation, Milestone 2, and Core Pipeline updates recorded |
| `README.md` | Documentation | Implemented | Project entry point, CLI guide, and architecture |
| `docs/RESEARCH_PROTOCOL.md` | Research Protocol | Implemented | Detailed 3-arm protocol, calibration & metrics |
| `docs/FULL_COHORT_BENCHMARK_RESULTS.md` | Results Report | Implemented | 18-run benchmark report with macro-averages and metrics |
| `docs/FULL_COHORT_AUDIT_REPORT.md` | Audit Report | Implemented | Forensic audit of cohort, power, non-inferiority, mask QC |
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
| `src/ovaryvision/statistics.py` | Core Module | Implemented | Paired DeLong test, bootstrap CIs, non-inferiority, McNemar |
| `src/ovaryvision/cli.py` | CLI Interface | Implemented | Click CLI with 11 core subcommands |
| `scripts/run_audit.py` | Automation | Implemented | Quick script runner for data audits |
| `scripts/ingest_mmotu.py` | Automation | Implemented | Safe, opt-in extraction and post-ingestion audit helper |
| `scripts/run_preprocessing.py` | Automation | Implemented | Batch preprocessing across image manifests |
| `scripts/gpu_train.py` | Automation | Implemented | Standalone PyTorch GPU training script for Colab/Kaggle |
| `scripts/run_artifact_annotation_pilot.py` | Automation | Implemented | 150-image real caliper mask extraction & QC pilot |
| `scripts/evaluate_synthetic_benchmark.py` | Automation | Implemented | Tri-partition comparative benchmark evaluator |
| `scripts/generate_figures.py` | Visualization | Implemented | Vector SVG/PNG figures for synthetic benchmark & calibration |
| `scripts/generate_tables.py` | Reporting | Implemented | LaTeX manuscript table generator for tri-partition integrity |
| `tests/` | Quality Gate | Implemented | Automated pytest suite (**69 tests passing in 1.65s**) |

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

- **Automated Tests**: Tested via `pytest tests/` (**69 of 69 tests passing** across 9 test suites in 1.65s):
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
  - `tests/test_statistics.py` (7 tests)
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
- **Publication Figures Generated**:
  - `outputs/figures/synthetic_benchmark_tripartition.svg` (and `.png`)
  - `outputs/figures/calibration_reliability_diagrams.svg` (and `.png`)
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

1. Scale researcher-annotated morphological mask generation across the remaining 1,319 MMOTU images, prioritizing the 53 malignant cases (files `1417`–`1469`) to enable full-cohort paired DeLong testing and sensitivity evaluation.
2. Complete full 3-arm training on scaled datasets under Arm B (Harmonic Inpainting) and Arm C (Zero-Masking) across all 3 seeds and both backbones.
3. Compute formal non-inferiority hypothesis tests ($\delta = -0.02$) between Arm C and Arm B on the full evaluation test set once positive cases are integrated.
4. Finalize manuscript figures and tables for November 2–3, 2026 milestone review.
5. User to manually transmit the approved formal data request letter (`docs/OVATUS_ACCESS_REQUEST.md`) to Prof. Le Thi Lan for OvaTUS-V1 surgical histopathology ground truth.
