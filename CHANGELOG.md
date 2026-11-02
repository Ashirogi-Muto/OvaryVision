# CHANGELOG.md — Project Modification Log: OvaryVision

All notable changes to the OvaryVision codebase and research specifications are documented in this file in reverse chronological order.

---

## [0.5.0] — 2026-11-01
### Added & Completed (Full Nine-Run Three-Seed Segmentation Benchmark & Multi-Seed Statistical Aggregation)
- **G4 Blackwell Training of Seeds 1337 and 2026**:
  - Successfully trained all remaining six runs across Arms A, B, and C for Seeds 1337 and 2026 (40 epochs each) on NVIDIA RTX PRO 6000 Blackwell Server Edition GPU (`g4-sprint2`). Total GPU runtime: ~3.97 hours.
  - Arm A (Raw): Seed 1337 (11.68m, best ep 23, val loss 0.1280), Seed 2026 (11.63m, best ep 16, val loss 0.1322).
  - Arm B (Inpainted): Seed 1337 (54.51m, best ep 29, val loss 0.1297), Seed 2026 (54.37m, best ep 17, val loss 0.1451).
  - Arm C (Masked): Seed 1337 (12.34m, best ep 40, val loss 0.1405), Seed 2026 (12.36m, best ep 39, val loss 0.1306).
  - Zero test set exposure maintained during all training phases (469 held-out test scans remained strictly unaccessed).
  - Preserved all checkpoints, histories, and summaries in `outputs/benchmark_runs/segmentation_benchmark/`.
- **Held-Out Test Set Evaluation (Seeds 1337 and 2026)**:
  - Evaluated Seed 1337 (146.32s) and Seed 2026 (144.06s) best checkpoints across all 469 official held-out test scans (382 tumour, 87 normal).
  - Remote session `g4-sprint2` terminated immediately upon artifact synchronization.
- **Three-Seed Benchmark Aggregator (`scripts/aggregate_three_seed_benchmark.py`)**:
  - Implemented multi-seed scan-level aggregation and paired pseudo-patient cluster bootstrap ($B=2,000$ resamples, BCa 95% CIs) across all 454 clusters (82 multi-frame clusters + 372 singletons).
  - Handles cluster multiplicity preservation and conditional rejection for degenerate subsets.
  - Added unit test `test_aggregate_three_seed_benchmark_cli_parser` in `tests/test_segmentation_benchmark.py` (86 passing tests).
  - Exported 10 summary CSVs, publication plots, and master JSON in `outputs/benchmark_runs/segmentation_benchmark/final_three_seed_benchmark/`:
    - `nine_runs_training_summary.csv`
    - `per_seed_tumour_metrics_summary.csv`, `per_seed_normal_ovary_metrics_summary.csv`, `per_seed_bootstrap_inferential_report.csv`
    - `three_seed_pooled_tumour_summary.csv`, `three_seed_pooled_normal_summary.csv`, `three_seed_pooled_inferential_report.csv`
    - `plots/three_seed_tumour_metrics_comparison.png`, `plots/three_seed_normal_ovary_comparison.png`, `plots/three_seed_bootstrap_forest_plot.png`
    - `final_benchmark_summary.json`
- **Core Scientific Findings (3-Seed Pooled Multi-Seed Results)**:
  - Tumour Delineation ($N=382$): Arm A DSC $0.8073 \pm 0.2490$, Arm B DSC $0.8172 \pm 0.2401$, Arm C DSC $0.8191 \pm 0.2168$. Arm C vs Arm B: $\Delta \text{DSC} = +0.0019$ (+0.19%), 95% BCa CI $[-0.0070, +0.0131]$, $p = 0.776$ (no significant difference).
  - Lesion Preservation Ratio (LPR on $N=193$ valid overlap cohort): Arm A $0.4105 \pm 0.3253$, Arm B $0.4584 \pm 0.3487$, Arm C $0.3873 \pm 0.3389$. Arm C vs Arm B: $\Delta \text{LPR} = -0.0710$ (-7.10 percentage points lower in-caliper sensitivity under masking), 95% BCa CI $[-0.0969, -0.0460]$, $p = 0.0005$. Inpainting preserves significantly more lesion tissue inside caliper bounds ($\Delta \text{LPR}_{B-A} = +0.0478$, $p = 0.002$).
  - Normal Ovary Specificity ($N=87$): Clean rejection rate: Arm A $57.47\%$, Arm B $54.02\%$, Arm C $45.98\%$. Arm C vs Arm B: $\Delta = -0.0805$ (-8.05%), 95% BCa CI $[-0.1515, -0.0152]$, $p = 0.022$. Intra-ovarian FPR: Arm A $14.14\%$, Arm B $13.23\%$, Arm C $22.38\%$. Arm C vs Arm B: $\Delta = +0.0914$ (+9.14%), 95% BCa CI $[+0.0532, +0.1382]$, $p = 0.0005$.
  - Methodological Conclusion: Non-generative masking creates a clinical double-edged sword: near-identical gross tumor DSC, but substantial localized lesion dropout under calipers and elevated false positive cystic mass detections on normal ovaries.
- **Architectural Decision D31 (`DECISIONS.md`)**:
  - Documented completion of 9-run matrix, test set isolation, and pooled inferential statistics.

---

## [0.4.4] — 2026-10-30
### Added & Evaluated (Three-Arm Seed 42 Held-Out Test Cohort Evaluation & BCa Statistical Analysis)
- **Held-Out Test Set Evaluation Script (`scripts/evaluate_seed42_test_cohort.py`)**:
  - Implemented comprehensive evaluation harness for 469 official held-out test scans (382 tumour, 87 normal ovary).
  - Evaluates DSC, IoU, Boundary IoU ($d=2$), and Lesion Preservation Ratio (LPR) strictly on the $N=193$ valid overlap cohort (excluding $N=189$ zero-overlap scans without imputation).
  - Evaluates normal ovary specificity via intra-ovarian FPR, extra-ovarian FPR, and clean rejection rate ($\mathbb{I}(\text{pred} = 0)$).
  - Performs 2,000 paired pseudo-patient-cluster bootstrap resamples with BCa 95% confidence intervals and percentile fallback tracking.
  - Generates publication-ready boxplots, FPR comparisons, and forest plots for bootstrap differences.
  - Added unit test `test_evaluate_seed42_cli_parser` in `tests/test_segmentation_benchmark.py` (85 passing tests).
- **GPU Evaluation Execution (G4 Blackwell Runtime)**:
  - Executed test evaluation in 146.75s on Google Colab NVIDIA RTX PRO 6000 Blackwell Server Edition (`g4-eval`).
  - Verified 100% SHA-256 match for all 469 test images and masks prior to inference.
  - Colab session terminated immediately upon output synchronization.
- **Empirical Findings**:
  - **Tumour DSC**: Arm C ($0.8419 \pm 0.2039$) significantly outperforms Arm B ($0.8179 \pm 0.2601$) with $\Delta_{C-B} = +0.0240$, BCa 95% CI $[+0.0060, +0.0444]$, $p = 0.013$.
  - **Lesion Preservation Ratio (LPR)**: Arm B ($0.4618 \pm 0.3854$) significantly outperforms Arm C ($0.3876 \pm 0.3748$) inside occluded caliper zones with $\Delta_{C-B} = -0.0742$, BCa 95% CI $[-0.1132, -0.0346]$, $p = 0.001$.
  - **Normal Ovary Specificity**: Arm C exhibits severe clinical degradation. Clean rejection rate drops from $52.87\%$ (Arm B) to $21.84\%$ (Arm C), $\Delta_{C-B} = -0.3103$, BCa 95% CI $[-0.4379, -0.1906]$, $p = 0.0005$. Intra-ovarian FPR increases from $14.26\%$ (Arm B) to $36.17\%$ (Arm C), $\Delta_{C-B} = +0.2191$, BCa 95% CI $[+0.1432, +0.3028]$, $p = 0.0005$. Extra-ovarian FPR is neutral ($1.51\%$ vs $1.66\%$, $\Delta_{C-B} = +0.0014$, BCa 95% CI $[-0.0060, +0.0071]$, $p = 0.665$).
  - **BCa Reliability**: 0% fallback to percentile intervals; 0 rejected empty draws for LPR.
- **Artifacts Preserved**:
  - `outputs/benchmark_runs/segmentation_benchmark/seed42_test_evaluation/` contains per-scan metrics CSV, summary CSVs, full JSON summary, and 3 diagnostic plots.
- **Architectural Decision D30 (`DECISIONS.md`)**:
  - Documented complete test evaluation protocol, empirical findings, and clinical implications.

---

## [0.4.3] — 2026-10-30
### Added & Executed (Three-Arm Seed 42 Segmentation Pilot on G4 Blackwell, Collation Fix & BCa Bootstrap)
- **Three-Arm Seed 42 Segmentation Pilot Execution**:
  - Successfully executed all 40 epochs across Arms A, B, and C with Seed 42 on Google Colab NVIDIA RTX PRO 6000 Blackwell Server Edition (94.97 GB VRAM).
  - Architecture: SMP U-Net with ImageNet-pretrained ResNet-34 encoder ($512 \times 512$, batch size 8, AMP FP16).
  - Training performance:
    - Arm A (`arm_a_raw`): 40 epochs in 706.5s (11.78 min; mean 17.66s/epoch); best checkpoint selected at Epoch 29 (internal val loss `0.1352`, internal val DSC `0.8665`).
    - Arm B (`arm_b_inpainted`): 40 epochs in 3,428.8s (57.15 min; mean 85.72s/epoch, reflective of on-the-fly OpenCV Navier-Stokes inpainting during DataLoader batching); best checkpoint selected at Epoch 33 (internal val loss `0.1339`, internal val DSC `0.8648`).
    - Arm C (`arm_c_masked`): 40 epochs in 750.6s (12.51 min; mean 18.76s/epoch); best checkpoint selected at Epoch 32 (internal val loss `0.1319`, internal val DSC `0.8623`).
  - Saved and verified all checkpoints (`best_checkpoint.pth`, `latest_checkpoint.pth`), histories (`history.csv`), and summaries (`run_summary.json`) in `outputs/benchmark_runs/segmentation_benchmark/`.
  - Zero test set exposure maintained (0 of 469 held-out test scans loaded or evaluated).
  - Colab session terminated immediately upon output collection to preserve compute units.
- **DataLoader Collation Bugfix (`scripts/run_three_arm_segmentation_benchmark.py`)**:
  - Implemented `collate_seg_fn` to stack image and mask tensors into batches while keeping metadata dictionaries as clean Python lists, resolving PyTorch 2.11 / Python 3.13 string collation `TypeError`.
  - Added unit test `test_collate_seg_fn` in `tests/test_segmentation_benchmark.py` (total 84 passing tests).
- **BCa Cluster Bootstrap Implementation (`src/ovaryvision/statistics.py`)**:
  - Implemented true BCa 95% Confidence Intervals with $O(K)$ cluster jackknife acceleration $a$, mid-p rank median bias correction $z_0$, and automatic percentile fallback.
  - Added unit tests for BCa selection, constant statistics, and tied replicates.
- **Architectural Decisions D28 & D29 (`DECISIONS.md`)**:
  - Documented BCa bootstrap standard and Seed 42 pilot empirical benchmark execution.

- **NVIDIA RTX PRO 6000 Blackwell G4 Benchmarking (`scripts/smoke_test_arm_a_g4.py`)**:
  - Measured empirical throughput on G4 runtime: 37.07 ms training step latency (215.8 imgs/sec), 12.78 ms validation step latency (625.9 imgs/sec).
  - Measured peak VRAM: 2,149.8 MB (2.10 GB allocated, 2.17 GB reserved; 2.2% of 94.97 GB capacity).
  - Verified numerical stability: GradScaler scale constant at 65,536.0 with zero NaNs and zero Infs.
- **Multiplicity-Preserving Paired Cluster Bootstrap (`src/ovaryvision/statistics.py`)**:
  - Hardened `paired_cluster_bootstrap_segmentation()` to preserve multinomial cluster draw multiplicities ($m_k \ge 0$).
  - Implemented conditional rejection sampling for degenerate sub-population resamples (e.g., zero artifact-overlap scans in LPR evaluation), guaranteeing exactly $B=2,000$ valid evaluations without arbitrary constant imputation.
  - Added unit tests verifying multiplicity weighting and conditional rejection.
- **Three-Arm Benchmark Training Implementation (`scripts/run_three_arm_segmentation_benchmark.py`)**:
  - Implemented full `train_single_run` pipeline with AMP FP16, AdamW, and ComboLoss.
  - Enforced deterministic, epoch-varying paired spatial augmentations across arms within `SegmentationDataset`.
  - Guaranteed checkpoint selection strictly on internal validation loss (150 scans).
  - Guaranteed held-out test scans (469 scans) are strictly excluded from training and model selection.
  - Added run-isolated directory names (`{arm}_seed{seed}`) and robust interrupted-checkpoint resumption.
- **Documentation & Decision Governance**:
  - Appended Decision D27 to `DECISIONS.md`.
  - Updated `docs/SEGMENTATION_PRE-EXECUTION_AUDIT.md` to document G4 empirical measurements, corrected bootstrap pseudocode, and compute recommendations.

---

## [0.4.1] — 2026-10-30
### Added (Pre-Execution Audit Finalization, GPU Smoke Test & Cluster Partitioning)
- **Cluster-Preserved Segmentation Split Manifest (`data/manifests/mmotu_2d_segmentation_split.csv`)**:
  - Partitioned the 1,000 official training images (`train.txt`) into exactly 850 `internal_train` and 150 `internal_val` using whole dHash pseudo-patient clusters with zero leakage.
  - Held out the 469 official test scans (`val.txt`) completely untouched with 100% SHA-256 match.
- **Two-Epoch GPU Smoke Test Execution (`scripts/smoke_test_arm_a.py`)**:
  - Successfully executed a 2-epoch training smoke test on a Tesla T4 GPU (Google Colab runtime) on Arm A using SMP U-Net with ResNet-34 encoder at $512 \times 512$ resolution.
  - Recorded actual metrics: runtime (5.66s ep1, 5.29s ep2; 12.23s total), peak VRAM (4.752 GB), and monotonic loss convergence ($1.0869 \to 0.8865 \to 0.8010$).
- **CDFI Manifest Reconciliation**:
  - Reconciled the draft summary count discrepancy (57 vs 44) by diagnosing confusion with the 1,202 tumour scans.
  - Froze `data/manifests/mmotu_2d_cdfi_classification.csv` (207 High CDFI, 16 Probable, 57 Ambiguous, 1,189 High B-mode) under SHA-256 `487f376afc8f3ac5dd32f2a82453cdaf8659d1de52060f342ba1cf1fee4f79ff`.
- **Architectural Decisions D25 & D26 (`DECISIONS.md`)**:
  - Recorded pre-execution audit gates, CDFI stratification, cluster bootstrap inferential standard, and smoke test verification.

---

## [0.4.0] — 2026-10-30
### Added (MMOTU Completeness & CEUS Modality Recovery Audit)
- **MMOTU Completeness Audit Report (`docs/MMOTU_DATASET_COMPLETENESS_AUDIT.md`)**:
  - Full forensic audit comparing 6 releases (arXiv paper, official Google Drive, Zenodo 21128657, Zenodo 20541610 tombstone, Figshare 25115633, and Figshare 25058690).
  - Proved that the 2D B-mode cohort (1,469 scans, 53 malignant) in `data/raw/mmotu/` is complete, unpruned, and authentic, whereas Figshare derivatives dropped all 267 Normal Ovary scans (1,202 scans remaining).
  - Proved that clinician artifact masks ($M_{art}$) do not exist in any public release; all symbols were burned into scan pixels during clinical exams.
  - Resolved the origin of the 2,938 local tumor masks: 1,469 unique binary masks and 1,469 byte-for-byte duplicate `<id>_binary_binary.PNG` files inherited from upstream Google Drive batch naming.
  - Documented that `OTU_2d` contains 216 Color Doppler Flow Images (CDFI) alongside 1,253 traditional B-mode scans as an unexplored clinical confounder.
- **Contrast-Enhanced Ultrasound (CEUS) Recovery & Staging (`data/raw/mmotu_ceus/`)**:
  - Safely staged 170 CEUS scans across 47 patients (9 malignant, 161 benign; zero Normal Ovaries) without touching the 2D benchmark.
  - Staged 170 binary tumor masks (`tumor_masks/`), 170 RGB multi-class palette masks (`original_masks/`), and official train/val labels (`labels/`).
  - Created machine-readable master manifest: `data/manifests/mmotu_ceus_master_manifest.csv` with perceptual dHash clustering (27 clusters), SHA-256 hashes, binary targets, and caliper flags (79 of 170 scans, 46.5%).
- **Release Comparison Table (`outputs/metrics/mmotu_release_comparison.csv`)**:
  - Machine-readable manifest across all public original and derivative releases.
- **Architectural Decision D24 (`DECISIONS.md`)**:
  - Appended Decision D24 formally accepting audit findings and defining 3 additional defensible research experiments: (1) 2D $\to$ CEUS Out-of-Domain Transfer, (2) CDFI Sub-Modality Ablation, and (3) Full 8-Class Morphological Classification.

---

## [0.3.0] — 2026-10-29
### Added (Milestones 2 & 3 Core Analytical, Preprocessing, Calibration & GPU Infrastructure)
- **Group-Stratified Split Engine (`src/ovaryvision/splits.py`)**:
  - Deterministic group-stratified train/val/test splitting (70/15/15) by `pseudo_patient_cluster` (with individual fallback) preserving class balance.
  - Two-tier leakage verification (`validate_manifest_leakage`) guaranteeing zero cross-role frame or cluster leakage.
  - Multi-seed split manifest export (`mmotu_split_seed*.csv`).
- **Three-Arm Preprocessing Engine (`src/ovaryvision/preprocessing.py`)**:
  - Arm A (Raw): Unmodified pass-through baseline.
  - Arm B (Inpainted): Strict exterior conservation ($I_B = I_{inpaint} \odot M_{art} + I_A \odot (1 - M_{art})$) with 2px cross morphological dilation. Pure CPU harmonic boundary diffusion (Laplace Jacobi relaxation) and OpenCV/deep inpainter hooks.
  - Arm C (Masked): Strict zero/neutral occlusion ($I_C = c \cdot M_{art} + I_A \cdot (1 - M_{art})$ with $c=0$) with 2px cross morphological dilation.
  - Batch runner script: `scripts/run_preprocessing.py`.
- **Probability Calibration Engine (`src/ovaryvision/calibration.py`)**:
  - Validation-only Temperature Scaling ($T^*$) optimized via negative log-likelihood minimization on validation logits.
  - Expected Calibration Error (ECE; uniform and quantile binning), Maximum Calibration Error (MCE), and Brier score.
- **Statistical Comparison & Hypothesis Testing Engine (`src/ovaryvision/statistics.py`)**:
  - Paired DeLong test for AUC differences between models on identical evaluation cohorts.
  - Pre-registered non-inferiority testing against margin $\delta = -0.02$ with sensitivity bounds $\{-0.01, -0.02, -0.03\}$.
  - Stratified paired bootstrap confidence intervals (1,000+ resamples; BCa and percentile).
  - McNemar's test for paired classification decision discordance with Edwards continuity correction.
- **GPU Training & Evaluation Pipeline (`scripts/gpu_train.py`)**:
  - Standalone, self-contained PyTorch training pipeline for Google Colab and Kaggle GPU runtimes.
  - ConvNeXt-Tiny and EfficientNet-B0 with ImageNet-1K pre-trained weights.
  - Class-Weighted Focal Loss ($\gamma = 2.0$), AdamW with Cosine Annealing, validation early stopping, post-hoc calibration, and prediction export.
- **Controlled Synthetic Benchmark Evaluator & Reporting (`scripts/evaluate_synthetic_benchmark.py`, `scripts/generate_figures.py`, `scripts/generate_tables.py`)**:
  - Empirical tri-partition evaluation comparing Arm A (Raw), Arm B (Inpainted), and Arm C (Masked) on synthetic ultrasound cases.
  - Generated vector SVG figures: `outputs/figures/synthetic_benchmark_tripartition.svg` and `outputs/figures/calibration_reliability_diagrams.svg`.
  - Generated academic LaTeX table: `outputs/figures/table1_synthetic_integrity.tex`.
- **CLI Subcommands**:
  - Added `generate-splits`, `run-preprocessing`, `evaluate-calibration`, and `evaluate-statistics`.
- **Automated Test Suite**:
  - Added `tests/test_splits.py` (6 tests).
  - Added `tests/test_preprocessing.py` (6 tests).
  - Added `tests/test_calibration.py` (5 tests).
  - Added `tests/test_statistics.py` (7 tests).
  - Expanded CLI tests in `tests/test_cli.py` (4 new tests).
  - Test suite now comprises **64 automated tests with 100% pass rate** in 1.62s.
- **Decisions**:
  - Added Decisions D14, D15, D16, and D17 to `DECISIONS.md`.

---

## [0.2.0] — 2026-10-29
### Added (Milestone 2 Controlled Synthetic Benchmark & Ingestion Preparation)
- **Synthetic Caliper Benchmark Module (`src/ovaryvision/synthetic_artifacts.py`)**:
  - Deterministic synthetic ultrasound annotation overlay generator (measurement calipers, crosshairs, dashed connecting lines, callout text).
  - Exact mask containment guarantee: strictly ensures $I_{clean} \neq I_{corrupted} \implies M_{art} = 1$.
  - Tri-partition image integrity evaluation engine: Global, In-Mask ($M_{art} = 1$), and Exterior conservation ($M_{art} = 0$) reporting for MAE, PSNR, SSIM, Local Texture Variance ($\Delta \text{LTV}$), and optional lazy LPIPS.
  - Export engine serializing clean reference, corrupted image, binary artifact mask, and metadata JSON with array SHA-256 digests.
- **MMOTU Ingestion Tooling & Guide**:
  - `scripts/ingest_mmotu.py`: Decompression and post-ingestion audit helper with explicit `--confirm` safety gate and path traversal validation.
  - `docs/MMOTU_INGESTION_GUIDE.md`: Comprehensive SOP for MMOTU acquisition from Zenodo (21128657) and Figshare (25058690), license verification, local placement, and audit execution.
- **CLI Subcommands**:
  - Added `generate-synthetic-benchmark` to generate reproducible synthetic cases and export metrics JSON.
  - Added `ingest-guide` displaying authoritative repository links and safety constraints.
- **Automated Test Suite**:
  - Added `tests/test_synthetic_artifacts.py` with 10 unit tests.
  - Added CLI tests in `tests/test_cli.py` for synthetic benchmark generation and ingestion guide.
  - Full test suite expanded to 36 tests (100% pass rate).
- **Architecture & Research Protocol Revisions**:
  - Documented synthetic benchmark design and limits in `SPEC.md`, `docs/RESEARCH_PROTOCOL.md`, and `docs/DATA_DICTIONARY.md`.
  - Added Decisions D12 and D13 to `DECISIONS.md`.

---

## [0.1.1] — 2026-10-29
### Changed & Audited (Milestone 1 Scientific & Implementation Audit)
- **Memory Consolidation**: Merged `AGENTS.md` and `AGENT.md` into a single canonical `/home/ubuntu/major/AGENT.md` complying strictly with `GEMINI.md` naming rules; deleted redundant `AGENTS.md`.
- **OvaTUS-V1 Label Taxonomy Correction**: Proved via SigM Lab (HUST / NHOG) authoritative records that OvaTUS-V1 contains six IOTA morphological categories (`Solid tumor`, `Multilocular cyst`, `Unilocular cyst`, `Dermoid cyst`, `Multilocular-solid cyst`, `Unilocular-solid cyst`), not confirmed benign/malignant histopathology. Removed speculative 6-class mapping and formally blocked binary evaluation on OvaTUS-V1 pending histological verification (BLK-03, Decision D09).
- **Leakage Gate Hardening**: Upgraded `validate_manifest_leakage` in `src/ovaryvision/manifests.py` to enforce two separate gates: direct image-level uniqueness (raising error on duplicate image allocation) and pseudo-patient cluster exclusivity. Added unit tests verifying both gates (Decision D11).
- **Patient Leakage Scientific Boundaries**: Clarified in `SPEC.md`, `AGENT.md`, and `docs/RESEARCH_PROTOCOL.md` that dHash near-duplicate clustering mitigates identical frame leakage but cannot prove patient-level independence in public MMOTU without hospital record linkage (Decision D10).
- **MMOTU Caliper Mask Dependencies**: Formally recorded that Zhao et al. (MMOTU) public release provides only tumor ROI masks, and Chen et al. (MGGAN) caliper artifact masks are unreleased, defining this as a mandatory prerequisite before running Arms B and C (BLK-02).
- **Statistical Plan Refinement**: Labeled $\delta = -0.02$ AUC non-inferiority margin as provisional, mandated sensitivity analysis across $\delta \in \{-0.01, -0.02, -0.03\}$, and established 1,000-sample bootstrap paired differences as the primary estimator.
- **Audit Deliverable**: Published complete findings and Go/No-Go recommendations in `docs/MILESTONE_1_AUDIT.md`.

---

## [0.1.0] — 2026-10-29
### Added
- **Canonical Specification**: Completely rewrote `SPEC.md` to v2.0.0 detailing research framing, clinical motivation, 5 core research questions, dataset contracts (MMOTU and OvaTUS-V1), 3-arm preprocessing protocols (Raw, Inpainted, Masked), split design, standardized CNN backbones (ConvNeXt, EfficientNet), probability calibration (ECE/MCE), tri-partition image-integrity metrics, statistical analysis plan (DeLong, bootstrap CIs, non-inferiority margin $\delta = -0.02$), and risk matrix.
- **Persistent Project Memory**:
  - `AGENTS.md`: Mandatory reading order, coding conventions, research-integrity rules, and environment boundaries.
  - `PROJECT_STATE.md`: Real-time tracking of implemented modules, dataset status, blockers (BLK-01 to BLK-04), and next actions.
  - `DECISIONS.md`: Initialized append-only architectural decision log with decisions D01 through D08.
  - `IMPLEMENTATION_PLAN.md`: Dependency-aware 5-milestone roadmap from repository setup to manuscript-ready outputs.
- **Operational Documentation**:
  - `docs/RESEARCH_PROTOCOL.md`: Detailed operational protocol for 3-arm experiments, calibration, and metrics.
  - `docs/DATA_DICTIONARY.md`: Complete JSON and CSV schemas for manifests, audits, records, and predictions.
  - `docs/EXPERIMENT_TRACKER.md`: Experiment registry schema and reproducible run tracking format.
  - `docs/architecture.md`: Architecture diagrams and OCI/GPU boundary documentation.
  - `docs/data_protocol.md`: Data ingestion protocol, artifact vs tumor mask contracts, and privacy protocols.
- **Packaging & Configuration**:
  - `pyproject.toml`: Modern PEP 621 package metadata with `ovaryvision` console script entry point.
  - `configs/default.yaml`: Base configuration with explicit parameters for datasets, arms, models, optimization, and metrics.
  - `configs/experiment.yaml`: Benchmark experiment configuration.
  - `.env.example`: Safe environment configuration template.
- **Source Modules (`src/ovaryvision/`)**:
  - `logging_utils.py`: Rotating file logger adhering to the 10MB x 5 file rotation rule in `/home/ubuntu/major/logs/`.
  - `environment.py`: System diagnostic scanner checking CPU, RAM, disk, OS, CUDA/GPU, and package dependencies.
  - `config.py`: Pydantic v2 typed configuration loading and validation from YAML and environment variables.
  - `manifests.py`: Pydantic schemas and CSV/JSON IO for ImageManifest, SplitManifest, and PredictionManifest, strictly separating tumor masks from artifact masks.
  - `provenance.py`: Deterministic SHA-256 execution provenance recording environment, config, git, and execution metadata.
  - `data_audit.py`: Image inventory and auditing engine featuring recursive search, dimensions/channels extraction, SHA-256 hashing, perceptual dHash near-duplicate clustering, label cross-checking, and JSON/CSV reporting with graceful error handling.
  - `cli.py`: Click CLI providing `check-env`, `init-dirs`, `validate-config`, `audit-data`, and `generate-manifest-template`.
- **Scripts**:
  - `scripts/run_audit.py`: Direct script runner for data auditing.
- **Automated Test Suite (`tests/`)**:
  - Unit tests for configuration, environment diagnostics, manifests, provenance, data auditing (with synthetic images and duplicate checks), and CLI subcommands.

---

## [0.0.1] — 2026-08-27
### Added
- Initial project registration in global agent rules.
- Draft `SPEC.md` and initial `AGENT.md`.
- Vector SVG and PNG architecture diagrams (`data_pipeline.svg`, `data_pipeline.png`).
- Phase-I academic report documents (`Major_Phase1_Report_Filled(ref).docx`).

