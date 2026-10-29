# CHANGELOG.md — Project Modification Log: OvaryVision

All notable changes to the OvaryVision codebase and research specifications are documented in this file in reverse chronological order.

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

