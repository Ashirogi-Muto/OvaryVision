# IMPLEMENTATION_PLAN.md — Master Dependency-Aware Implementation Plan: OvaryVision

This document outlines the end-to-end execution roadmap for the OvaryVision research benchmark, from repository initialization through manuscript-ready experimental tables and figures.

---

## Milestone 1: Research Specification, Persistent Memory & Executable Infrastructure
**Goal**: Establish complete project documentation, research contracts, packaging, logging, configuration, diagnostics, manifests, and data audit tools with 100% automated test coverage.  
**Target Platform**: OCI CPU-only Instance  
**Status**: In Progress / Near-Completion  

### Task 1.1: Canonical Research Specification (`SPEC.md`)
- **Goal**: Rewrite `SPEC.md` into an operational, mathematically precise specification detailing research framing, dataset contracts, 3-arm protocols, modeling, calibration, image-integrity metrics, statistical analysis plan, and risks.
- **Dependencies**: None.
- **Expected Files**: `/home/ubuntu/major/SPEC.md`
- **Acceptance Criteria**: Covers all 10 core sections; avoids unsupported claims; defines clear mathematical formulations for ECE, LTV, Focal Loss, and non-inferiority margins.
- **Validation Procedure**: Manual review against project prompt requirements.
- **Status**: **Complete**

### Task 1.2: Persistent Operating Documents & Memory Architecture
- **Goal**: Create durable operating documents so any future agent can continue seamlessly without conversational context.
- **Dependencies**: Task 1.1
- **Expected Files**: `AGENT.md`, `PROJECT_STATE.md`, `DECISIONS.md`, `IMPLEMENTATION_PLAN.md`, `CHANGELOG.md`, `README.md`
- **Acceptance Criteria**: Clear reading order, research integrity rules, memory boundary rules, append-only decision log, live state tracker.
- **Validation Procedure**: Cross-file consistency inspection.
- **Status**: **Complete**

### Task 1.3: Operational Research Protocols & Data Dictionaries
- **Goal**: Provide formal technical documentation for experiment protocols, schema contracts, experiment registries, and architecture.
- **Dependencies**: Task 1.1, Task 1.2
- **Expected Files**:
  - `docs/RESEARCH_PROTOCOL.md`
  - `docs/DATA_DICTIONARY.md`
  - `docs/EXPERIMENT_TRACKER.md`
  - `docs/architecture.md`
  - `docs/data_protocol.md`
- **Acceptance Criteria**: Strict schema definitions for manifests, audits, runs, and metrics; clear distinction between tumor ROI masks and artifact masks; external label compatibility contract.
- **Validation Procedure**: Markdown linter / cross-reference audit.
- **Status**: **In Progress**

### Task 1.4: Repository Setup, Configuration & Packaging
- **Goal**: Set up standard `src/` layout with `pyproject.toml`, `.env.example`, directory hierarchy, and Pydantic v2 typed configuration parser.
- **Dependencies**: Task 1.2
- **Expected Files**:
  - `pyproject.toml`
  - `.env.example`
  - `configs/default.yaml`
  - `configs/experiment.yaml`
  - `src/ovaryvision/__init__.py`
  - `src/ovaryvision/config.py`
  - `src/ovaryvision/logging_utils.py`
  - `src/ovaryvision/environment.py`
- **Acceptance Criteria**: Validates YAML configs against schema; enforces 10MB x 5 rotating log files in `logs/`; detects host hardware (ARM64 CPU-only OCI instance).
- **Validation Procedure**: Unit tests in `tests/test_config.py` and `tests/test_environment.py`.
- **Status**: **Complete**

### Task 1.5: Data Manifest & Provenance Engines
- **Goal**: Implement typed data models and serializers for image manifests, split manifests, and reproducible run provenance hashing.
- **Dependencies**: Task 1.4
- **Expected Files**:
  - `src/ovaryvision/manifests.py`
  - `src/ovaryvision/provenance.py`
- **Acceptance Criteria**: Strict validation of image paths, hashes, labels, and separate artifact vs tumor mask paths; deterministic SHA-256 provenance hashing of environment + config + git state.
- **Validation Procedure**: Unit tests in `tests/test_manifests.py` and `tests/test_provenance.py`.
- **Status**: **Complete**

### Task 1.6: Data Auditing & Near-Duplicate Clustering Tool
- **Goal**: Build a robust image inventory and auditing engine capable of recursively discovering images, calculating SHA-256 checksums, computing difference hashes (dHash) for near-duplicate clustering, checking label/mask integrity, and generating audit reports.
- **Dependencies**: Task 1.5
- **Expected Files**:
  - `src/ovaryvision/data_audit.py`
  - `scripts/run_audit.py`
- **Acceptance Criteria**: Detects exact duplicates and near-duplicates; identifies absent patient IDs; fails gracefully with actionable guidance if dataset path is missing; exports JSON/CSV audit reports.
- **Validation Procedure**: Unit tests in `tests/test_data_audit.py` covering synthetic images, duplicate detection, and missing directories.
- **Status**: **Complete**

### Task 1.7: Unified CLI & Test Suite
- **Goal**: Provide a clean command-line interface (`ovaryvision`) exposing `check-env`, `init-dirs`, `validate-config`, `audit-data`, `generate-manifest-template`, `generate-synthetic-benchmark`, and `ingest-guide`, verified by an automated test suite.
- **Dependencies**: Tasks 1.4–1.6
- **Expected Files**:
  - `src/ovaryvision/cli.py`
  - `tests/test_cli.py`
- **Acceptance Criteria**: 100% test pass rate with pytest; zero non-deterministic failures.
- **Validation Procedure**: `pytest tests/ -v`.
- **Status**: **Complete**

---

## Milestone 2: Controlled Synthetic-Artifact Benchmark & Dataset Ingestion Readiness
**Goal**: Implement the deterministic synthetic-artifact benchmark with tri-partition metrics (Global, In-Mask, Exterior, LTV), verify dataset ingestion readiness, and establish authoritative MMOTU acquisition procedures.  
**Target Platform**: OCI CPU Orchestrator  
**Status**: **In Progress (Synthetic Benchmark Complete; Ingestion Tooling Ready)**  

### Task 2.1: MMOTU Ingestion Tooling & Forensic Data Audit Workflow
- **Goal**: Implement opt-in ingestion tooling, safety confirmation gates, and authoritative acquisition documentation for MMOTU. Ingest 1,469 B-mode images, 1,469 tumor segmentation masks, and official label files.
- **Dependencies**: Milestone 1.
- **Expected Files**: `scripts/ingest_mmotu.py`, `docs/MMOTU_INGESTION_GUIDE.md`, `data/manifests/mmotu_master_manifest.csv`, `docs/DATA_READINESS_REPORT.md`
- **Acceptance Criteria**: Ingested 1,469 images without corruption; documented 8-class distribution; identified 136 near-duplicate clusters ($d_H \le 4$); generated master manifest.
- **Status**: **Complete**.

### Task 2.2: Annotation Artifact Mask Procurement & Verification
- **Goal**: Procure verified annotation artifact masks (from Chen et al. MGGAN author release) or apply an audited caliper detection pipeline. Verify that zero tumor ROI masks are mislabeled as artifact masks.
- **Dependencies**: Task 2.1.
- **Expected Files**: `data/interim/artifact_masks/`, `docs/artifact_mask_audit.md`
- **Status**: **Blocked (BLK-02)**.

### Task 2.3: Perceptual Near-Duplicate Clustering & Split Manifest Generation
- **Goal**: Cluster near-duplicate ultrasound sweep frames using dHash/pHash ($d_H \le 4$), assign pseudo-patient cluster IDs, and generate leak-resistant 70/15/15 Group-Stratified splits across seeds 42, 1337, 2026.
- **Dependencies**: Task 2.1, Task 2.2.
- **Expected Files**: `src/ovaryvision/splits.py`, `tests/test_splits.py`, `data/manifests/mmotu_split_seed42.csv`, `data/manifests/mmotu_split_seed1337.csv`, `data/manifests/mmotu_split_seed2026.csv`
- **Acceptance Criteria**: 136 near-duplicate clusters atomically assigned to single folds; exact 70/15/15 proportions; 3.6% positive class balance preserved; zero frame leakage across partitions.
- **Status**: **Complete**.

### Task 2.4: Controlled Synthetic Caliper Insertion Benchmark
- **Goal**: Implement digital caliper injection on unannotated scans to provide an absolute ground-truth reference for image-integrity evaluation (SSIM, PSNR, MAE, Local Texture Variance $\Delta \text{LTV}$, and lazy LPIPS).
- **Dependencies**: Milestone 1.
- **Expected Files**: `src/ovaryvision/synthetic_artifacts.py`, `tests/test_synthetic_artifacts.py`, `data/interim/synthetic_benchmark/`, `scripts/evaluate_synthetic_benchmark.py`
- **Acceptance Criteria**: Deterministic generation under fixed seed; exact mask containment ($I_{clean} \neq I_{corrupted} \implies M_{art} = 1$); tri-partition metric reporting; 100% automated test coverage.
- **Validation Procedure**: `pytest tests/test_synthetic_artifacts.py -v`.
- **Status**: **Complete**

---

## Milestone 3: 3-Arm Preprocessing Pipeline & GPU Execution Scripts
**Goal**: Implement the three preprocessing transformations (Raw, Inpainted, Zero-Masked), probability calibration engine, and self-contained GPU training scripts for Colab/Kaggle execution.  
**Target Platform**: External GPU (Colab/Kaggle) with local OCI orchestration  
**Status**: **Complete**  

### Task 3.1: Preprocessing Pipeline (Arms A, B, C)
- **Goal**: Implement deterministic transforms for:
  - Arm A: Pass-through raw image.
  - Arm B: Harmonic / verified inpainting engine with in-mask restriction.
  - Arm C: Non-generative zero-mask occlusion ($c = 0$).
- **Dependencies**: Milestone 2.
- **Expected Files**: `src/ovaryvision/preprocessing.py`, `scripts/run_preprocessing.py`, `tests/test_preprocessing.py`
- **Status**: **Complete**.

### Task 3.2: Self-Contained GPU Training Pipeline (`scripts/gpu_train.py`)
- **Goal**: Develop GPU training script supporting ConvNeXt-Tiny and EfficientNet-B0 with Class-Weighted Focal Loss, AdamW, Cosine Annealing, validation early stopping, post-hoc calibration, and prediction export.
- **Dependencies**: Task 3.1.
- **Expected Files**: `scripts/gpu_train.py`
- **Status**: **Complete**.

### Task 3.3: Post-Hoc Temperature Scaling & Calibration Engine
- **Goal**: Implement Temperature Scaling optimization on validation logits only, calculating ECE, MCE, and Brier score.
- **Dependencies**: Task 3.2.
- **Expected Files**: `src/ovaryvision/calibration.py`, `tests/test_calibration.py`
- **Status**: **Complete**.

---

## Milestone 4: Model Training & Evaluation Across All Three Arms
**Goal**: Execute training across Arms A, B, and C over 3 seeds on GPU runtime; synchronize predictions and metrics back to OCI workspace.  
**Target Platform**: External GPU  
**Status**: Orchestration & Ingestion Ready (Awaiting GPU Training Execution)  

### Task 4.1: Internal Benchmark Training & Validation (MMOTU)
- **Goal**: Train 18 total model runs (3 Arms $\times$ 2 Backbones $\times$ 3 Seeds); evaluate internal test sets.
- **Dependencies**: Milestone 3.
- **Expected Files**: `outputs/metrics/mmotu_internal_results.csv`, `outputs/checkpoints/`
- **Status**: **Tooling Ready (`scripts/gpu_train.py`); Awaiting GPU Execution Run**.

### Task 4.2: External Benchmark Evaluation (OvaTUS-V1)
- **Goal**: Ingest OvaTUS-V1 post-approval, verify label compatibility mapping, and evaluate all 18 trained models without fine-tuning.
- **Dependencies**: Task 4.1, OvaTUS-V1 access approval (**BLK-03**).
- **Expected Files**: `outputs/metrics/ovatus_external_results.csv`
- **Status**: **Blocked (BLK-03)**.

---

## Milestone 5: Statistical Analysis, Image Integrity & Publication Artifacts
**Goal**: Perform paired statistical testing (DeLong, McNemar, BCa bootstrap), evaluate tri-partition image integrity (LTV, exterior PSNR, synthetic SSIM/LPIPS), and generate publication-quality figures and tables.  
**Target Platform**: OCI CPU Orchestrator  
**Status**: **Analytical Engine & Artifact Generators Complete**  

### Task 5.1: Paired Statistical Comparison & Hypothesis Testing
- **Goal**: Compute $\Delta \text{AUC}_{C-B}$ and 95% bootstrap confidence intervals against non-inferiority margin $\delta = -0.02$.
- **Dependencies**: Milestone 4.
- **Expected Files**: `src/ovaryvision/statistics.py`, `tests/test_statistics.py`
- **Status**: **Complete**.

### Task 5.2: Tri-Partition Image Integrity Evaluation
- **Goal**: Compute exterior tissue conservation (PSNR/MAE), in-mask texture variance shift ($\Delta \text{LTV}$), and synthetic caliper reconstruction metrics.
- **Dependencies**: Milestone 3, Milestone 4.
- **Expected Files**: `scripts/evaluate_synthetic_benchmark.py`, `outputs/metrics/synthetic_benchmark_comparative_table.csv`
- **Status**: **Complete**.

### Task 5.3: Publication-Grade Visualizations & Manuscript Tables
- **Goal**: Generate ROC curves, reliability diagrams, inpainting artifact case studies, and LaTeX/CSV summary tables.
- **Dependencies**: Task 5.1, Task 5.2.
- **Expected Files**: `scripts/generate_figures.py`, `scripts/generate_tables.py`, `outputs/figures/synthetic_benchmark_tripartition.svg`, `outputs/figures/calibration_reliability_diagrams.svg`, `outputs/figures/table1_synthetic_integrity.tex`
- **Status**: **Complete**.

