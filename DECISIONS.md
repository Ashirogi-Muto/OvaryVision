# DECISIONS.md — Architecture & Research Methodology Decision Log: OvaryVision

This document is an **append-only** log of all fundamental architectural, scientific, and implementation decisions made in the OvaryVision project. Earlier decisions must not be rewritten to retroactively justify new assumptions; subsequent refinements are appended as new decision records.

---

## Decision D01: Strategic Pivot from 10-Component Overengineering to Data-Centric Preprocessing Benchmark
- **Date**: 2026-08-27
- **Status**: Accepted & Canonical
- **Decision**: Pivot the project scope away from a saturated 10-component classification/segmentation pipeline on MMOTU towards a rigorous data-centric benchmark comparing Generative Inpainting (Arm B) against Non-Generative Masking (Arm C) and Raw Scans (Arm A).
- **Rationale**:
  - Literature audit of 2024–2026 publications revealed that standard CNN architectures on MMOTU report saturated metrics (AUC > 0.98, Accuracy > 91.9%), driven primarily by patient frame leakage and measurement caliper shortcut learning.
  - Adding complex attention or multi-task modules provides zero translational clinical value if networks merely exploit caliper shortcuts.
  - Generative inpainting inside diagnostic lesions creates severe safety hazards by hallucinating fake solid components or smoothing genuine abnormalities.
- **Alternatives Considered**:
  - *Alternative 1*: Implement the full 10-component pipeline (ConvNeXt + Swin + MedSAM + Focal Loss + Grad-CAM). Rejected as technically saturated and scientifically incremental.
  - *Alternative 2*: 3-Class Benign vs Borderline vs Malignant classification. Rejected due to sample scarcity (borderline cases represent < 5% of dataset, providing inadequate statistical power for reliable cross-validation).
- **Evidence / Assumptions**:
  - Supported by Zhao et al. (MMOTU 2022), Wang et al. (MGGAN 2023), and 2025/2026 comparative literature audits.
- **Consequences**: Project focus shifts strictly to data-centric integrity, rigorous probability calibration, and external cross-center validation.

---

## Decision D02: Strict Enforcement of Standardized Backbones over Proprietary Model Tuning
- **Date**: 2026-08-27 (Refined 2026-10-29)
- **Status**: Accepted
- **Decision**: Avoid proprietary or hyper-customized network architectures; benchmark standardized, representative convolutional architectures (`ConvNeXt` and `EfficientNet`) across all preprocessing arms under identical training budgets.
- **Rationale**: Ensures that observed performance differences are strictly attributable to the data preprocessing arms (raw vs inpainted vs masked) rather than model-specific inductive biases or proprietary tuning.
- **Alternatives Considered**:
  - Committing exclusively to ConvNeXt-Small. Rejected to prevent single-architecture bias.
  - Using Vision Transformers (ViT/Swin). Rejected as primary due to extreme data appetite and propensity to overfit on small ultrasound sample sizes (<2,000 images).
- **Evidence / Assumptions**: Standard CNNs exhibit stronger spatial inductive locality biases on small ultrasound datasets.
- **Consequences**: Training and evaluation code must be parameterized by architecture family (`convnext_tiny`, `efficientnet_b0`).

---

## Decision D03: Strict Separation between Tumor Segmentation Masks and Annotation Artifact Masks
- **Date**: 2026-10-29
- **Status**: Accepted
- **Decision**: Enforce a strict architectural, manifest, and data model separation between **Tumor ROI Segmentation Masks** ($M_{roi}$) and **Clinician Annotation Artifact Masks** ($M_{art}$).
- **Rationale**:
  - Tumor segmentation masks outline the pathological lesion boundary.
  - Annotation artifact masks outline synthetic clinician measurement markers (calipers, crosses, text).
  - Conflating the two would result in occluding the entire tumor lesion rather than the measurement symbols, completely destroying the experimental design.
- **Alternatives Considered**: Automatically thresholding high-intensity pixels within the tumor mask to infer caliper markers. Rejected as unverified and prone to false positives on hyperechoic calcifications or dermoid plugs.
- **Evidence / Assumptions**: Public MMOTU contains tumor ROI masks but lacks official caliper masks.
- **Consequences**: The data manifest explicitly defines two separate columns: `artifact_mask_path` and `tumor_mask_path`. Pipeline code strictly rejects processing an arm without verified `artifact_mask_path`.

---

## Decision D04: Multi-Center External Grounding on OvaTUS-V1 with Explicit Compatibility Gate
- **Date**: 2026-10-29
- **Status**: Accepted (Pending Data Access)
- **Decision**: Mandate external validation on OvaTUS-V1 (Samsung Medison W80i and GE Voluson S6 hardware), but block execution until an audited `LabelCompatibilityAudit` validates the mapping from OvaTUS-V1's 6 morphological classes to the binary Benign/Malignant schema.
- **Rationale**: Evaluating on external data collected on different hardware vendors provides true clinical generalization evidence, free of MMOTU frame leakage. However, evaluating on an incompatible label taxonomy produces scientifically invalid conclusions.
- **Alternatives Considered**: Evaluating only on MMOTU random splits. Rejected because random splits suffer from unmitigated frame leakage.
- **Evidence / Assumptions**: Nguyen et al. (2025) published OvaTUS-V1 under restricted clinical research terms.
- **Consequences**: Code must provide an audited label compatibility layer and must halt with clear instructions if unmapped classes are detected.

---

## Decision D05: Host/Compute Bifurcation (OCI CPU Orchestrator vs External GPU Trainer)
- **Date**: 2026-10-29
- **Status**: Accepted
- **Decision**: Partition the infrastructure so that the local OCI instance (ARM Neoverse-N1 CPU-only) performs repository orchestration, configuration management, data integrity auditing, and unit testing, while deep learning training loops and batch inpainting execute exclusively on external GPU environments (Google Colab / Kaggle).
- **Rationale**: The OCI instance is an ARM64 CPU-only host without CUDA support. Attempting to train deep convolutional backbones on 4 ARM vCPUs would be prohibitively slow and could destabilize other hosted services.
- **Alternatives Considered**: Installing PyTorch CPU on OCI and attempting small-batch training. Rejected as an inefficient use of resources and counter to production host safety rules.
- **Evidence / Assumptions**: `nvidia-smi` returned command not found; CPU is ARM Neoverse-N1.
- **Consequences**: All model training and evaluation scripts must be self-contained and runnable via CLI on external GPU platforms, with outputs serialized to standardized JSON/CSV formats.

---

## Decision D06: Mandatory Perceptual Hash Audit for Near-Duplicate Frame Clustering on MMOTU
- **Date**: 2026-10-29
- **Status**: Accepted
- **Decision**: Because public MMOTU releases lack patient IDs, enforce automated perceptual hashing (dHash/pHash with Hamming distance threshold $d_H \le 4$) during data auditing. Group near-duplicate frames into pseudo-patient clusters prior to generating train/test split manifests.
- **Rationale**: Prevents near-identical ultrasound sweep frames from appearing simultaneously in train and test folds, mitigating the severe frame leakage pervasive in prior benchmark studies.
- **Alternatives Considered**: Relying on random splitting. Rejected as scientifically invalid.
- **Evidence / Assumptions**: Documented in MMOTU forensic investigation (Zhao et al., 2022).
- **Consequences**: Splitting utility must accept pseudo-patient cluster IDs and execute Group-KFold partitioning.

---

## Decision D07: Evaluation of Ground Truth Integrity via Tri-Partition Texture Auditing
- **Date**: 2026-10-29
- **Status**: Accepted
- **Decision**: Do not compute full-image SSIM or LPIPS directly against raw annotated scans as a proxy for inpainting quality. Instead, implement a **Tri-Partition Echotextural Audit**: (1) unmasked exterior conservation (PSNR/MAE), (2) in-mask local texture variance shift ($\Delta \text{LTV}$) against adjacent tissue, and (3) synthetic digital caliper insertion on clean scans for ground-truth reference benchmarking.
- **Rationale**: In real clinical scans, the true anatomical pixels occluded by calipers are unobservable. Comparing inpainting against the caliper artifact rewards preserving the caliper rather than inpainting tissue.
- **Alternatives Considered**: Using global SSIM between $I_{raw}$ and $I_{inpainted}$. Rejected as scientifically deceptive.
- **Evidence / Assumptions**: Image processing principles of non-observable ground truth in medical artifact removal.
- **Consequences**: Image integrity evaluation metrics must explicitly isolate in-mask and exterior regions.

---

## Decision D08: Post-Hoc Temperature Scaling Restricted Strictly to Validation Split
- **Date**: 2026-10-29
- **Status**: Accepted
- **Decision**: Post-hoc probability calibration (Temperature Scaling) must optimize the temperature parameter $T$ strictly on the held-out internal Validation split, and apply the fixed $T$ to the Test split and external OvaTUS-V1 benchmark.
- **Rationale**: Fitting calibration parameters on the test set is a form of target leakage that produces optimistically biased calibration errors (ECE/MCE).
- **Alternatives Considered**: Fitting temperature scaling globally on test data. Rejected as a violation of statistical integrity.
- **Evidence / Assumptions**: Guo et al., On Calibration of Modern Neural Networks (ICML 2017).
- **Consequences**: Calibration module must accept separate validation logits and test logits.


---

## Decision D09: Rejection of Speculative OvaTUS-V1 Histological Mapping & Strict Block on External Binary Evaluation
- **Date**: 2026-10-29
- **Status**: Accepted & Canonical
- **Decision**: Formally reject and remove the provisional 6-class histological mapping for OvaTUS-V1. Authoritative documentation (SigM Lab, HUST / NHOG) confirms OvaTUS-V1 annotations represent **six IOTA morphological categories** (`Solid tumor`, `Multilocular cyst`, `Unilocular cyst`, `Dermoid cyst`, `Multilocular-solid cyst`, `Unilocular-solid cyst`), not confirmed histopathology. Binary CAD evaluation on OvaTUS-V1 is **strictly blocked** until patient-matched histopathological ground-truth records are procured and verified.
- **Rationale**:
  - In clinical ultrasonography, IOTA categories represent acoustic morphological features with heterogeneous malignancy risks (e.g., 30%–60% risk for multilocular-solid cysts).
  - Forcing IOTA morphological descriptors into binary benign/malignant labels without surgical histology ground truth produces scientifically invalid performance metrics.
- **Alternatives Considered**:
  - *Alternative 1*: Map `Unilocular` and `Dermoid` to Benign (`0`), and `Solid` / `Multilocular-solid` to Malignant (`1`), discarding `Multilocular` and `Unilocular-solid`. Rejected as unprincipled ad-hoc data dredging.
  - *Alternative 2*: Treat OvaTUS-V1 as an independent morphological classification benchmark (evaluating whether inpainting alters multi-class morphological recognition). Retained as an approved alternative evaluation path if histology records are unavailable.
- **Evidence / Assumptions**:
  - SigM Lab dataset documentation (`sigm-seee.github.io/datasets/Ovarian.html`) and IOTA consensus terminology (Timmerman et al.).
- **Consequences**: Blocker BLK-03 is escalated to require surgical histopathology verification before binary CAD testing.

---

## Decision D10: Formal Methodological Boundaries of dHash Pseudo-Patient Clustering
- **Date**: 2026-10-29
- **Status**: Accepted
- **Decision**: Define perceptual difference hashing (dHash with $d_H \le 4$) strictly as an identical/near-identical sweep frame mitigation tool. Prohibit describing dHash clustering as "patient-level separation" or "patient-independent splitting."
- **Rationale**:
  - Clinical transvaginal ultrasound examinations acquire multiple sweep angles, zoomed views, and variable depth/gain settings for a single patient study. Such frames have visual dissimilarities resulting in Hamming distances $d_H \gg 4$ and will fail to cluster together.
  - dHash clustering prevents identical sweep duplicate frames from bridging train and test folds, but cannot guarantee true patient independence without unblinded hospital clinical records.
- **Alternatives Considered**:
  - Increasing Hamming threshold to 15 or 20. Rejected because high thresholds falsely merge distinct patients presenting with standard unilocular hypoechoic cysts.
- **Evidence / Assumptions**:
  - Zhao et al. (MMOTU 2022) confirmed public releases have no patient IDs; ultrasound acoustic physics show angular speckle variation.
- **Consequences**: Mandatory disclosure statement enforced in `SPEC.md`, `AGENT.md`, and all publications.

---

## Decision D11: Two-Tier Split Manifest Leakage Verification Gate
- **Date**: 2026-10-29
- **Status**: Accepted
- **Decision**: Harden `validate_manifest_leakage` in `src/ovaryvision/manifests.py` to enforce two separate verification tiers: (1) direct image-level uniqueness (an `image_id` must appear at most once across the entire split), and (2) pseudo-patient cluster exclusivity (no cluster ID may span multiple split roles).
- **Rationale**:
  - Previously, `validate_manifest_leakage` checked only `pseudo_patient_cluster`. If a row lacked a cluster ID, an identical `image_id` present in both train and val could pass undetected.
- **Alternatives Considered**:
  - Checking only image-level uniqueness. Rejected because pseudo-patient cluster leakage must also be caught.
- **Evidence / Assumptions**:
  - Discovered during Milestone 1 implementation audit.
- **Consequences**: `tests/test_manifests.py` expanded with direct image leakage and duplicate image test cases; 100% pass rate confirmed.

---

## Decision D12: Controlled Synthetic Caliper Benchmark Architecture and Tri-Partition Image Integrity Standard
- **Date**: 2026-10-29
- **Status**: Accepted & Implemented
- **Decision**: Implement a deterministic synthetic ultrasound caliper generator (`src/ovaryvision/synthetic_artifacts.py`) and a tri-partition image integrity evaluation standard: (1) global metrics, (2) in-mask metrics ($M_{art} = 1$), (3) exterior conservation ($M_{art} = 0$), and (4) local texture variance shift ($\Delta \text{LTV}$) comparing in-mask variance against a 10-pixel surrounding parenchyma band.
- **Rationale**:
  - In retrospective clinical datasets, anatomical tissue underneath doctor-added calipers was never recorded. Naive full-image SSIM or LPIPS comparing raw and inpainted scans rewards keeping the artifact.
  - Synthetic injection onto audited clean scans provides an absolute ground-truth reference image, allowing rigorous, unconfounded evaluation of inpainting reconstruction and zero-mask occlusion.
- **Alternatives Considered**:
  - *Alternative 1*: Using raw annotated scans with naive global SSIM. Rejected as scientifically deceptive.
  - *Alternative 2*: Relying exclusively on GPU-dependent LPIPS. Rejected because the OCI development host is CPU-only; LPIPS is implemented as an optional lazy import that never halts CPU execution.
- **Evidence / Assumptions**:
  - Image processing principles of non-observable medical ground truth and acoustic speckle variance.
- **Consequences**:
  - `src/ovaryvision/synthetic_artifacts.py` implemented; `tests/test_synthetic_artifacts.py` added with 10 passing tests. `generate-synthetic-benchmark` command integrated into CLI.

---

## Decision D13: MMOTU Ingestion Safety Gate and Opt-In Acquisition SOP
- **Date**: 2026-10-29
- **Status**: Accepted & Implemented
- **Decision**: Mandate that dataset acquisition and archive extraction are strictly opt-in, requiring explicit user authorization and confirmation. Prohibit automated background downloads.
- **Rationale**:
  - Prevents unintended bandwidth consumption, disk saturation, or overwriting of local files on the OCI host.
  - Ensures compliance with data safety standards and prevents unverified downloads.
- **Alternatives Considered**:
  - Automated downloading in background via `wget`/`curl`. Rejected per research integrity and host safety rules.
- **Evidence / Assumptions**:
  - Project execution rules prohibiting unverified data creation or uncontrolled downloads.
- **Consequences**:
  - Created `scripts/ingest_mmotu.py` requiring explicit `--confirm` flag, with path traversal security checks for archive decompression.
  - Published comprehensive SOP in `docs/MMOTU_INGESTION_GUIDE.md`.

---

## Decision D14: Group-Stratified Split Architecture and Greedy Bin Allocation
- **Date**: 2026-10-29
- **Status**: Accepted & Implemented
- **Decision**: Partition datasets into 70% train, 15% validation, and 15% internal test cohorts grouped strictly by `pseudo_patient_cluster` (with individual fallback), stratified across binary diagnostic labels using deterministic greedy deficit allocation.
- **Rationale**:
  - Enforces patient/sweep isolation while maintaining target class distributions across folds.
  - Automated two-tier leakage verification (`validate_manifest_leakage`) guarantees that zero images or clusters bridge roles.
- **Alternatives Considered**:
  - Naive random sampling: Rejected due to fatal train-test frame leakage from sweep videos.
  - Standard StratifiedKFold: Inadequate because clusters have varying numbers of frames.
- **Consequences**:
  - Implemented `src/ovaryvision/splits.py` and `tests/test_splits.py` with 100% automated test coverage.
  - Exposed `generate-splits` CLI command.

---

## Decision D15: Three-Arm Preprocessing Engine with Dilated Morphological Boundary Invariance
- **Date**: 2026-10-29
- **Status**: Accepted & Implemented
- **Decision**: Implement standardized transformations for Arm A (Raw identity), Arm B (Inpainted with strict exterior conservation), and Arm C (Non-generative zero-mask occlusion). Enforce a 2-pixel cross morphological dilation ($r_{dil} = 2$) on artifact masks before inpainting or masking to eliminate anti-aliased edge artifacts.
- **Rationale**:
  - Mathematical exterior conservation guarantees that $I_B(u, v) = I_A(u, v)$ for all pixels outside the dilated mask, preventing hallucination in unaffected parenchymal tissue.
  - Pure CPU-friendly harmonic boundary diffusion (Laplace Jacobi relaxation) provides robust inpainting without requiring C++ or GPU dependencies on the OCI host.
- **Alternatives Considered**:
  - Undilated inpainting: Leaves high-frequency bright boundary artifacts along caliper edges.
  - Whole-image inpainting without mask constraint: Rejected because it corrupts genuine lesion pathology.
- **Consequences**:
  - Implemented `src/ovaryvision/preprocessing.py`, `scripts/run_preprocessing.py`, and `tests/test_preprocessing.py`.

---

## Decision D16: Validation-Only Post-Hoc Temperature Scaling with Strict Test Set Isolation
- **Date**: 2026-10-29
- **Status**: Accepted & Implemented
- **Decision**: Optimize the post-hoc temperature scaling parameter $T^*$ strictly on validation fold logits via negative log-likelihood minimization. Never allow test set labels or logits to inform temperature parameter estimation.
- **Rationale**:
  - Guarantees unbiased generalization assessment and complies with clinical validation standards.
  - Evaluates both equal-width and adaptive quantile binning for Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier score.
- **Consequences**:
  - Implemented `src/ovaryvision/calibration.py`, `tests/test_calibration.py`, and `evaluate-calibration` CLI command.

---

## Decision D17: Paired DeLong and Empirical Bootstrap Non-Inferiority Testing Framework
- **Date**: 2026-10-29
- **Status**: Accepted & Implemented
- **Decision**: Implement paired statistical comparison using DeLong et al. (1988) covariance formulation, stratified paired bootstrap confidence intervals (1,000+ resamples), pre-registered non-inferiority testing against margin $\delta = -0.02$ with sensitivity bounds $\{-0.01, -0.02, -0.03\}$, and McNemar's paired discordance test.
- **Rationale**:
  - Provides mathematical and empirical certainty for comparing non-generative masking (Arm C) against generative inpainting (Arm B).
  - Accommodates both asymptotic covariance and non-parametric empirical distributions.
- **Consequences**:
  - Implemented `src/ovaryvision/statistics.py`, `tests/test_statistics.py`, and `evaluate-statistics` CLI command.

---

## Decision D18: Multi-Seed Real-Data Baseline Execution on Colab Blackwell G4 Runtime
- **Date**: 2026-10-29
- **Status**: Accepted & Executed
- **Decision**: Execute the approved benchmark models (ConvNeXt-Small and EfficientNet-B3) on the full MMOTU dataset under Arm A (Raw baseline) across all three leak-free seeds (42, 1337, 2026) using the Google Colab G4 runtime (NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.97 GB VRAM).
- **Rationale**:
  - The Blackwell G4 GPU architecture provided massive throughput (50–60s per 25-epoch run), enabling complete 6-run multi-seed execution with early stopping and post-hoc temperature scaling in ~5 minutes of wall-clock time at negligible compute unit cost (2.11 CUs).
  - Pre-chunked upload of the audited 1,469 MMOTU images ensured exact data parity with the local OCI orchestrator.
  - Validation temperature scaling ($T^*$) proved critical: reduced test ECE by 41.1% on EfficientNet-B3 ($0.2852 \rightarrow 0.1681$) while preserving test discrimination (AUROC = 0.8931 ± 0.0443).
- **Consequences**: Complete multi-seed predictions and metrics harvested into `outputs/benchmark_runs/colab/` and aggregated in `outputs/metrics/arm_a_real_benchmark_summary.csv`.

---

## Decision D19: Gate B Real-Image Pilot Benchmark Execution and Findings
- **Date**: 2026-10-29
- **Status**: Accepted & Executed
- **Decision**: Approve and execute the 140-case real-image pilot benchmark under Arms A (Raw), B (Harmonic Inpainting with 2px dilation), and C (Zero-Masking with 2px dilation) across all 6 baseline models (ConvNeXt-Small and EfficientNet-B3 for seeds 42, 1337, 2026). Enforce strict separation between held-out test cases (primary analysis) and training cases (training-leakage sensitivity analysis).
- **Rationale**:
  - Validates end-to-end image transformation and inference pipelines prior to scaling across the complete 1,469-scan MMOTU dataset.
  - Proved strict exterior conservation ($MAE_{ext} = 0.000000, PSNR_{ext} = \infty$) across 100% of transformed images.
  - Demonstrated that Zero-Masking systematically elevates predicted malignancy probabilities ($\Delta p \in [+0.0017, +0.0328]$) and degrades Brier calibration on benign scans due to high-contrast artificial step occlusions, whereas Harmonic Inpainting preserves calibration ($Brier \approx 0.0017$) with minimal drift ($\Delta p \approx 0$).
  - Disclosed that the first 150 MMOTU scans contain 0 malignant cases, requiring the pilot evaluation to be formally reported as a feasibility, image integrity, and probability drift study rather than a powered AUROC comparison.
- **Consequences**: Complete outputs generated in `data/processed/real_pilot/`, `outputs/metrics/real_pilot_integrity_metrics.csv`, `outputs/metrics/real_pilot_classification_metrics.csv`, and comprehensive report in `docs/GATE_B_PILOT_RESULTS.md`.

---

## Decision D20: Gate D Resolution — Option A (OvaTUS-V1 Histopathology Access Request)
- **Date**: 2026-10-29
- **Status**: Approved & Adopted
- **Decision**: Confirm Option A for Gate D: keep OvaTUS-V1 external binary validation pending until access and compatible surgical histopathology ground truth are confirmed. Prohibit modifying the primary research question to 6-class morphology classification at this stage. Formally draft and finalize the academic data request letter (`docs/OVATUS_ACCESS_REQUEST.md`) addressed to Prof. Le Thi Lan at HUST/NHOG.
- **Rationale**:
  - Preserves alignment with the approved project scope and medical AI validation standards (STARD-AI/TRIPOD-AI).
  - Prevents conflating acoustic morphology with pathological histology.
- **Consequences**: `docs/OVATUS_ACCESS_REQUEST.md` finalized with exact transmission details and marked as DRAFT ONLY / PENDING USER TRANSMISSION. Binary external CAD evaluation on OvaTUS-V1 remains blocked.

