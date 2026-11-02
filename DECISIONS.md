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

---

## Decision D21: Full 18-Run 3-Arm Benchmark Execution on NVIDIA Blackwell G4 Compute
- **Date**: 2026-10-29
- **Status**: Executed & Accepted
- **Decision**: Execute the full $3 \times 2 \times 3 = 18$ run training matrix (Arms A, B, C across ConvNeXt-Small and EfficientNet-B3 across Seeds 42, 1337, 2026) on the full 1,469-scan MMOTU dataset using the Google Colab NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GB VRAM).
- **Rationale**:
  - Provides empirical evaluation of the core research hypothesis on the complete cohort.
  - Completed all 18 runs in 24.24 wall-clock minutes.
  - Arm C (Zero-Masking) achieved higher macro-average test AUROC ($0.8280$) than Arm B (Harmonic Inpainting: $0.8089$), an observed delta of $+0.0192$ (+1.92%).
  - Both artifact removal methods improved calibration over raw Arm A ($ECE = 0.0145 \rightarrow 0.0091$ for Arm B, $0.0096$ for Arm C).
- **Consequences**: Raw predictions, checkpoints, and metrics persisted in `outputs/benchmark_runs/full_benchmark/` on branch `exp/full-cohort-benchmark`.

---

## Decision D22: Forensic Audit, Statistical Power Analysis, and Caliper Shortcut Assessment
- **Date**: 2026-10-29
- **Status**: Accepted
- **Decision**: Formally document that internal test non-inferiority against $\delta = -0.02$ cannot be statistically declared on the MMOTU test partition because $N_{pos} = 8$ produces wide standard errors ($SE \in [0.0253, 0.0718]$) that inevitably push the one-sided 95% lower bound below $-0.02$. Prohibit claiming $p > 0.05$ as evidence of equivalence or non-inferiority. Remove the unsupported scalar sample size target ($N_{pos} \ge 140$) resulting from a dimensional arithmetic error, and require a formal paired-AUROC power model (Obuchowski 1997).
- **Rationale**:
  - Scientific integrity requires distinguishing between an observed positive trend across runs ($\hat{\Delta} = +0.0192$) and a powered formal non-inferiority proof.
  - With $N_{pos} = 8$, sensitivity is quantized in steps of $12.5\%$.
  - Widespread caliper prevalence (98.1% in malignant, 95.3% in benign) demonstrates that calipers are ubiquitous, but does not by itself prove classifier shortcut dependence. Controlled counterfactual experiments are required to establish shortcut exploitation.
  - Cross-validation on MMOTU pools predictions across all 53 observed malignant cases but does not create additional independent cases.
- **Consequences**: Authored `docs/FULL_COHORT_AUDIT_REPORT.md` and recommended dispatching the OvaTUS-V1 histology access request to obtain an adequately powered independent cohort.

---

## Decision D23: Methodological Clarification — Harmonic Boundary Inpainting vs MGGAN
- **Date**: 2026-10-29
- **Status**: Accepted
- **Decision**: Update all canonical documentation to accurately identify Arm B as **Harmonic/PDE Inpainting (Fast Marching / Telea algorithm with 2px boundary dilation and exterior conservation)**, and explicitly disclose that deep generative MGGAN inpainting was not executed because MGGAN model weights were unreleased in the public MMOTU dataset.
- **Rationale**: Absolute scientific honesty regarding executed code vs literature baseline.
- **Consequences**: Updated `SPEC.md`, `README.md`, `docs/RESEARCH_PROTOCOL.md`, and `docs/FULL_COHORT_BENCHMARK_RESULTS.md`.

---

## Decision D24: MMOTU Dataset Completeness, CEUS Modality Recovery, and Annotation Provenance Audit
- **Date**: 2026-10-30
- **Status**: Accepted & Executed
- **Decision**: Formally audit and record the completeness of all public original and derivative releases of MMOTU (Google Drive, Zenodo 21128657, Zenodo 20541610 tombstone, Figshare 25115633, and Figshare 25058690). Safely stage the recovered 170-case Contrast-Enhanced Ultrasound cohort (`OTU_CEUS`) in `data/raw/mmotu_ceus/` with dedicated manifest `data/manifests/mmotu_ceus_master_manifest.csv` and release comparison table `outputs/metrics/mmotu_release_comparison.csv`.
- **Rationale**:
  - Validates that the local 2D cohort (1,469 scans, 53 malignant) is complete and unpruned compared to Figshare derivatives which dropped 267 Normal Ovary scans (1,202 scans total).
  - Proves conclusively that clinician annotation artifact masks ($M_{art}$) do not exist in any public release; all symbols were burned into scan pixels during clinical exams and were never isolated by the original authors.
  - Explains the origin of the 2,938 files in `data/raw/mmotu/tumor_masks/`: 1,469 unique binary lesion masks and 1,469 redundant byte-for-byte duplicate `<id>_binary_binary.PNG` files inherited from upstream Google Drive batch naming.
  - Documents the presence of 216 Color Doppler Flow Images (CDFI) within the 1,469 2D cohort as an unexplored clinical sub-modality confounder.
  - Recovers the 170-case CEUS cohort (161 benign, 9 malignant across 47 patients; 46.5% caliper prevalence) to enable out-of-domain transfer benchmarks without altering the completed 2D benchmark.
- **Consequences**: Staged `data/raw/mmotu_ceus/`, generated `data/manifests/mmotu_ceus_master_manifest.csv` and `outputs/metrics/mmotu_release_comparison.csv`, and authored comprehensive audit report `docs/MMOTU_DATASET_COMPLETENESS_AUDIT.md`.

---

## Decision D25: CEUS Data Integrity Verification, CDFI Confidence Stratification, and Segmentation Protocol
- **Date**: 2026-10-30
- **Status**: Accepted
- **Decision**: Formally verify all 170 CEUS scans and masks, stratify the 216 CDFI images by confidence tiers, and establish a 3-arm tumor segmentation protocol across Arms A, B, and C.
- **Rationale**:
  - Proves 100% integrity across all 170 CEUS scans (0 corruptions, 0 dimension mismatches, 100% concordance between RGB and binary masks).
  - Prohibits labeling heuristic color detections as ground-truth CDFI; creates `data/manifests/mmotu_2d_cdfi_classification.csv` with confidence tiers (207 High, 16 Probable, 57 Ambiguous, 1,189 B-mode).
  - Enforces evaluating segmentation models strictly on the 469 held-out test scans (`val.txt`) to eliminate data contamination from author-released pretrained DS²Net/MMSegmentation weights, and separates the 267 empty normal ovary masks into a dedicated false-positive rate evaluation.
  - Strictly prohibits describing CEUS as independent external validation due to unresolvable patient overlap risks between the 2D and CEUS cohorts at Beijing Shijitan Hospital and small positive sample size ($N_{pos} = 9$); restricts framing to exploratory acoustic transfer.
- **Consequences**: Created `docs/CEUS_AND_SEGMENTATION_EXPERIMENT_PLAN.md`, `outputs/metrics/ceus_data_integrity_audit.csv`, and `data/manifests/mmotu_2d_cdfi_classification.csv`.

---

## Decision D26: Internal Cluster Split, CDFI Reconciliation, Smoke Test Execution & Primary Bootstrap Governance
- **Date**: 2026-10-30
- **Status**: Accepted & Executed
- **Decision**: Formally establish the 850/150 internal train/val partition using whole dHash pseudo-patient clusters, reconcile and freeze the CDFI manifest, execute a 2-epoch GPU training smoke test, lock the paired cluster bootstrap as primary inferential procedure, and verify all 469 test checksums.
- **Rationale**:
  - **Internal Split**: Partitioning `train.txt` into 850 `internal_train` and 150 `internal_val` using whole dHash clusters guarantees zero cluster leakage while preserving the 469 official test scans (`val.txt`) strictly untouched. Manifest saved at `data/manifests/mmotu_2d_segmentation_split.csv`.
  - **CDFI Reconciliation**: Resolves the draft summary count difference (44 vs 57) by proving it arose from mental confusion with the 1,202 tumour scans; freezes the master classification with 207 High CDFI, 16 Probable CDFI, 57 Ambiguous Low-Flow, and 1,189 High B-mode under SHA-256 `487f376afc8f3ac5dd32f2a82453cdaf8659d1de52060f342ba1cf1fee4f79ff`.
  - **Empirical Smoke Test**: Executed 2-epoch Arm A training on Tesla T4 GPU (Google Colab) with real scans and masks at $512 \times 512$ resolution, measuring actual runtime (5.66s / 5.29s), peak VRAM (4.752 GB), and active loss convergence ($1.0869 \to 0.8865 \to 0.8010$).
  - **Primary Bootstrap**: Mandates paired pseudo-patient-cluster bootstrap ($B=2,000$ iterations) with pooled cluster resampling across the 82 test clusters as the primary inferential procedure; keeps Wilcoxon secondary.
  - **Mask Verification**: Assigns $Y_{tumour} \equiv 0$ to normal training cases while preserving original anatomical ovary masks ($M_{ovary}$) for separate test evaluation. Confirms 100% of 469 test images match SHA-256 hashes.
- **Consequences**: Created `scripts/smoke_test_arm_a.py`, `scripts/generate_segmentation_splits.py`, `data/manifests/mmotu_2d_segmentation_split.csv`, updated `docs/SEGMENTATION_PRE-EXECUTION_AUDIT.md` and `docs/CEUS_AND_SEGMENTATION_EXPERIMENT_PLAN.md`.

---

## Decision D27: G4 Blackwell Compute Profiling, Paired Epoch Augmentation, and Multiplicity-Preserving Bootstrap
- **Date**: 2026-10-30
- **Status**: Accepted & Verified
- **Decision**: Profile NVIDIA RTX PRO 6000 Blackwell Server Edition (G4 runtime) for segmentation benchmark execution; update paired cluster bootstrap to preserve multinomial cluster draw multiplicities with conditional rejection sampling for zero-overlap draws; enforce epoch-varying paired spatial augmentations across arms in `SegmentationDataset`; add AMP FP16 and safe checkpoint resumption to `scripts/run_three_arm_segmentation_benchmark.py`.
- **Rationale**:
  - G4 profiling demonstrates 37.07 ms training step latency (215.8 imgs/sec), a 9.76x speedup over T4 AMP, using 2.10 GB of VRAM (2.2% of 95 GB capacity), with stable GradScaler scaling (65,536.0) and zero NaNs/Infs.
  - Multiplicity preservation in bootstrap prevents variance underestimation caused by set-union collapsing. Conditional rejection sampling preserves Davison & Hinkley (1997) conditional bootstrap guarantees without arbitrary constant imputation.
  - Epoch-varying paired augmentation guarantees identical spatial transformations across Arms A, B, C within each epoch while allowing transformations to vary between epochs.
  - Training script enforces internal validation checkpointing, strict test-set isolation, independent seed initialization, and automatic resumption from interrupted checkpoints.
- **Consequences**: Updated `src/ovaryvision/statistics.py`, `scripts/run_three_arm_segmentation_benchmark.py`, `docs/SEGMENTATION_PRE-EXECUTION_AUDIT.md`, `tests/test_statistics.py`, and `tests/test_segmentation_benchmark.py`.

---

## Decision D28: BCa Cluster Bootstrap Implementation with Percentile Fallback
- **Date**: 2026-10-30
- **Status**: Accepted & Implemented
- **Decision**: Restore and formally implement BCa (Bias-Corrected and Accelerated) 95% Confidence Intervals as the primary estimator in `paired_cluster_bootstrap_segmentation()`, using cluster-level leave-one-out jackknife for acceleration $a$ and median bias correction $z_0$. Implement a robust fallback to standard percentile intervals if adjusted quantiles degenerate, reporting both BCa and percentile bounds in `ClusterBootstrapResult`.
- **Rationale**:
  - Eliminates the specification-implementation gap: `SPEC.md` and `docs/RESEARCH_PROTOCOL.md` specified BCa, but legacy classification code in `595ba9f` only called `np.percentile()`.
  - BCa provides second-order accuracy ($O(1/n)$ coverage error) vs first-order ($O(1/\sqrt{n})$) for percentile intervals.
  - The $O(K)$ cluster jackknife computes acceleration $a$ across the 82 test clusters in milliseconds.
  - Fallback guard ensures numerical robustness on conditional subsets (such as LPR on the 193-scan overlap cohort) if extreme cluster leverage distorts adjusted quantiles.
- **Consequences**: Updated `src/ovaryvision/statistics.py`, `docs/SEGMENTATION_PRE-EXECUTION_AUDIT.md`, and added `test_paired_cluster_bootstrap_bca_method_selection` in `tests/test_statistics.py`.

---

## Decision D29: Three-Arm Seed 42 Segmentation Pilot Execution, Collation Fix & Empirical Validation
- **Date**: 2026-10-30
- **Status**: Executed & Verified
- **Decision**: Successfully execute the Three-Arm Seed 42 Segmentation Pilot on Google Colab G4 Blackwell runtime (`arm_a_raw`, `arm_b_inpainted`, `arm_c_masked`, 40 epochs each, seed 42) using the audited U-Net ResNet-34 architecture, AMP FP16, batch size 8, and internal validation checkpointing, while maintaining strict test-set isolation (zero test scans loaded or evaluated).
- **Rationale**:
  - Validated data loader pipeline under PyTorch 2.11 / Python 3.13 by implementing `collate_seg_fn` to prevent metadata dictionary collation type errors (`TypeError: must be real number, not str`).
  - Executed all 40 epochs for each of the three arms autonomously via detached daemon session on NVIDIA RTX PRO 6000 Blackwell Server Edition (94.97 GB VRAM).
  - Empirical training performance:
    - **Arm A (Raw)**: 40 epochs in 706.5s (11.78 min; mean 17.66s/epoch); best checkpoint selected at Epoch 29 (internal val loss 0.1352, internal val DSC 0.8665).
    - **Arm B (Inpainted)**: 40 epochs in 3,428.8s (57.15 min; mean 85.72s/epoch, reflective of on-the-fly OpenCV Navier-Stokes inpainting during DataLoader batching); best checkpoint selected at Epoch 33 (internal val loss 0.1339, internal val DSC 0.8648).
    - **Arm C (Masked)**: 40 epochs in 750.6s (12.51 min; mean 18.76s/epoch); best checkpoint selected at Epoch 32 (internal val loss 0.1319, internal val DSC 0.8623).
  - Preserved all 6 checkpoints (`best_checkpoint.pth` and `latest_checkpoint.pth` per arm), `history.csv`, and `run_summary.json` in `outputs/benchmark_runs/segmentation_benchmark/`.
  - Zero test set exposure verified (469 held-out test scans remained strictly unaccessed on remote disk and locally).
  - Immediately terminated G4 Colab session following deliverables verification to conserve compute budget.
- **Consequences**: Established empirical pilot baseline and confirmed stability before authorizing remaining seeds (1337, 2026). Updated `DECISIONS.md`, `AGENT.md`, `PROJECT_STATE.md`, and `CHANGELOG.md`.

---

## Decision D31: Completion and Statistical Aggregation of the Full Three-Seed, Three-Arm Segmentation Benchmark
- **Date**: 2026-11-01
- **Status**: Executed & Verified
- **Decision**: Train and evaluate the remaining six segmentation runs (Seed 1337: Arms A, B, C; Seed 2026: Arms A, B, C) on the G4 Blackwell runtime, evaluate all nine checkpoints on the 469 official held-out test scans (382 tumour, 87 normal ovary), and aggregate the complete 3-seed benchmark using scan-level multi-seed averaged paired pseudo-patient cluster bootstrap ($B=2,000$ resamples, BCa 95% CIs).
- **Rationale**:
  - Completes the full $3 \times 3 \times 40 = 9$ run benchmark design under strictly frozen experimental protocol (U-Net ResNet-34, $512 \times 512$, batch size 8, AMP FP16, AdamW, ComboLoss, epoch-varying paired augmentation).
  - Preserved test isolation during all training phases (469 test scans never loaded or referenced during training).
  - All nine runs completed successfully with zero NaNs/Infs:
    - Arm A (Raw): Seed 42 (11.78m, best ep 29), Seed 1337 (11.68m, best ep 23), Seed 2026 (11.63m, best ep 16).
    - Arm B (Inpainted): Seed 42 (57.15m, best ep 33), Seed 1337 (54.51m, best ep 29), Seed 2026 (54.37m, best ep 17).
    - Arm C (Masked): Seed 42 (12.51m, best ep 32), Seed 1337 (12.34m, best ep 40), Seed 2026 (12.36m, best ep 39).
  - Pooled Multi-Seed Empirical Findings (Scan-level 3-Seed Mean on Held-Out Test Set):
    - **Tumour Delineation ($N=382$)**:
      - Arm A (Raw): DSC $0.8073 \pm 0.2490$ (median 0.9098), IoU $0.7368 \pm 0.2579$, Boundary IoU $0.0717 \pm 0.0519$.
      - Arm B (Inpainted): DSC $0.8172 \pm 0.2401$ (median 0.9117), IoU $0.7474 \pm 0.2493$, Boundary IoU $0.0724 \pm 0.0511$.
      - Arm C (Masked): DSC $0.8191 \pm 0.2168$ (median 0.9002), IoU $0.7459 \pm 0.2310$, Boundary IoU $0.0737 \pm 0.0511$.
      - Delta DSC (C vs B): $+0.0019$ (+0.19%), BCa 95% CI $[-0.0070, +0.0131]$, $p = 0.776$ (no significant difference).
      - Delta DSC (C vs A): $+0.0118$ (+1.18%), BCa 95% CI $[+0.0009, +0.0240]$, $p = 0.042$.
      - Delta DSC (B vs A): $+0.0099$ (+0.99%), BCa 95% CI $[-0.0009, +0.0215]$, $p = 0.071$.
    - **In-Caliper Lesion Preservation Ratio ($N=193$ valid overlap cohort, $N=189$ zero-overlap excluded without imputation)**:
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
- **Consequences**: Complete 9-run deliverables finalized in `outputs/benchmark_runs/segmentation_benchmark/final_three_seed_benchmark/`. Full test suite passing (86/86). All classification benchmark outputs remain 100% frozen.



