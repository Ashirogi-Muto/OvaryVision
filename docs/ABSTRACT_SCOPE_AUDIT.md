# ABSTRACT_SCOPE_AUDIT.md — Authoritative Research Scope Audit: OvaryVision

**Date of Audit**: 2026-10-29  
**Auditor**: Autonomous Senior ML Engineer & Methodology Lead  
**Status**: Authoritative Reference Identified & Harmonized  

---

## 1. Authoritative Source Documents Identified in Workspace

Two master docx documents exist in the repository root:
1. `DOC-20260827-WA0005.docx`: *OvaryVision: Rigorous Methodology for Ovarian Tumor Classification and Risk Stratification via Ultrasound Computer Vision* (Phase-I initial proposal, August 27, 2026).
2. `OvaryVision Second Deep Research Investigation and....docx`: *OvaryVision Second Deep Research Investigation and Methodological Re-Framing* (Authoritative revised research specification and critical novelty audit, August 2026).

---

## 2. Quoted Approved Abstract & Research Specification

The authoritative research scope is formally defined in **Section 15.1 of `OvaryVision Second Deep Research Investigation and....docx`**, titled **"Minimum Publishable Study (MPS)"**:

> **Title**: *Evaluating the Diagnostic Integrity of Generative Symbol Inpainting versus Non-Generative Masking in Ovarian Ultrasound Artificial Intelligence*
>
> **Primary Scientific Objective**:  
> "Determine whether MGGAN/IOPaint generative symbol inpainting introduces synthetic texture artifacts that impair downstream diagnostic accuracy compared to simple non-generative zero-masking on clean external clinical test sets." (Paragraph 134)
>
> **Execution Timeline**: 3 to 4 Months.
>
> **Datasets Utilized**:  
> "Training on MMOTU (1,469 B-mode images); external validation on a manually sanitized artifact-free test split of OvaTUS-V1." (Paragraph 136)
>
> **Methodology & Model Architectures**:  
> "Train **ConvNeXt-Small** and **EfficientNet-B3** feature extractors under three preprocessing regimes:  
> (a) Raw images containing artifacts,  
> (b) MGGAN-inpainted clean images,  
> (c) Non-generative zero-masked symbol ROIs." (Paragraph 138)
>
> **Evaluation Outcomes**:  
> - "Evaluate downstream classification accuracy, AUC, and Expected Calibration Error (ECE) on the clean external OvaTUS-V1 dataset." (Paragraph 139)  
> - "Compute structural similarity (SSIM), LPIPS, and local texture variation metrics inside the inpainted symbol regions to quantify pixel distortion." (Paragraph 140)
>
> **Target Outcome**:  
> "Proves that non-generative zero-masking matches or exceeds generative GAN inpainting accuracy while eliminating hallucinated texture risks, establishing a safer preprocessing standard for ultrasound AI." (Paragraph 141)

---

## 3. Discrepancy Audit: Approved Design vs. Current Codebase

| Dimension | Approved Abstract (Section 15.1) | Current Codebase / Prior Documentation | Audit Finding & Action Required |
|---|---|---|---|
| **Backbone 1** | **`convnext_small`** | `convnext_tiny` (in `scripts/gpu_train.py`, `SPEC.md`, configs) | **MISMATCH**: Update `scripts/gpu_train.py`, `SPEC.md`, and configs to default to `convnext_small` (while retaining `convnext_tiny` as a fast smoke-test option). |
| **Backbone 2** | **`efficientnet_b3`** | `efficientnet_b0` (in `scripts/gpu_train.py`, `SPEC.md`, configs) | **MISMATCH**: Update `scripts/gpu_train.py`, `SPEC.md`, and configs to default to `efficientnet_b3` (retaining `b0` for smoke-testing). |
| **Preprocessing Arms** | 3 conditions: (a) Raw, (b) MGGAN inpainting, (c) Zero-masking | 3 conditions: Arm A (Raw), Arm B (Inpainted), Arm C (Masked) | **ALIGNED**: Exact conceptual alignment. Must ensure Arm B is not falsely claimed as MGGAN when running Laplace/Harmonic diffusion. |
| **Target Dataset (Train)** | MMOTU (1,469 2D B-mode images) | MMOTU (1,469 2D B-mode images) | **ALIGNED**: Ground truth verified. |
| **Target Dataset (External)**| OvaTUS-V1 (sanitized external split) | OvaTUS-V1 (Samsung Medison W80i vs GE Voluson S6) | **RESTRICTED**: Verified OvaTUS-V1 annotations provide 6 IOTA morphological categories. Binary evaluation blocked until surgical pathology records are audited. |
| **Primary Metrics** | Accuracy, ROC-AUC, ECE, SSIM, LPIPS, Local Texture Variation ($\Delta \text{LTV}$) | Accuracy, ROC-AUC, ECE, MCE, Brier, SSIM, PSNR, MAE, $\Delta \text{LTV}$, DeLong test, bootstrap CIs | **ALIGNED & RIGOROUS**: Core abstract metrics fully supported. |
| **Temperature Scaling** | Excluded in abstract; abstract specifies ECE evaluation | Previously implemented post-hoc Temperature Scaling ($T^*$) | **CLARIFICATION**: Uncalibrated ECE must be reported as the primary baseline. Temperature scaling is reported strictly as post-hoc analysis, never conflated with intrinsic calibration. |
| **Explicit Exclusions** | No segmentation decoders, no U-Net, no Pointing Game, no Grad-CAM++, no O-RADS 5-tier, no BOT 3-class | All previously rejected in Decision D01 | **ALIGNED**: Scope discipline maintained. |

---

## 4. Scope Boundaries & Anti-Drift Contract

1. **Architecture Standard**: Full benchmark runs will use **ConvNeXt-Small** and **EfficientNet-B3**. Smaller variants (`convnext_tiny`, `efficientnet_b0`) are strictly designated as development smoke-test artifacts.
2. **Inpainting Provenance**: The baseline inpainting implementation is pure CPU Harmonic Boundary Diffusion (Laplace Jacobi relaxation). It must be described accurately as Harmonic Boundary Inpainting, **never falsely labeled as MGGAN** until genuine MGGAN checkpoint weights are executed.
3. **External Benchmark Validity**: OvaTUS-V1 binary benign/malignant evaluation remains blocked because public OvaTUS-V1 provides 6 IOTA morphological categories, not confirmed binary histopathology.
4. **Synthetic Benchmark Boundary**: Synthetic caliper experiments (`data/interim/synthetic_benchmark`) evaluate image-processing fidelity and algorithm boundary behavior only; they are strictly separated from clinical diagnostic evidence.
