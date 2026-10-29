# FULL_COHORT_AUDIT_REPORT.md — Forensic Scientific & Statistical Audit

**Audit Date**: October 29, 2026
**Auditor**: Lead Agentic Research Pair
**Target Repository**: `Ashirogi-Muto/OvaryVision`
**Target Branch**: `exp/full-cohort-benchmark`
**Primary Focus**: Cohort Validation, Statistical Recomputations, Non-Inferiority Verification, Mask QC, Method Reconciliation, Power Analysis, and Scientific Claim Restraint

---

## 1. Categorized Executive Summary

To adhere to rigorous clinical AI evaluation standards (STARD-AI / TRIPOD-AI), all findings from the 18-run full-cohort benchmark are categorized into verified observations, exploratory results, statistical limitations, unresolved questions, and next experiments:

### 1.1 Verified Observations
- **Evaluation Cohort & Split Consistency**: Verified across all 18 runs. The 1,469 MMOTU scans were split into 1,029 training, 220 validation, and 220 internal test scans across three fixed random seeds (42, 1337, 2026). Within each seed, all three experimental arms (Arm A, Arm B, Arm C) evaluated the exact same 220 test image IDs in identical row order with identical ground-truth labels.
- **Extreme Class Imbalance**: In every seed, the 220-image internal test split contains only **8 malignant cases** ($N_{pos} = 8$) and **212 benign cases** ($N_{neg} = 212$), reflecting a malignant prevalence of $3.64\%$. The full MMOTU dataset contains only **53 malignant cases** ($3.61\%$).
- **Artifact Prevalence**: Automated morphological detection identified annotation calipers in **1,402 of 1,469 scans (95.44%)** across the full MMOTU cohort: **52 of 53 malignant scans (98.11%)** and **1,350 of 1,416 benign scans (95.34%)**.
- **Transformation Invariance**: Arm B (Harmonic Inpainting) and Arm C (Zero-Masking) enforced exact mathematical exterior tissue conservation ($I_{B} = I_{A}$ and $I_{C} = I_{A}$ outside the 2-pixel dilated artifact mask), preventing unintended corruption of parenchymal tissue.
- **Implemented Preprocessing Method**: Arm B executed classical **Fast Marching / Telea Harmonic PDE Inpainting** with a 2-pixel mask dilation. Deep generative inpainting with MGGAN (Wang et al., 2023) was **not executed** because author checkpoint weights were unreleased in the public MMOTU distribution.
- **External Dataset Status**: External binary CAD validation on OvaTUS-V1 is **strictly blocked** under Gate D Option A because public OvaTUS-V1 annotations provide six IOTA morphological categories, not confirmed surgical histopathology.

### 1.2 Exploratory Results
- **Observed AUROC**: Across the 18 trained models (3 Arms $\times$ 2 Backbones $\times$ 3 Seeds), observed macro-average test AUROC across runs was:
  - **Arm A (Raw Baseline)**: $0.8409 \pm 0.0884$ (range: $0.7235 - 0.9298$)
  - **Arm B (Harmonic Inpainting)**: $0.8089 \pm 0.0967$ (range: $0.6621 - 0.9292$)
  - **Arm C (Zero-Masking)**: $0.8280 \pm 0.1140$ (range: $0.6663 - 0.9381$)
- **Probability Calibration**: Validation-only temperature scaling reduced calibration error relative to uncleaned raw models:
  - **Arm A**: Calibrated ECE $= 0.0144 \pm 0.0087 \mid$ Brier Score $= 0.0316 \pm 0.0031$
  - **Arm B**: Calibrated ECE $= 0.0091 \pm 0.0049 \mid$ Brier Score $= 0.0328 \pm 0.0013$
  - **Arm C**: Calibrated ECE $= 0.0096 \pm 0.0046 \mid$ Brier Score $= 0.0323 \pm 0.0022$

### 1.3 Statistical Limitations
- **Non-Inferiority Inconclusive**: Statistical non-inferiority of zero-masking relative to harmonic inpainting ($\delta = -0.02$, $\alpha = 0.05$) was **formally False across all 6 paired comparisons**. This failure is an inevitable mathematical consequence of evaluating only $N_{pos} = 8$ positive cases per test split, which inflates DeLong standard errors ($SE \in [0.0253, 0.0718]$).
- **Quantization of Hard Decision Metrics**: Because $N_{pos} = 8$, any hard threshold metric (e.g. sensitivity at $\tau = 0.5$) steps in discrete increments of $1/8 = 12.5\%$.
- **Perceptual Hash Grouping Cannot Prove Patient Independence**: Perceptual difference hashing ($\text{dHash} \le 4$) clustered 290 scans into 136 multi-frame clusters with zero split leakage. However, different ultrasound views of the same patient taken at distinct gain settings or angles exceed the $\text{dHash} \le 4$ threshold and enter separate splits as pseudo-independent singletons. Patient-level independence cannot be established without clinical hospital accession numbers.
- **Power Calculation Correction**: The previously reported formula $\frac{(z_{0.95} + z_{0.80})^2 \cdot \sigma_d^2}{\delta^2} \approx 140$ contained a critical mathematical defect (the arithmetic evaluates to $13.92$, and $0.030$ was an observed standard error from $N_{pos}=8$, not an asymptotic variance per case). Quoting an unsupported numeric target of 140 is invalid.

### 1.4 Unresolved Questions
- **Shortcut Dependence Unproven**: Documenting high caliper prevalence (98.1% in malignant, 95.3% in benign) does **not** prove that deep classifiers rely on calipers as a predictive shortcut. Because calipers are present in nearly all scans regardless of class, direct evidence of shortcut dependence (such as controlled counterfactual caliper insertion/ablation or gradient attribution analysis) is required before shortcut exploitation can be claimed.
- **True External Diagnostic Generalization**: It remains unknown whether zero-masking or harmonic inpainting yields superior diagnostic accuracy on an independent multi-center cohort until verified histopathological labels are procured for OvaTUS-V1.

### 1.5 Next Experiments
- **Out-of-Fold 5-Fold Cross-Validation on MMOTU**: Evaluate all 5 folds out-of-fold to yield predictions across all 53 observed malignant cases. Note: this increases out-of-fold sample size but does not create additional independent patients.
- **OvaTUS-V1 Histopathology Access Request**: Dispatch `docs/OVATUS_ACCESS_REQUEST.md` to HUST/NHOG to procure verified surgical histopathology ground truth.

---

## 2. Evaluation Cohort Validation

### 2.1 Partition Sizes & Class Counts
The benchmark partitions were verified across all three fixed random seeds (42, 1337, 2026), generated via Group-Stratified K-Fold partitioning based on perceptual difference hashing ($\text{dHash} \le 4$):

| Split Role | Benign ($y=0$) | Malignant ($y=1$) | Total Scans | Class Prevalence |
|---|---|---|---|---|
| **Training Fold** (70%) | 992 | 37 | 1,029 | 3.60% Malignant |
| **Validation Fold** (15%) | 212 | 8 | 220 | 3.64% Malignant |
| **Internal Test Fold** (15%) | 212 | 8 | 220 | 3.64% Malignant |
| **Full MMOTU Cohort** (100%) | **1,416** | **53** | **1,469** | **3.61% Malignant** |

- **Image ID & Label Parity**: Confirmed 100% identical across all three arms within each seed. Every model evaluated the exact same 220 test images in identical order with identical binary labels.
- **Label Derivation**: MMOTU classes 0–6 are mapped to Benign ($y=0$) and class 7 is mapped to Malignant ($y=1$). Labels were derived strictly from the official annotation catalog, not inferred from segmentation masks.

### 2.2 Mathematical Consequences of $N_{pos} = 8$
With only $N_{pos} = 8$ malignant scans in the test partition:
1. **Sensitivity Quantization**: Decision metrics step in increments of $1/8 = 12.5\%$. A single misclassification shifts sensitivity from $100\%$ to $87.5\%$ (Wilson 95% CI: $[47.3\%, 99.7\%]$) or $75.0\%$ (Wilson 95% CI: $[34.9\%, 96.8\%]$). Point estimates of sensitivity carry severe uncertainty.
2. **AUROC Variance**: Under the Hanley-McNeil formulation, AUROC variance scales inversely with $N_{pos} \cdot N_{neg}$. With $N_{pos} = 8$ and $N_{neg} = 212$, a single discordant ranking of a malignant scan shifts AUROC by up to $0.125$.
3. **dHash Limitations**: Grouping by perceptual difference hash ($\text{dHash} \le 4$) isolated 136 multi-frame clusters (290 scans) without split leakage. However, ultrasound scans from the same examination acquired at different acoustic frequencies, probe orientations, or depths have $\text{dHash} > 4$ and enter separate folds. Perceptual hashing mitigates near-duplicate frame leakage but cannot prove patient-level independence.

---

## 3. Statistical Audit & Recomputations

### 3.1 Recomputed Metrics Across All 18 Runs
All numbers were recomputed directly from the saved prediction files (`predictions.csv` across all 18 runs):

| Run Name | Architecture | Seed | Test AUROC | Calibrated ECE | Calibrated Brier | Training Time |
|---|---|---|---|---|---|---|
| `arm_a_convnext_small_seed42` | ConvNeXt-Small | 42 | 0.9298 | 0.0283 | 0.0255 | 105.3s |
| `arm_a_convnext_small_seed1337` | ConvNeXt-Small | 1337 | 0.8349 | 0.0057 | 0.0319 | 102.0s |
| `arm_a_convnext_small_seed2026` | ConvNeXt-Small | 2026 | 0.7476 | 0.0162 | 0.0336 | 102.5s |
| `arm_a_efficientnet_b3_seed42` | EfficientNet-B3 | 42 | 0.9204 | 0.0196 | 0.0318 | 43.2s |
| `arm_a_efficientnet_b3_seed1337` | EfficientNet-B3 | 1337 | 0.8892 | 0.0100 | 0.0329 | 42.6s |
| `arm_a_efficientnet_b3_seed2026` | EfficientNet-B3 | 2026 | 0.7235 | 0.0069 | 0.0339 | 42.9s |
| `arm_b_convnext_small_seed42` | ConvNeXt-Small | 42 | 0.9292 | 0.0155 | 0.0310 | 96.1s |
| `arm_b_convnext_small_seed1337` | ConvNeXt-Small | 1337 | 0.8308 | 0.0055 | 0.0317 | 113.4s |
| `arm_b_convnext_small_seed2026` | ConvNeXt-Small | 2026 | 0.7288 | 0.0147 | 0.0341 | 112.2s |
| `arm_b_efficientnet_b3_seed42` | EfficientNet-B3 | 42 | 0.8620 | 0.0078 | 0.0330 | 50.6s |
| `arm_b_efficientnet_b3_seed1337` | EfficientNet-B3 | 1337 | 0.8402 | 0.0033 | 0.0330 | 59.0s |
| `arm_b_efficientnet_b3_seed2026` | EfficientNet-B3 | 2026 | 0.6621 | 0.0080 | 0.0343 | 58.7s |
| `arm_c_convnext_small_seed42` | ConvNeXt-Small | 42 | 0.9239 | 0.0131 | 0.0285 | 113.0s |
| `arm_c_convnext_small_seed1337` | ConvNeXt-Small | 1337 | 0.8373 | 0.0077 | 0.0320 | 113.6s |
| `arm_c_convnext_small_seed2026` | ConvNeXt-Small | 2026 | 0.7117 | 0.0080 | 0.0345 | 95.1s |
| `arm_c_efficientnet_b3_seed42` | EfficientNet-B3 | 42 | 0.8909 | 0.0049 | 0.0331 | 59.4s |
| `arm_c_efficientnet_b3_seed1337` | EfficientNet-B3 | 1337 | 0.9381 | 0.0171 | 0.0316 | 59.0s |
| `arm_c_efficientnet_b3_seed2026` | EfficientNet-B3 | 2026 | 0.6663 | 0.0070 | 0.0344 | 58.6s |

**Arm Macro-Averages Across the 6 Runs Per Arm** (mean $\pm$ sample SD across runs):
- **Arm A (Raw Baseline)**: $\text{AUROC} = 0.8409 \pm 0.0884 \mid \text{ECE} = 0.0144 \pm 0.0087 \mid \text{Brier} = 0.0316 \pm 0.0031$
- **Arm B (Harmonic Inpainting)**: $\text{AUROC} = 0.8089 \pm 0.0967 \mid \text{ECE} = 0.0091 \pm 0.0049 \mid \text{Brier} = 0.0328 \pm 0.0013$
- **Arm C (Zero-Masking)**: $\text{AUROC} = 0.8280 \pm 0.1140 \mid \text{ECE} = 0.0096 \pm 0.0046 \mid \text{Brier} = 0.0323 \pm 0.0022$

### 3.2 Paired DeLong & Non-Inferiority Verification
The paired non-inferiority hypothesis tests were audited for non-inferiority margin $\delta = -0.02$ at one-sided $\alpha = 0.05$:
$$H_0: \text{AUC}_{\text{Arm C}} - \text{AUC}_{\text{Arm B}} \le \delta \quad \text{vs} \quad H_1: \text{AUC}_{\text{Arm C}} - \text{AUC}_{\text{Arm B}} > \delta \quad (\delta = -0.02)$$

The one-sided 95% confidence lower bound is defined strictly as:
$$\text{Lower Bound} = \hat{\Delta}_{\text{observed}} - z_{1-\alpha} \cdot SE_{\text{DeLong}} = \hat{\Delta}_{\text{observed}} - 1.6449 \cdot SE_{\text{DeLong}}$$

Non-inferiority is declared if and only if:
$$\text{Lower Bound} > \delta \iff \hat{\Delta}_{\text{observed}} - 1.6449 \cdot SE_{\text{DeLong}} > -0.0200$$

| Architecture | Seed | $\text{AUC}_B$ | $\text{AUC}_C$ | $\hat{\Delta}_{C-B}$ | $SE_{\text{DeLong}}$ | $z_{\text{NI}}$ | $p_{\text{NI}}$ (1-sided) | Lower 95% Bound | Declared Non-Inferior? |
|---|---|---|---|---|---|---|---|---|---|
| ConvNeXt-Small | 42 | 0.9292 | 0.9239 | -0.0053 | 0.0253 | +0.5814 | 0.2805 | -0.0469 | **False** |
| ConvNeXt-Small | 1337 | 0.8308 | 0.8373 | +0.0065 | 0.0304 | +0.8716 | 0.1917 | -0.0435 | **False** |
| ConvNeXt-Small | 2026 | 0.7288 | 0.7117 | -0.0171 | 0.0372 | +0.0779 | 0.4689 | -0.0783 | **False** |
| EfficientNet-B3 | 42 | 0.8620 | 0.8909 | +0.0289 | 0.0400 | +1.2221 | 0.1108 | -0.0369 | **False** |
| EfficientNet-B3 | 1337 | 0.8402 | 0.9381 | +0.0979 | 0.0718 | +1.6419 | 0.0503 | -0.0202 | **False** |
| EfficientNet-B3 | 2026 | 0.6621 | 0.6663 | +0.0041 | 0.0528 | +0.4573 | 0.3237 | -0.0827 | **False** |

### 3.3 Mathematical Cause of Non-Inferiority Rejection
In every run, the decision was formally **False**:
- Because the test fold contains only $N_{pos} = 8$ positive cases, the DeLong standard errors are wide ($SE \in [0.0253, 0.0718]$).
- The margin $\delta = -0.0200$ is narrow. The subtracted term $1.645 \cdot SE$ ranges from $0.0416$ to $0.1181$.
- Consequently, even when Arm C achieved an observed AUROC advantage of nearly $+10\%$ (`efficientnet_b3_seed1337`: $\hat{\Delta} = +0.0979$), the large standard error ($0.0718$) pushed the lower bound to $-0.0202$, narrowly missing the $-0.0200$ threshold ($p_{\text{NI}} = 0.0503$).
- **Strict Scientific Rule**: A non-significant test ($p > 0.05$) must **never** be cited or reported as evidence of equivalence or non-inferiority.

---

## 4. Artifact Mask & Quality Control (QC) Audit

### 4.1 Prevalence & QC Across the 1,469-Scan Cohort
The automated caliper detection tool was evaluated against all 1,469 MMOTU images and cross-referenced with clinical tumor segmentation masks (`data/raw/mmotu/tumor_masks/`):

| Diagnostic Label | Total Scans | Calipers Present | FLAGGED EMPTY | FLAGGED TUMOR OVERLAP (>50%) | FLAGGED EXCESSIVE AREA (>5%) | PASSED ALL QC |
|---|---|---|---|---|---|---|
| **Benign Scans** ($y=0$) | 1,416 | 1,350 (95.34%) | 66 (4.66%) | 56 (3.95%) | 11 (0.78%) | 1,283 (90.61%) |
| **Malignant Scans** ($y=1$) | 53 | 52 (98.11%) | 1 (1.89%) | 0 (0.00%) | 0 (0.00%) | 52 (98.11%) |
| **Combined Cohort** | **1,469** | **1,402 (95.44%)** | **67 (4.56%)** | **56 (3.81%)** | **11 (0.75%)** | **1,335 (90.88%)** |

### 4.2 Definition and Behavior of QC Flags
- **Artifact Area Ratio**: $|M_{\text{art}}| / (H \times W)$. If $> 0.05$, flagged as `FLAGGED_EXCESSIVE_AREA`.
- **Tumor Overlap**: $|M_{\text{art}} \cap M_{\text{tumor}}| / |M_{\text{art}}|$ (the fraction of detected caliper pixels that fall within the tumor boundary). If $> 0.50$, flagged as `FLAGGED_TUMOR_OVERLAP`.
- **Exclusivity**: QC statuses are assigned hierarchically in the script (Empty $\rightarrow$ Excessive Area $\rightarrow$ Tumor Overlap $\rightarrow$ Passed). Zero cases in the entire cohort met both excessive area and tumor overlap criteria.
- **Malignant Scans**:
  - Exactly 1 of 53 malignant scans (Scan `1452`) was flagged empty (1.89%).
  - 52 of 53 malignant scans (98.11%) contain detected calipers.
  - In malignant scans with calipers, mean tumor overlap was **5.02%** (maximum 46.38% on Scan `1437`). Zero malignant scans exceeded 50% tumor overlap.

### 4.3 Rigorous Assessment of the Shortcut Hypothesis
- **Prevalence vs Shortcut Dependence**: Calipers are present in **98.1% of malignant scans** and **95.3% of benign scans**. This high prevalence demonstrates that calipers are ubiquitous in MMOTU, but it does **not** prove that deep networks use calipers as a discriminative shortcut. Because calipers occur at nearly identical rates across both classes, caliper presence alone carries minimal class-discriminative mutual information.
- **Unperformed Tests**: Proving classifier shortcut dependence requires controlled experiments—such as injecting synthetic calipers onto clean benign images to measure false-positive rate increases, ablating calipers on raw images, or computing pixel attribution maps (e.g. Integrated Gradients). These experiments have not been performed on the full cohort.
- **Claim Restraint**: It is incorrect to claim that "shortcut dependence is confirmed" or that "models successfully remove shortcuts without collapse." We report only that caliper markers are widespread, that zero-masking achieved an observed test AUROC of 0.8280 compared to 0.8409 for raw scans on this 8-case test set, and that the effect of artifact removal remains an exploratory observation.

### 4.4 Automated Mask Limitations & Visual Contact Sheets
- **Detection Method**: The masks were produced by an automated morphological filter isolating bright pixels ($\ge 240$) and linear structures. They are **not clinician-annotated artifact masks**.
- **Limitations**: The automated detector can produce false positives (capturing hyperechoic pelvic reflections or bone interfaces) and false negatives (missing low-contrast or anti-aliased crosshair tick marks).
- **Inspection Contact Sheets**:
  - `outputs/qc/contact_sheets/malignant_cases_artifact_contact_sheet.png`: All 53 malignant cases with detected artifact overlays (yellow) and tumor boundaries (cyan contour).
  - `outputs/qc/contact_sheets/benign_sample_artifact_contact_sheet.png`: 36 representative benign scans.
  - *Data Policy*: These image files remain local in `outputs/qc/` and are untracked by Git per `.gitignore` repository policy.

---

## 5. Method Reconciliation (Harmonic vs MGGAN)

- **Executed Arm B Implementation**:
  The benchmark executed **Classical Fast Marching Harmonic/PDE Inpainting (Telea algorithm)** with a 2-pixel morphological boundary dilation and strict exterior tissue conservation.
- **MGGAN Disclosure**:
  Deep generative inpainting using Mask-Guided GAN (MGGAN; Wang et al., 2023) was **not executed**. MGGAN weights were not packaged in the public Zenodo/Figshare release of MMOTU.
- **Documentation Alignment**:
  `SPEC.md`, `README.md`, `docs/RESEARCH_PROTOCOL.md`, and `docs/FULL_COHORT_BENCHMARK_RESULTS.md` have been updated to describe Arm B as Harmonic PDE Inpainting. Classical inpainting is not described as generative inpainting.

---

## 6. Sample Size & Power Analysis Refinement

### 6.1 Dissection of the Previous Calculation Error
The previous draft reported:
$$N_{pos} \ge \frac{(1.645 + 0.842)^2 \cdot (0.030)^2}{(-0.02)^2} \approx 140$$

This statement contained two critical mathematical flaws:
1. **Arithmetic Error**: $(1.645 + 0.842)^2 \approx 6.185$. Then $(0.030 / 0.020)^2 = 1.5^2 = 2.25$. The product is $6.185 \times 2.25 = 13.916$, not 140.
2. **Dimensional Flaw**: The scalar $0.030$ was an observed standard error ($SE_{\text{DeLong}}$) from a test fold of $N_{pos} = 8$, not an asymptotic population standard deviation per observation ($\sigma$). Because standard error already incorporates sample size ($SE = \sigma / \sqrt{N}$), plugging $SE$ into a sample size formula for variance is dimensionally invalid.

### 6.2 Valid Methodological Framework for Paired AUROC Non-Inferiority
Under validated statistical frameworks for paired diagnostic ROC comparison (Obuchowski 1997; Zhou, Obuchowski, McClish 2002), a defensible sample size calculation cannot be derived from a single scalar formula. It requires estimating:
1. Expected baseline AUROC for both models ($\theta_1, \theta_2$)
2. Latent rating variance components in positive ($Q_1$) and negative ($Q_2$) populations
3. Correlation of paired diagnostic predictions on diseased cases ($r_{pos}$) and non-diseased cases ($r_{neg}$)
4. Cohort prevalence ratio $k = N_{neg} / N_{pos}$
5. Pre-registered non-inferiority margin $\delta$ and true expected difference $\Delta_0$

Because these bivariate correlation parameters have not been estimated on an external cohort, **any quoted single numeric sample size target (such as 140) is unsupported and has been removed**.

### 6.3 Five-Fold Cross-Validation Clarification
- Five-fold cross-validation pools predictions across the entire available MMOTU dataset, providing out-of-fold evaluation for all **53 observed malignant cases** rather than only 8.
- **Critical Caveat**: Cross-validation **does not create additional independent malignant cases**. The total number of unique malignant scans in MMOTU remains 53. Because training sets overlap across folds, out-of-fold predictions are statistically correlated, violating independent-sample assumptions. Cross-validation alone cannot establish non-inferiority.
- A definitive non-inferiority evaluation requires an adequately powered independent external test cohort with verified histopathology.
