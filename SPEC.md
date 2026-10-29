# SPEC.md — Canonical Research Specification: OvaryVision

**Title**: OvaryVision: A Data-Centric Benchmark of Generative Inpainting vs. Non-Generative Masking for Ovarian Ultrasound Computer-Aided Diagnosis  
**Version**: 2.0.0 (Canonical Specification)  
**Date**: 2026-10-29  
**Status**: Active Research Specification  
**Canonical Path**: `/home/ubuntu/major/SPEC.md`  

---

## 1. Research Framing

### 1.1 Problem Statement & Clinical Motivation
Ovarian cancer remains among the most lethal gynecological malignancies worldwide, often diagnosed at advanced stages (FIGO III/IV) due to insidious, non-specific early clinical presentations and high operator variability in frontline Transvaginal Ultrasound (TVUS) interpretation. While computer-aided diagnosis (CAD) deep learning systems trained on open ultrasound benchmarks (e.g., MMOTU) report diagnostic accuracy and Area Under the ROC Curve (AUC) exceeding 0.90, their translational clinical validity is undermined by three severe methodological confounding factors:

1. **Annotation Artifact Shortcuts (Clever Hans Effects)**: Sonographers and radiologists routinely annotate suspicious ovarian masses during live examinations with synthetic symbols: measurement calipers (crosses, dashed lines, tick marks), measurement callouts, text labels, and boundary traces. These symbols are disproportionately placed directly on or adjacent to complex, suspicious lesions. Deep neural networks rapidly exploit these synthetic high-frequency markers as predictive shortcuts rather than learning bona fide morphological echotexture (e.g., ground-glass acoustic attenuation, acoustic shadows, papillary projections, irregular septations).
2. **Generative Inpainting Hallucination Risks**: Prior attempts to mitigate shortcut learning (such as Mask-Guided GANs / MGGAN or diffusion inpainters) synthesize replacement pixels within annotation-occluded regions. However, generative networks have no access to the true unobserved underlying anatomy; they synthesize echotexture guided purely by probabilistic priors. In diagnostic ultrasonography, small focal structures (e.g., a 3 mm solid papillary projection within a cyst wall) separate benign cystadenomas from malignant cystadenocarcinomas. Generative inpainting risks hallucinating non-existent solid structures or smoothing out true pathological microstructures, creating unacceptable clinical safety hazards.
3. **Pervasive Patient Frame Leakage in Benchmark Splits**: Public releases of ultrasound datasets often partition image slices randomly into training, validation, and test splits without linking multiple frames acquired from the same patient study. This causes severe evaluation leakage where near-duplicate acoustic profiles, speckle signatures, and machine gain settings appear across train and test folds.

### 1.2 Research Hypothesis
> **Primary Hypothesis**: Non-generative masking (occlusion of clinician annotation regions with a neutral or zero constant value) matches or exceeds generative inpainting in downstream diagnostic discrimination (Macro-AUC, Balanced Accuracy) and probability calibration (Expected Calibration Error) when evaluated on truly independent, clean external ultrasound examinations, while eliminating synthetic pixel hallucination and dramatically reducing computational overhead.

*Status*: This is an empirical research hypothesis to test, **not** an established result.

### 1.3 Core Research Questions (RQs)
1. **RQ1 (Shortcut Learning Severity)**: What is the quantitative performance drop in diagnostic discrimination and calibration when classifiers trained on raw annotated scans (Arm A) are evaluated on out-of-distribution, clean external scans lacking annotations?
2. **RQ2 (Comparative Preprocessing Efficacy)**: Does generative inpainting (Arm B) provide any statistically significant diagnostic or calibration benefit over non-generative masking (Arm C) on clean external validation data?
3. **RQ3 (Acoustic Integrity & Texture Distortion)**: What measurable localized echotexture distortions, boundary artifacts, or variance shifts are introduced by generative inpainting into the lesion and its surrounding margin?
4. **RQ4 (Architectural Invariance)**: Are the comparative findings between Arm B and Arm C invariant across standard convolutional backbones exhibiting distinct inductive biases and receptive field hierarchies?
5. **RQ5 (Cross-Dataset Label Transferability)**: Can binary diagnostic risk models trained on MMOTU transfer reliably to multi-center external cohorts (OvaTUS-V1) with differing morphological classification systems and vendor hardware profiles?

### 1.4 Expected Scientific Contributions
- **Intended Contribution vs. Proven Result**:
  - *Intended*: To provide the first rigorous, reproducible, open data-centric benchmark comparing generative inpainting against non-generative occlusion across paired internal and external ultrasound cohorts with rigorous calibration and texture audit metrics.
  - *Current Status*: The project is in the foundation and infrastructure implementation phase. No experimental results, model weights, or clinical superiority are claimed or fabricated at this stage.

### 1.5 Scope Boundaries & Limitations
- **In Scope**: Binary classification (Benign vs. Malignant), standardized 3-arm image preprocessing, probability calibration, local texture variance audits, out-of-distribution evaluation.
- **Out of Scope**: 3-class Borderline Ovarian Tumor (BOT) prediction (statistical power calculations establish that BOT prevalence in MMOTU is insufficient for well-powered independent stratification); live hospital clinical deployment; real-time video stream processing.
- **Methodological Boundary**: We do not claim clinical tissue reconstruction fidelity from generative inpainters because true tissue under annotation artifacts is unobservable in retrospect.

---

## 2. Dataset Contracts

### 2.1 MMOTU (Multi-Modality Ovarian Tumor Ultrasound)
- **Role**: Primary dataset for internal training, cross-validation, and internal benchmark testing.
- **Source & Provenance**: Zhao et al., Beijing Shijitan Hospital, Capital Medical University. Published in IEEE ISBI 2022 / arXiv 2022. Distributed on Zenodo (DOI: 10.5281/zenodo.21128657) and Figshare (DOI: 10.6084/m9.figshare.25058690) under Creative Commons Attribution 4.0 / Apache 2.0.
- **Reported Volume**: 1,639 total ultrasound images:
  - `OTU_2D`: 1,469 conventional B-mode grayscale ultrasound images.
  - `OTU_CEUS`: 170 contrast-enhanced ultrasound images (excluded from initial 2D B-mode benchmark to preserve modality homogeneity).
- **Image Specifications**: Grayscale B-mode images, 2D PNG/JPG format, nominal resolutions ~400×800 pixels.
- **Labels**: Binary diagnostic class (0 = Benign, 1 = Malignant) and 8-class fine-grained histology.
- **Ground Truth Masks Available**: Pixel-wise binary tumor Region of Interest (ROI) segmentation masks outlining the mass boundary.
- **Critical Limitation & Blocker (Patient IDs)**: *Public releases of MMOTU completely lack patient identifiers, examination IDs, or scan timestamps.* Therefore, exact patient-level grouping is unavailable in the public files. Random train/test splits inevitably carry severe frame leakage risk. The pipeline must explicitly record that patient-level leakage cannot be ruled out on MMOTU, and must enforce near-duplicate image auditing.
- **Critical Limitation & Blocker (Annotation Masks)**: *Public MMOTU releases do not include pre-extracted binary artifact masks* (calipers/text). Wang et al. (MGGAN, 2023) created manual artifact masks for MGGAN training. These artifact masks must either be procured, extracted via an audited heuristic/caliper detection pipeline, or manually annotated. Under no circumstances may tumor ROI masks be used as artifact masks.
- **Status Summary**:
  - Image count and labels: *Sourced from literature; local verification pending data download.*
  - Patient identifiers: *Unresolved in public release; acknowledged as an inherent dataset limitation.*
  - Annotation artifact masks: *Awaiting verified extraction protocol or external mask source.*

### 2.2 OvaTUS-V1 (External Multi-Center Benchmark)
- **Role**: Primary external out-of-distribution test benchmark for evaluating generalization without annotation shortcut artifacts.
- **Source & Provenance**: National Hospital of Obstetrics and Gynecology (NHOG), Hanoi, Vietnam; published by Nguyen et al. (2025). Multi-center cohort collected on Samsung Medison W80i and GE Voluson S6 hardware platforms.
- **Expected Data**: 2D B-mode transvaginal ultrasound images, confirmed to have clean baseline scans or standardized symbol removal, with fine-grained morphological segmentation and pathology records.
- **Access & Licensing Requirements**: OvaTUS-V1 is governed by institutional data-use agreements and restricted access protocols. The dataset cannot be downloaded automatically without formal access approval.
- **Label Schema & Compatibility Contract**:
  - OvaTUS-V1 annotations provide six IOTA (International Ovarian Tumor Analysis) morphological categories: `Solid tumor`, `Multilocular cyst`, `Unilocular cyst`, `Dermoid cyst`, `Multilocular-solid cyst`, and `Unilocular-solid cyst`.
  - *Compatibility Blocker (BLK-03)*: These 6 classes represent acoustic morphologies with heterogeneous malignancy risk, **not** confirmed binary benign/malignant histopathology. Mapping classes such as `Multilocular-solid cyst` or `Solid tumor` to binary labels without surgical histopathological records is scientifically invalid.
  - *Resolution Gate*: Binary classification on OvaTUS-V1 is **strictly blocked** until patient-matched surgical pathology records are verified, or until an approved alternative evaluation protocol (e.g. morphological classification transfer) is formalized.
- **Status Summary**:
  - Availability: *Restricted access; access request procedure documented in project memory.*
  - Label mapping: *Unverified and blocked for binary CAD evaluation pending surgical histopathology records.*

---

## 3. Experimental Arms & Preprocessing Protocols

The benchmark compares three mutually exclusive image preprocessing conditions across identical training cohorts, splits, backbones, and optimization budgets:

```
                  ┌───────────────────────────────────────────────┐
                  │           Source Ultrasound Image             │
                  │        (e.g., MMOTU 2D B-Mode Scan)          │
                  └───────────────────────┬───────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
         [Artifact Mask Exists]                     [Clean / No Artifacts]
                    │                                           │
       ┌────────────┼────────────┐                              │
       ▼            ▼            ▼                              │
  ┌─────────┐  ┌─────────┐  ┌─────────┐                         │
  │  Arm A  │  │  Arm B  │  │  Arm C  │                         │
  │  (Raw)  │  │ (MGGAN) │  │(Masked) │                         │
  └────┬────┘  └────┬────┘  └────┬────┘                         │
       │            │            │                              │
       │            │            └─ Set artifact pixels to 0 ───┤
       │            └─ Run inpainting through verified MGGAN ───┤
       └────────────── Keep raw scan intact ────────────────────┤
                                                                ▼
                                                   Identical Training & Eval
```

### 3.1 Arm A — Raw Scans (Shortcut Baseline)
- **Input**: Original B-mode ultrasound image with all clinician measurement calipers, crosses, text labels, and overlay markers intact.
- **Purpose**: Establishes the baseline performance of standard models when shortcut cues are available during training and internal testing.

### 3.2 Arm B — Generative Inpainting (MGGAN / Verified Inpainter)
- **Input**: Image processed through a verified generative inpainter using the corresponding binary annotation artifact mask $M_{art}$.
- **Reference Model**: Mask-Guided GAN (MGGAN) utilizing Fast Fourier Convolutions (FFCs) and Squeeze-and-Excitation attention blocks (Wang et al., 2023), or an explicitly verified, open-source ultrasound inpainter (e.g., IOPaint/LaMa with documented configuration).
- **Constraint**: Inpainting is applied *only* within the region defined by $M_{art}$ dilated by a configurable kernel radius $r_{dil} \in [1, 3]$ pixels to prevent caliper boundary bleed. All exterior pixels $x \notin M_{art}$ must remain mathematically identical to the source image.
- **Provenance**: Model checkpoint hash, weights URL, generator configuration, and execution environment must be logged in the experiment manifest.

### 3.3 Arm C — Non-Generative Masking (Zero / Neutral Occlusion)
- **Input**: Image processed by setting all pixels inside the annotation artifact mask $M_{art}$ (dilated by $r_{dil}$) to a constant neutral value:
  $$I_{masked}(u, v) = \begin{cases} c_{neutral}, & \text{if } M_{art}(u, v) = 1 \\ I_{raw}(u, v), & \text{if } M_{art}(u, v) = 0 \end{cases}$$
- **Default Neutral Value**: $c_{neutral} = 0$ (black / zero-fill, matching standard ultrasound acoustic shadow background). Alternative configurable values: mean image intensity, median speckle value.
- **Clinical Advantage**: Modifies exactly zero unoccluded pixels; introduces zero synthetic morphological structures; costs zero inference computation.

### 3.4 Rigorous Rules Across All Arms
- **Cohort Traceability**: No image may be dropped from one arm without being identically accounted for in all other arms. If an image has no visible annotation artifacts ($M_{art} = \mathbf{0}$), it passes through all three arms unchanged ($I_A = I_B = I_C = I_{raw}$). The number of samples evaluated across arms must remain strictly identical.
- **Normalization**: After arm-specific preprocessing, all images undergo identical resizing ($256 \times 256$ or $224 \times 224$ with bicubic interpolation), identical grayscale-to-RGB 3-channel broadcast, and identical ImageNet or dataset-level mean/std z-score standardization.
- **Caching & Checksums**: Preprocessed images are cached with deterministic SHA-256 keys derived from `(source_checksum, mask_checksum, arm_id, preprocessing_config_hash)`.

---

## 4. Dataset Splitting, Leakage Prevention & Audit

### 4.1 Splitting Hierarchy
1. **Source-Aware Grouping**: Where patient, examination, or study identifiers are available, split allocation must strictly enforce Group K-Fold or Group Stratified splitting, ensuring zero cross-fold patient overlap.
2. **Public MMOTU Fallback Protocol**:
   - Because public MMOTU lacks patient IDs, the project must run a forensic **Image Hash & Near-Duplicate Audit** before splitting.
   - Algorithms:
     - Exact duplicate detection via SHA-256.
     - Perceptual near-duplicate detection via Difference Hash (dHash) and Perceptual Hash (pHash) with a Hamming distance threshold $d_H \le 4$.
   - Near-duplicate pairs must be clustered into composite pseudo-patient groups, preventing near-identical sweep frames from bridging training and test sets.
   - *Mandatory Disclosure*: Every report and manuscript output must state explicitly: *"In the absence of source hospital record identifiers, pseudo-patient clusters mitigate but cannot mathematically guarantee complete patient-level separation."*
3. **Split Proportions**:
   - Provisional Default: 70% Training, 15% Validation (for hyperparameter tuning and early stopping), 15% Internal Test.
   - External Cohort (OvaTUS-V1): 100% held-out independent test set; never seen during training, hyperparameter search, or temperature scaling.
4. **Reproducibility**: All splits are defined via persistent, version-controlled CSV split manifests (`data/manifests/split_v1.csv`) generated with fixed random seeds ($S \in \{42, 1337, 2026\}$).

---

## 5. Modeling Specifications

### 5.1 Architecture Selection
To guarantee that findings reflect data preprocessing dynamics rather than specific neural inductive biases, the evaluation benchmarks two complementary, representative convolutional backbone families:
1. **Modern Hierarchical Inverted Bottleneck CNN**: `ConvNeXt-Small` (primary approved benchmark architecture per Section 15.1 MPS design; `ConvNeXt-Tiny` permitted strictly for fast development smoke-tests).
2. **Compound-Scaled Efficient CNN**: `EfficientNet-B3` (primary approved benchmark architecture per Section 15.1 MPS design; `EfficientNet-B0` permitted strictly for fast development smoke-tests).

*Constraint*: Backbones are initialized from official ImageNet pre-trained weights. Architectural parameters remain identical across Arm A, Arm B, and Arm C.

### 5.2 Loss Function & Class Imbalance
Ovarian ultrasound cohorts exhibit inherent pathological class imbalances (typically 2:1 to 4:1 benign-to-malignant ratio). To counteract gradient domination by majority benign samples:
- **Primary Loss**: Class-Weighted Focal Loss (Lin et al.):
  $$\mathcal{L}_{focal}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  where $\gamma \in [1.0, 2.0]$ focuses learning on hard boundary samples, and $\alpha_t$ is inversely proportional to class frequency: $\alpha_c = \frac{N}{K \cdot N_c}$.
- **Configurable Fallback**: Class-Weighted Binary Cross-Entropy with label smoothing $\epsilon = 0.05$.

### 5.3 Optimization & Hyperparameter Budget
- **Optimizer**: AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $\lambda = 10^{-4}$) or SGD with Nesterov momentum ($0.9$, weight decay $10^{-4}$).
- **Learning Rate Schedule**: Cosine annealing with 5-epoch linear warmup. Initial learning rate $\eta_0 \in [10^{-4}, 5 \cdot 10^{-4}]$.
- **Batch Size**: 32 (or 16 on GPU memory-constrained environments).
- **Epoch Budget**: Maximum 50 epochs; early stopping with patience of 10 epochs monitored strictly on **Validation Macro-F1**.
- **Model Selection Rule**: The checkpoint with the highest Validation Macro-F1 is selected for final evaluation. The test fold is evaluated exactly once per trained model.

---

## 6. Evaluation Protocol & Metrics

### 6.1 Diagnostic Discrimination
- **Primary Metric**: Macro-AUC (Area Under the Receiver Operating Characteristic Curve).
- **Secondary Metrics**:
  - Balanced Accuracy: $\frac{1}{2} (\text{Sensitivity} + \text{Specificity})$
  - Sensitivity (Recall of Malignant class, positive class = 1): $\frac{TP}{TP + FN}$
  - Specificity (Recall of Benign class, negative class = 0): $\frac{TN}{TN + FP}$
  - Macro-F1 score: unweighted harmonic mean of precision and recall across classes.
- **Threshold Policy**: Classification metrics reported at standard threshold $\tau = 0.5$ and at the Youden-index optimal threshold $\tau^*$ derived solely from the validation split.

### 6.2 Clinical Reliability & Probability Calibration
Medical AI models must produce calibrated probabilities that represent true empirical likelihoods of malignancy:
- **Expected Calibration Error (ECE)**:
  $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
  evaluated with $M = 10$ and $M = 15$ equal-frequency and equal-width bins.
- **Maximum Calibration Error (MCE)**: $\max_{m \in \{1, \dots, M\}} |\text{acc}(B_m) - \text{conf}(B_m)|$.
- **Brier Score**: $\frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$.
- **Post-Hoc Temperature Scaling**:
  - Calibrated logit $\hat{z}_i = z_i / T$, where $T > 0$ is optimized via NLL minimization **strictly on the Validation split**.
  - *Hard Rule*: The external test set is never used to fit or tune the temperature parameter $T$.

---

## 7. Image-Integrity & Echotextural Distortion Analysis

### 7.1 Ground Truth Limitation & Reference Requirements
In medical ultrasound, the true anatomical tissue occluded by clinician calipers is unobservable. Comparing an inpainted image $I_B$ against the annotated raw image $I_A$ using full-image SSIM or LPIPS is methodologically deceptive: a higher similarity to $I_A$ rewards keeping the caliper artifacts!

To resolve this paradox, the evaluation protocol establishes a **Tri-Partition Echotextural Audit**:

```
 ┌───────────────────────────────────────────────────────────────┐
 │ Whole Ultrasound Scan (B-Mode)                                │
 │                                                               │
 │    ┌──────────────────────────────┐                           │
 │    │ Surrounding Unmasked Margin  │                           │
 │    │  ┌─────────────────────────┐ │                           │
 │    │  │ Inpainted / Masked      │ │                           │
 │    │  │ Caliper Region (M_art)  │ │                           │
 │    │  └─────────────────────────┘ │                           │
 │    └──────────────────────────────┘                           │
 └───────────────────────────────────────────────────────────────┘
```

1. **Exterior Conservation (Boundary Leakage Audit)**:
   - Evaluates whether inpainting alters pixels outside the annotation mask.
   - Metric: Peak Signal-to-Noise Ratio (PSNR) and Mean Absolute Error (MAE) computed strictly over unmasked coordinates $\{ (u, v) \mid M_{art}(u, v) = 0 \}$.
   - Criterion: Exterior PSNR should approach $\infty$ (or $>50 \text{ dB}$); any alteration of exterior tissue pixels indicates illegal generative spillover.
2. **Local Texture Variance (LTV) Inside Mask**:
   - Compares the variance of pixel intensities inside $M_{art}$ with the immediate surrounding unmasked diagnostic tissue band $\partial M_{art}$ (a 10-pixel morphological dilation dilation band):
     $$\Delta \text{LTV} = |\sigma^2(I(M_{art})) - \sigma^2(I(\partial M_{art}))|$$
   - Highlights unnatural smoothing or hyper-variance introduced by generative priors.
3. **Synthetic Caliper Insertion Experiment (Controlled Reconstruction Benchmark)**:
   - To establish a ground-truth reference, clean, unannotated external scans (or audited unannotated MMOTU scans) are corrupted with synthetic digital calipers matching clinical geometry (`src/ovaryvision/synthetic_artifacts.py`).
   - The generator synthesizes realistic crosshairs (plus/x markers), dashed connecting lines, and measurement callout text ("D1: 24.5mm") with mathematically verified containment ($I_{clean} \neq I_{corrupted} \implies M_{art} = 1$).
   - Reconstructed images (Arm B inpainting and Arm C zero-masking) are compared directly against the known, ground-truth clean reference using SSIM, PSNR, MAE, and $\Delta \text{LTV}$.
   - **Critical Scientific Limitations**:
     1. *Known Synthetic vs. Unknown Real Ground Truth*: The synthetic benchmark evaluates reconstruction of artificial geometric overlays on known clean reference pixels. In genuine clinical ultrasound, the true anatomical tissue beneath doctor-added calipers is unobservable.
     2. *Domain Shift*: High reconstruction fidelity (high PSNR/SSIM) on synthetic calipers demonstrates geometric filling ability, but does not prove the absence of clinical hallucinations (e.g., creating fake papillary projections) on genuine pathological lesions.
     3. *Software Tests vs. Scientific Evidence*: Synthetic test fixtures in automated test suites verify software execution and arithmetic correctness; they must never be presented as empirical clinical findings.

---

## 8. Statistical Analysis Plan

### 8.1 Hypotheses & Endpoints
- **Primary Endpoint**: Difference in Macro-AUC on the external test set between Arm C (Masking) and Arm B (Inpainting):
  $$\Delta \text{AUC}_{C-B} = \text{AUC}_{\text{Arm C}} - \text{AUC}_{\text{Arm B}}$$
- **Equivalence / Non-Inferiority Margin**: Provisional non-inferiority margin established at $\delta = -0.02$ AUC. Because no definitive clinical consensus exists for inpainting non-inferiority margins, a mandatory sensitivity analysis across $\delta \in \{-0.01, -0.02, -0.03\}$ must be reported. If the lower bound of the 95% Confidence Interval for $\Delta \text{AUC}_{C-B}$ exceeds $\delta$, Arm C is declared non-inferior to Arm B. If the lower bound exceeds $0.0$, Arm C is declared superior.
- **Secondary Endpoints**:
  - $\Delta \text{ECE}_{C-B}$ (difference in Expected Calibration Error).
  - Paired accuracy differences across individual image predictions.
  - Computational throughput (frames per second, preprocessing latency in ms).

### 8.2 Statistical Testing Procedures
- **Primary Statistical Estimator**: 1,000-iteration stratified bootstrap paired difference confidence intervals (BCa / percentile method) for $\Delta \text{AUC}_{C-B}$ and calibration metrics.
- **Paired AUC Comparisons**: DeLong's test for paired ROC curves evaluated on the identical test set as a secondary asymptotic check.
- **Paired Classification Decisions**: McNemar's test on binary 0/1 outcome concordance matrices at fixed and Youden-optimal thresholds.
- **Multi-Seed Variance**: All training experiments executed across 3 fixed random seeds ($42, 1337, 2026$) minimum, with 5 seeds recommended if compute budget allows. Mean and standard deviations reported.
- **Reporting Rule for Null Results**: If no statistically significant difference is observed between Arm B and Arm C, the conclusion must state honestly that generative inpainting provided no measurable benefit over simple masking, confirming the non-generative hypothesis without post-hoc rationalization.

---

## 9. Compute & Execution Infrastructure

### 9.1 Environment Division of Labor
- **OCI Host (ARM CPU-only Instance)**:
  - Repository orchestration, configuration management, directory setup, unit testing.
  - Dataset inspection, manifest generation, SHA-256 auditing, near-duplicate detection.
  - Experiment tracking, documentation maintenance, and report generation.
  - *Hard Rule*: Large deep learning training loops and batch generative inpainting are never executed on this CPU-only host.
- **External GPU Compute (Google Colab / Kaggle GPU Runtime)**:
  - Deep learning model training (ConvNeXt, EfficientNet).
  - Batch MGGAN / inpainting execution.
  - Post-hoc temperature scaling and bootstrap metric calculation.
  - Generates standardized prediction JSONs, metric CSVs, and checkpoint weights, synchronized back to OCI repository via documented manifest protocols.

### 9.2 Reproducibility & Security Gate
- Configuration files are strictly decoupled from code (`configs/default.yaml`).
- All runs generate a deterministic experiment hash logging: git commit hash, configuration checksum, python environment manifest, hardware metadata, and timestamp.
- No medical imaging data or credentials may be committed to version control.

---

## 10. Risks, Blockers & Acceptance Criteria

### 10.1 Identified Blockers & Scientific Risks
| Risk / Blocker ID | Description | Severity | Mitigation Strategy |
|---|---|---|---|
| **BLK-01** | Public MMOTU lacks patient IDs (frame leakage risk) | Critical | Execute exact & near-duplicate perceptual hash audit. Cluster duplicate frames into pseudo-patient groups. Disclose limitation in all publications. |
| **BLK-02** | MMOTU public release lacks pre-extracted artifact masks | High | Implement heuristic caliper detection tool or procure MGGAN author masks; audit masks before running Arm B/C. Separate tumor masks from artifact masks. |
| **BLK-03** | OvaTUS-V1 access is restricted and requires formal approval | High | Prepare formal access application; formulate exact label-compatibility mapping in project memory; run pipeline on internal split first. |
| **BLK-04** | Transient GPU disconnections on free notebook runtimes | Medium | Implement epoch-level checkpoint saving and stateful resume flags in GPU training scripts. |

### 10.2 Project Acceptance Criteria (Milestone 1)
- [x] Canonical `SPEC.md` rewritten with complete research framing, dataset contracts, and statistical rigor.
- [x] Persistent operating documentation initialized (`AGENT.md`, `PROJECT_STATE.md`, `DECISIONS.md`, `IMPLEMENTATION_PLAN.md`, `CHANGELOG.md`, `README.md`).
- [ ] Operational docs created (`docs/RESEARCH_PROTOCOL.md`, `docs/DATA_DICTIONARY.md`, `docs/EXPERIMENT_TRACKER.md`).
- [ ] Initial Python package (`src/ovaryvision/`) created with CLI, configuration loader, environment diagnostics, logging, manifests, provenance, and data audit tool.
- [ ] Automated tests implemented covering all core modules, passing with 100% test success on OCI host.
- [ ] Graceful failure verified for absent datasets.
- [ ] Zero unverified claims, fabricated data, or artificial benchmark numbers.
