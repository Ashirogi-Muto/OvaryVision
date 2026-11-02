# Final Scientific Limitations Register: OvaryVision

**Date**: 2026-11-01  
**Auditor**: Antigravity (Advanced Agentic Coding / DeepMind)  
**Applicability**: Mandatory disclosure in all academic manuscripts, conference submissions, thesis chapters, and public repositories arising from the OvaryVision project.

---

## 1. Dataset & Ground Truth Limitations

### 1.1 Pseudo-Patient Clusters vs True Patient Independence (MMOTU)
- **Constraint**: Public MMOTU releases completely lack patient, examination, and acquisition metadata.
- **Mitigation**: Perceptual difference hashing (dHash $\le 4$) was employed to identify and cluster visually near-identical sweep frames into pseudo-patient groups (e.g., 82 multi-frame clusters in the test set).
- **Hard Limitation**: **dHash clustering must never be claimed to prove patient independence**. Ultrasound frames from the same patient acquired under differing probe orientations, acoustic gains, transducer frequencies, or focal depths can exhibit Hamming distances exceeding $4$, potentially placing frames from the same patient across train and test partitions. True patient-level separation cannot be mathematically verified on public MMOTU.

### 1.2 Annotation Artifact Mask Provenance
- **Constraint**: As proven in `docs/MMOTU_DATASET_COMPLETENESS_AUDIT.md`, radiologist annotation artifact masks ($M_{art}$) do not exist in any public release of MMOTU (they were burned into scan pixels during clinical exams).
- **Mitigation**: An automated morphological extraction pipeline ($V \ge 220$, saturation $\le 30$, connected component area $\le 3,500$ px, dilation $r=2$ px) was validated.
- **Hard Limitation**: Inpainting and zero-masking are conditioned on algorithmic artifact masks. Minor false-positive detections on hyperechoic pelvic structures (e.g., calcifications) or missed sub-threshold text characters cannot be ruled out.

### 1.3 External Cohort Ground Truth (OvaTUS-V1)
- **Constraint**: Authoritative documentation confirms that OvaTUS-V1 annotations represent **six IOTA morphological categories**, not verified histopathological diagnoses.
- **Hard Limitation**: Direct binary CAD evaluation on OvaTUS-V1 is **strictly blocked** (Decision D09, BLK-03). Claims of cross-center histological generalization remain unproven until surgical pathology records are audited.

---

## 2. Statistical & Inferential Limitations

### 2.1 Absence of Significance $\neq$ Statistical Equivalence
- **Statement**: On the tumour cohort ($N=382$), the observed difference in Dice Similarity Coefficient between Arm C (Zero-Masked) and Arm B (Harmonic Inpainting) was $\Delta \text{DSC} = +0.0019$ (+0.19%), with a 95% BCa CI of $[-0.0070, +0.0131]$ and $p = 0.776$.
- **Correction**: This finding demonstrates **non-significance**, NOT statistical equivalence. Establishing formal equivalence requires a two-one-sided test (TOST) against a prespecified equivalence margin $\delta$ (e.g., $\delta = \pm 0.02$). Because a formal segmentation equivalence margin was not prespecified in the protocol, claims of "proven equivalence" are invalid.

### 2.2 Multiple Comparison Multiplicity
- **Constraint**: The final aggregated report evaluates 7 metrics across 3 pairwise arm comparisons ($7 \times 3 = 21$ hypothesis tests).
- **Nominal vs Adjusted $p$-values**:
  - The reported $p$-values are unadjusted nominal values.
  - Applying Bonferroni family-wise error correction ($\alpha = 0.05 / 21 \approx 0.00238$):
    - $\Delta \text{LPR}_{C-B} = -0.0710$ ($p = 0.0005$) **remains statistically significant** ($p < 0.00238$).
    - $\Delta \text{Intra FPR}_{C-B} = +0.0914$ ($p = 0.0005$) **remains statistically significant** ($p < 0.00238$).
    - $\Delta \text{Clean Rejection}_{C-B} = -0.0805$ ($p = 0.022$) **does NOT survive Bonferroni correction** ($0.022 > 0.00238$).
    - $\Delta \text{DSC}_{C-A} = +0.0118$ ($p = 0.042$) **does NOT survive Bonferroni correction** ($0.042 > 0.00238$).
  - Primary conclusions must focus on the primary comparison (Arm C vs Arm B) and highlight multiple-testing caveats for secondary comparisons.

### 2.3 Interpretation of Scan-Level Averaging
- **Constraint**: Multi-seed pooling was performed by taking scan-level means across the three seeds $\bar{Y}_i = \frac{1}{3} \sum_{s} Y_{i, s}$.
- **Hard Limitation**: Paired cluster bootstrap resamples patient scans/clusters $i$, holding the three trained models fixed. The confidence intervals quantify **patient cohort sampling variability**, NOT the posterior predictive distribution of training a newly seeded model from scratch.

### 2.4 Training-Seed Variability ($N=3$ Seeds)
- **Constraint**: Three random seeds (42, 1337, 2026) were benchmarked.
- **Hard Limitation**: A sample size of $N=3$ training seeds is insufficient for asymptotic inferential tests of seed variability. Standard deviations across seeds must be reported descriptively (as in `seed_variability_summary.csv`), without claiming statistical proof of optimizer stability.

### 2.5 Prohibition of Unsubstantiated "Clinical Significance"
- Claims of "clinical significance" or "clinically acceptable margins" are prohibited unless mapped directly to a literature-grounded clinical decision threshold (e.g., maximum allowable false-positive cyst detection rate in ultrasound triage). Observed differences must be reported as **statistically detectable empirical shifts**.

---

## 3. Separation of Observed Findings vs Mechanistic Hypotheses

| Feature | Observed Empirical Fact | Mechanistic Hypothesis (Requires Further Study) |
|---|---|---|
| **LPR Degradation** | Zero-masking reduces Lesion Preservation Ratio by $7.10\%$ ($p = 0.0005$) inside caliper zones ($N=193$). | Hypothesis: The network's convolutional kernels suffer localized feature map zero-dropout, preventing spatial interpolation across dark bands. |
| **Normal Ovary False Positives** | Zero-masking increases intra-ovarian FPR by $+9.14\%$ ($p = 0.0005$) on normal ovaries ($N=87$). | Hypothesis: Dark zero-intensity patches resemble fluid-filled (anechoic) cystic lesions, tricking cyst-detection feature maps into firing. |
| **Inpainting Stability** | Inpainting maintains low cross-seed variability ($CV = 3.69\%$ on clean rejection) compared to masking ($CV = 55.17\%$). | Hypothesis: Texture continuity regularizes the loss landscape across stochastic mini-batch trajectories. |

