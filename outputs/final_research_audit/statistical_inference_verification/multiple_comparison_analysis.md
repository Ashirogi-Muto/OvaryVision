# Multiple Comparison Analysis & Multiplicity Control (Revised Final Report)

**Audit Date**: 2026-11-02  
**Dataset**: OvaryVision 3-Seed Pooled Segmentation Benchmark (469 Scans, 454 Pseudo-Patient Clusters)  
**Verification Script**: `scripts/verify_statistical_inference.py`  
**Output Tables**:
- Full 21 comparisons: `recalculated_inferential_results.csv` and `manuscript_table_full_21_comparisons.csv`
- Primary 7-test family: `manuscript_table_primary_family.csv`

---

## 1. Multiple Testing Framework & Mathematical Protocol

Testing multiple endpoints across multiple preprocessing arms inflates the Family-Wise Error Rate (FWER) if unadjusted $p$-values are interpreted in isolation.

### 1.1 Decision Rules and Equivalence Protocol
To prevent errors in statistical reporting:
1. **Rule A (Adjusted $P$-Value Comparison)**:
   The Bonferroni-adjusted $p$-value $p_{\text{adj}} = \min(1.0, M \times p_{\text{raw}})$ is compared directly against the family-wise significance threshold:
   $$\text{Reject } H_0 \iff p_{\text{adj}} < \alpha \quad (\alpha = 0.05)$$
2. **Rule B (Raw $P$-Value Comparison)**:
   Alternatively, the raw unadjusted $p$-value $p_{\text{raw}}$ is compared against the adjusted significance threshold:
   $$\text{Reject } H_0 \iff p_{\text{raw}} < \frac{\alpha}{M}$$
3. **CRITICAL VIOLATION PROHIBITION**:
   **Never compare an adjusted $p$-value against $\alpha / M$.** Comparing $p_{\text{adj}}$ against $\alpha / M$ erroneously double-counts the multiplicity penalty by $M^2$.
   Rules A and B are mathematically identical:
   $$p_{\text{adj}} < 0.05 \iff M \times p_{\text{raw}} < 0.05 \iff p_{\text{raw}} < \frac{0.05}{M}$$

Every adjusted $p$-value reported below has been verified arithmetically against this equivalence.

---

## 2. Hypothesis Families & Hierarchical Tiering

### 2.1 Pre-Registered Primary Confirmatory Family ($M = 7$)
The primary research objective of OvaryVision (SPEC.md, Decision D01) is the direct comparison between **Non-Generative Masking (Arm C)** and **Harmonic Inpainting (Arm B)** across the 7 prespecified endpoints:
1. Tumour DSC ($N=382$)
2. Tumour IoU ($N=382$)
3. Tumour Boundary IoU ($N=382$)
4. Lesion Preservation Ratio (LPR, $N=193$)
5. Normal Ovary Clean Rejection Rate ($N=87$)
6. Normal Ovary Intra-Ovarian FPR ($N=87$)
7. Normal Ovary Extra-Ovarian FPR ($N=87$)

- **Family-wise error threshold**: $\alpha = 0.05$
- **Per-test raw significance threshold**: $\alpha_{\text{primary}} = \frac{0.05}{7} \approx 0.007143$

### 2.2 Full Benchmark Secondary & Exploratory Family ($M = 21$)
Encompasses all 3 pairwise comparisons (Arm C vs Arm B, Arm C vs Arm A, Arm B vs Arm A) across all 7 metrics:
- Comparisons involving Arm A (Raw) represent secondary benchmark checks against un-preprocessed inputs.
- **Family-wise error threshold**: $\alpha = 0.05$
- **Per-test raw significance threshold**: $\alpha_{\text{full}} = \frac{0.05}{21} \approx 0.002381$

---

## 3. Disentangling Uncertainty: Scan-Level Bootstrap vs Training-Seed Variability

A rigorous statistical analysis of deep learning models must separate two distinct sources of variance:
1. **Scan-Level Bootstrap Uncertainty ($\text{SE}_{\text{jack}}$ / BCa 95% CI)**:
   Reflects sampling variability across the finite test cohort of clinical ultrasound scans ($N=469$, grouped into $K=454$ pseudo-patient clusters). This captures patient-to-patient anatomical and acoustic variation.
2. **Training-Seed Variability ($s_{\text{seed}}$)**:
   Reflects algorithmic / optimization stochasticity across different weight initializations and data loader shuffle orders during 40 epochs of training (evaluated across Seeds 42, 1337, and 2026).
   $$s_{\text{seed}} = \sqrt{\frac{1}{2} \sum_{s \in \{42, 1337, 2026\}} (\Delta_s - \bar{\Delta})^2}$$

---

## 4. Primary Confirmatory Family Table (Arm C vs Arm B, $M = 7$)

| # | Metric | Cohort | Observed Diff ($\bar{\Delta}$) | 95% BCa CI | Seed 42 Diff | Seed 1337 Diff | Seed 2026 Diff | Seed SD ($s_{\text{seed}}$) | Jackknife SE | Raw BCa $p$ | Display $p_{\text{raw}}$ | Bonf $p_{\text{adj}}$ ($M=7$) | FWER Decision ($\alpha=0.05$) | Confirmatory Verdict |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **1** | **dsc** | Tumour (382) | $+0.0019$ | $[-0.0070, +0.0131]$ | $+0.0240$ | $-0.0145$ | $-0.0037$ | $0.0199$ | $0.0052$ | $0.6551$ | $0.6551$ | $1.0000$ | $p_{\text{adj}} \ge 0.05$ | Non-significant |
| **2** | **iou** | Tumour (382) | $-0.0015$ | $[-0.0105, +0.0091]$ | $+0.0158$ | $-0.0213$ | $+0.0009$ | $0.0187$ | $0.0051$ | $0.8185$ | $0.8185$ | $1.0000$ | $p_{\text{adj}} \ge 0.05$ | Non-significant |
| **3** | **biou_d2** | Tumour (382) | $+0.0013$ | $[-0.0008, +0.0036]$ | $+0.0006$ | $-0.0012$ | $+0.0045$ | $0.0029$ | $0.0011$ | $0.2397$ | $0.2397$ | $1.0000$ | $p_{\text{adj}} \ge 0.05$ | Non-significant |
| **4** | **lpr** | Overlap (193) | $\mathbf{-0.0710}$ | $\mathbf{[-0.0969, -0.0460]}$ | $-0.0742$ | $-0.0195$ | $-0.1194$ | $0.0500$ | $0.0136$ | $0.0011$ | $0.0011$ | $\mathbf{0.0078}$ | $\mathbf{p_{\text{adj}} < 0.05}$ | **Confirmed Significant** |
| **5** | **clean_rej** | Normal (87) | $-0.0805$ | $[-0.1515, -0.0152]$ | $-0.3103$ | $-0.1264$ | $+0.1954$ | $0.2560$ | $0.0351$ | $0.0185$ | $0.0185$ | $0.1298$ | $p_{\text{adj}} \ge 0.05$ | Secondary Nominally Sig. |
| **6** | **fpr_intra** | Normal (87) | $\mathbf{+0.0914}$ | $\mathbf{[+0.0532, +0.1382]}$ | $+0.2191$ | $+0.1191$ | $-0.0640$ | $0.1435$ | $0.0220$ | $0.0005$ | $\mathbf{< 0.001}$ | $\mathbf{0.0035}$ | $\mathbf{p_{\text{adj}} < 0.05}$ | **Confirmed Significant** |
| **7** | **fpr_extra** | Normal (87) | $-0.0003$ | $[-0.0030, +0.0023]$ | $+0.0014$ | $-0.0022$ | $-0.0002$ | $0.0018$ | $0.0014$ | $0.8316$ | $0.8316$ | $1.0000$ | $p_{\text{adj}} \ge 0.05$ | Non-significant |

### Primary Findings Summary:
1. **LPR Impairment Confirmed**: Non-generative masking (Arm C) significantly degrades in-caliper lesion sensitivity compared to inpainting (Arm B) by an estimated $7.10$ percentage points ($p_{\text{raw}} = 0.0011 < 0.00714 \iff p_{\text{adj}} = \mathbf{0.0078} < 0.05$).
2. **Intra-Ovarian False Positives Confirmed**: Arm C significantly elevates false-positive area on normal ovaries compared to Arm B by an estimated $9.14$ percentage points ($p_{\text{raw}} < 0.0005 < 0.00714 \iff p_{\text{adj}} = \mathbf{0.0035} < 0.05$).
3. **Clean Rejection Rate Drop Fails FWER Control**: Although nominally significant at $\alpha = 0.05$ ($p_{\text{raw}} = 0.0185$), the adjusted $p_{\text{adj}} = 0.1298 > 0.05$. This reflects substantial seed-to-seed variability ($s_{\text{seed}} = 0.2560$), as Seed 2026 showed higher clean rejection for Arm C while Seeds 42 and 1337 favoured Arm B.

---

## 5. Full Benchmark Table (All 21 Comparisons)

| # | Comparison | Metric | Cohort | Obs Diff ($\bar{\Delta}$) | 95% BCa CI | Seed SD ($s_{\text{seed}}$) | Jackknife SE | Raw BCa $p$ | Display $p_{\text{raw}}$ | Null-Centered $p$ | Bonf $p_{\text{adj}}$ ($M=21$) | Full FWER ($\alpha=0.05$) | Tier & Verdict |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **1** | **Arm C vs Arm B** | **dsc** | Tumour (382) | $+0.0019$ | $[-0.0070, +0.0131]$ | $0.0199$ | $0.0052$ | $0.6551$ | $0.6551$ | $0.7196$ | $1.0000$ | Non-significant | Primary Endpoint: Equivalence |
| **2** | **Arm C vs Arm B** | **iou** | Tumour (382) | $-0.0015$ | $[-0.0105, +0.0091]$ | $0.0187$ | $0.0051$ | $0.8185$ | $0.8185$ | $0.7736$ | $1.0000$ | Non-significant | Primary Endpoint: Equivalence |
| **3** | **Arm C vs Arm B** | **biou_d2** | Tumour (382) | $+0.0013$ | $[-0.0008, +0.0036]$ | $0.0029$ | $0.0011$ | $0.2397$ | $0.2397$ | $0.2559$ | $1.0000$ | Non-significant | Primary Endpoint: Equivalence |
| **4** | **Arm C vs Arm B** | **lpr** | Overlap (193) | $\mathbf{-0.0710}$ | $\mathbf{[-0.0969, -0.0460]}$ | $0.0500$ | $0.0136$ | $0.0011$ | $0.0011$ | $<0.001$ | $\mathbf{0.0233}$ | **Significant ($p_{\text{adj}} < 0.05$)** | **Primary Confirmatory: Deficit** |
| **5** | **Arm C vs Arm B** | **clean_rej** | Normal (87) | $-0.0805$ | $[-0.1515, -0.0152]$ | $0.2560$ | $0.0351$ | $0.0185$ | $0.0185$ | $0.0275$ | $0.3893$ | Non-significant | Secondary Nominal Only |
| **6** | **Arm C vs Arm B** | **fpr_intra** | Normal (87) | $\mathbf{+0.0914}$ | $\mathbf{[+0.0532, +0.1382]}$ | $0.1435$ | $0.0220$ | $0.0005$ | $<0.001$ | $<0.001$ | $\mathbf{0.0105}$ | **Significant ($p_{\text{adj}} < 0.05$)** | **Primary Confirmatory: Deficit** |
| **7** | **Arm C vs Arm B** | **fpr_extra** | Normal (87) | $-0.0003$ | $[-0.0030, +0.0023]$ | $0.0018$ | $0.0014$ | $0.8316$ | $0.8316$ | $0.8001$ | $1.0000$ | Non-significant | Primary Endpoint: Equivalence |
| **8** | **Arm C vs Arm A** | **dsc** | Tumour (382) | $+0.0118$ | $[+0.0009, +0.0240]$ | $0.0116$ | $0.0059$ | $0.0354$ | $0.0354$ | $0.0460$ | $0.7435$ | Non-significant | Exploratory Nominal Only |
| **9** | **Arm C vs Arm A** | **iou** | Tumour (382) | $+0.0091$ | $[-0.0022, +0.0206]$ | $0.0107$ | $0.0057$ | $0.1075$ | $0.1075$ | $0.1119$ | $1.0000$ | Non-significant | Exploratory Non-significant |
| **10** | **Arm C vs Arm A** | **biou_d2** | Tumour (382) | $+0.0020$ | $[-0.0003, +0.0042]$ | $0.0036$ | $0.0012$ | $0.0793$ | $0.0793$ | $0.0845$ | $1.0000$ | Non-significant | Exploratory Non-significant |
| **11** | **Arm C vs Arm A** | **lpr** | Overlap (193) | $-0.0232$ | $[-0.0512, +0.0070]$ | $0.0237$ | $0.0149$ | $0.1328$ | $0.1328$ | $0.1179$ | $1.0000$ | Non-significant | Exploratory Non-significant |
| **12** | **Arm C vs Arm A** | **clean_rej** | Normal (87) | $\mathbf{-0.1149}$ | $\mathbf{[-0.1777, -0.0588]}$ | $0.2193$ | $0.0296$ | $0.0007$ | $0.0007$ | $<0.001$ | $\mathbf{0.0151}$ | **Significant ($p_{\text{adj}} < 0.05$)** | **Secondary Benchmark: Deficit** |
| **13** | **Arm C vs Arm A** | **fpr_intra** | Normal (87) | $\mathbf{+0.0824}$ | $\mathbf{[+0.0522, +0.1240]}$ | $0.1492$ | $0.0181$ | $0.0005$ | $<0.001$ | $<0.001$ | $\mathbf{0.0105}$ | **Significant ($p_{\text{adj}} < 0.05$)** | **Secondary Benchmark: Deficit** |
| **14** | **Arm C vs Arm A** | **fpr_extra** | Normal (87) | $+0.0025$ | $[+0.0003, +0.0059]$ | $0.0038$ | $0.0013$ | $0.0205$ | $0.0205$ | $0.0560$ | $0.4311$ | Non-significant | Exploratory Nominal Only |
| **15** | **Arm B vs Arm A** | **dsc** | Tumour (382) | $+0.0099$ | $[-0.0009, +0.0215]$ | $0.0115$ | $0.0056$ | $0.0816$ | $0.0816$ | $0.0865$ | $1.0000$ | Non-significant | Exploratory Non-significant |
| **16** | **Arm B vs Arm A** | **iou** | Tumour (382) | $+0.0106$ | $[+0.0002, +0.0219]$ | $0.0136$ | $0.0055$ | $0.0495$ | $0.0495$ | $0.0595$ | $1.0000$ | Non-significant | Exploratory Nominal Only |
| **17** | **Arm B vs Arm A** | **biou_d2** | Tumour (382) | $+0.0007$ | $[-0.0014, +0.0026]$ | $0.0038$ | $0.0010$ | $0.5590$ | $0.5590$ | $0.4943$ | $1.0000$ | Non-significant | Exploratory Non-significant |
| **18** | **Arm B vs Arm A** | **lpr** | Overlap (193) | $\mathbf{+0.0478}$ | $\mathbf{[+0.0216, +0.0763]}$ | $0.0694$ | $0.0140$ | $0.0015$ | $0.0015$ | $0.0015$ | $\mathbf{0.0325}$ | **Significant ($p_{\text{adj}} < 0.05$)** | **Secondary Benchmark: Benefit** |
| **19** | **Arm B vs Arm A** | **clean_rej** | Normal (87) | $-0.0345$ | $[-0.0996, +0.0275]$ | $0.0398$ | $0.0331$ | $0.2908$ | $0.2908$ | $0.3103$ | $1.0000$ | Non-significant | Exploratory Non-significant |
| **20** | **Arm B vs Arm A** | **fpr_intra** | Normal (87) | $-0.0090$ | $[-0.0445, +0.0267]$ | $0.0058$ | $0.0183$ | $0.6327$ | $0.6327$ | $0.6037$ | $1.0000$ | Non-significant | Exploratory Non-significant |
| **21** | **Arm B vs Arm A** | **fpr_extra** | Normal (87) | $\mathbf{+0.0029}$ | $\mathbf{[+0.0012, +0.0050]}$ | $0.0021$ | $0.0010$ | $0.0005$ | $<0.001$ | $0.0045$ | $\mathbf{0.0105}$ | **Significant ($p_{\text{adj}} < 0.05$)** | **Secondary Benchmark: Shift** |

---

## 6. Synthesis of Multiplicity Survival

### 6.1 Surviving Comparisons ($p_{\text{adj}} < 0.05 \iff p_{\text{raw}} < \alpha / M$)
Exactly 6 of 21 comparisons survive full FWER control across the benchmark:
1. **Arm C vs Arm B on LPR** ($\Delta = -0.0710$, $p_{\text{raw}} = 0.0011 < 0.00238 \iff p_{\text{adj}} = 0.0233 < 0.05$)
2. **Arm C vs Arm B on Intra FPR** ($\Delta = +0.0914$, $p_{\text{raw}} < 0.0005 < 0.00238 \iff p_{\text{adj}} = 0.0105 < 0.05$)
3. **Arm C vs Arm A on Clean Rejection** ($\Delta = -0.1149$, $p_{\text{raw}} = 0.0007 < 0.00238 \iff p_{\text{adj}} = 0.0151 < 0.05$)
4. **Arm C vs Arm A on Intra FPR** ($\Delta = +0.0824$, $p_{\text{raw}} < 0.0005 < 0.00238 \iff p_{\text{adj}} = 0.0105 < 0.05$)
5. **Arm B vs Arm A on LPR** ($\Delta = +0.0478$, $p_{\text{raw}} = 0.0015 < 0.00238 \iff p_{\text{adj}} = 0.0325 < 0.05$)
6. **Arm B vs Arm A on Extra FPR** ($\Delta = +0.0029$, $p_{\text{raw}} < 0.0005 < 0.00238 \iff p_{\text{adj}} = 0.0105 < 0.05$)

### 6.2 Non-Surviving Nominally Significant Comparisons ($p_{\text{raw}} < 0.05$, but $p_{\text{adj}} \ge 0.05$)
The following comparisons are nominally significant in unadjusted terms, but fail full FWER control and must be reported as secondary or exploratory observations:
- **Arm C vs Arm B on Clean Rejection**: $p_{\text{raw}} = 0.0185$, $p_{\text{adj}, 7} = 0.1298$, $p_{\text{adj}, 21} = 0.3893$.
- **Arm C vs Arm A on Tumour DSC**: $p_{\text{raw}} = 0.0354$, $p_{\text{adj}, 21} = 0.7435$.
- **Arm C vs Arm A on Extra FPR**: $p_{\text{raw}} = 0.0205$, $p_{\text{adj}, 21} = 0.4311$.
- **Arm B vs Arm A on Tumour IoU**: $p_{\text{raw}} = 0.0495$, $p_{\text{adj}, 21} = 1.0000$.
