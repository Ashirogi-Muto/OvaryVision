# Final Scientific Claims Register: OvaryVision Segmentation Benchmark

**Date**: 2026-11-02  
**Scope**: Publication-Facing Claims, Statistical Backing, Approved Scientific Phrasing, and Prohibited Terminology  
**Reference Tables**:
- [`corrected_manuscript_table_primary_family.csv`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/corrected_manuscript_table_primary_family.csv)
- [`corrected_manuscript_table_full_21_comparisons.csv`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/corrected_manuscript_table_full_21_comparisons.csv)

---

## 1. Registry of Approved and Prohibited Phrasing

| Claim ID | Scientific Domain / Comparison | Approved Manuscript Phrasing | Prohibited Phrasing | Statistical Evidence | Scope & Estimand Conditions | Multiplicity Survival |
|:---:|---|---|---|---|---|:---:|
| **CLM-01** | **Tumour Delineation** (Arm C vs Arm B on DSC, IoU, Boundary IoU) | "No statistically significant difference in tumour delineation was detected between non-generative masking and harmonic inpainting; formal equivalence was not tested." | ❌ "Equivalence confirmed"<br>❌ "Models are equivalent"<br>❌ "Proved non-inferior" | - $\Delta \text{DSC} = +0.0019$ ($[-0.0070, +0.0131]$, $p_{\text{raw}} = 0.6551$, $p_{\text{adj}} = 1.0000$)<br>- $\Delta \text{IoU} = -0.0015$ ($[-0.0105, +0.0091]$, $p_{\text{raw}} = 0.8185$, $p_{\text{adj}} = 1.0000$)<br>- $\Delta \text{BIoU} = +0.0013$ ($[-0.0008, +0.0036]$, $p_{\text{raw}} = 0.2397$, $p_{\text{adj}} = 1.0000$) | Three-seed scan-level mean metric conditional on the three fixed trained checkpoints across 382 test tumour scans. | Non-significant before and after correction ($p_{\text{adj}} = 1.0$) |
| **CLM-02** | **In-Caliper Lesion Sensitivity** (Arm C vs Arm B on LPR) | "Non-generative masking resulted in a statistically significant deficit in lesion preservation within measurement caliper zones compared to harmonic inpainting, reducing sensitivity by an estimated 7.10 percentage points." | ❌ "Ensemble predictor dropped LPR"<br>❌ "Proves loss of anatomical continuity" (LPR is localized sensitivity, not bridging) | - $\Delta \text{LPR} = -0.0710$<br>- 95% BCa CI: $[-0.0969, -0.0460]$<br>- Jackknife SE: $0.0136$<br>- Seed SD: $0.0500$<br>- Raw BCa $p = 0.0011$<br>- Null-centered $p < 0.001$<br>- Primary Bonferroni $p_{\text{adj}} = 0.0078$<br>- Full Bonferroni $p_{\text{adj}} = 0.0233$ | Conditioned strictly on the 193 scans where caliper artifacts overlap ground-truth tumour tissue ($|\Omega_{\text{overlap}}| > 0$). Zero-overlap cases are excluded without imputation. | **Survives Primary ($M=7$) and Full ($M=21$) Bonferroni FWER Control ($p_{\text{adj}} < 0.05$)** |
| **CLM-03** | **Normal Ovary False-Positive Inflation** (Arm C vs Arm B on Intra-Ovarian FPR) | "The three-seed scan-level mean showed higher intra-ovarian FPR under zero-masking (+0.0914), but the direction varied across training seeds (Seed 42: +0.2191, Seed 1337: +0.1191, Seed 2026: -0.0640), indicating sensitivity to model initialization." | ❌ "Intra-ovarian FPR inflation is consistent across all three seeds"<br>❌ "Robust across all seeds"<br>❌ "Ensemble elevated false positives" | - $\Delta \text{FPR}_{\text{intra}} = +0.0914$<br>- 95% BCa CI: $[+0.0532, +0.1382]$<br>- Jackknife SE: $0.0220$<br>- Seed SD: $0.1435$<br>- Raw BCa $p < 0.001$<br>- Null-centered $p < 0.001$<br>- Primary Bonferroni $p_{\text{adj}} = 0.0035$<br>- Full Bonferroni $p_{\text{adj}} = 0.0105$ | Conditioned strictly on the 87 held-out normal ovary test scans. Seed-averaged scan-level mean metric differs from individual seed behavior; Seed 2026 reverses direction. | **Survives Primary ($M=7$) and Full ($M=21$) Bonferroni FWER Control ($p_{\text{adj}} < 0.05$) on pooled estimand** |
| **CLM-04** | **Normal Ovary Clean Rejection Rate** (Arm C vs Arm B on Clean Rejection) | "Non-generative masking showed a nominally significant reduction in the proportion of normal ovary scans correctly rejected with zero false-positive tumour pixels ($\Delta = -8.05\%$, unadjusted $p = 0.0185$), but this reduction did not survive family-wise multiplicity adjustment ($p_{\text{adj}} = 0.1298$) due in part to substantial training-seed variability." | ❌ "Masking significantly impairs clean rejection after Bonferroni correction"<br>❌ "Proved failure of rejection" | - $\Delta \text{Clean Rej} = -0.0805$<br>- 95% BCa CI: $[-0.1515, -0.0152]$<br>- Jackknife SE: $0.0351$<br>- Seed SD: $0.2560$ (Seed 42: -31.0%, Seed 1337: -12.6%, Seed 2026: +19.5%)<br>- Raw BCa $p = 0.0185$<br>- Primary Bonferroni $p_{\text{adj}} = 0.1298$<br>- Full Bonferroni $p_{\text{adj}} = 0.3893$ | Conditioned strictly on the 87 normal ovary test scans. Must be reported as a secondary nominal trend, not a primary confirmatory finding. | **Fails Primary ($M=7$) and Full ($M=21$) Multiplicity Control ($p_{\text{adj}} \ge 0.05$)** |
| **CLM-05** | **Extra-Ovarian False Positives** (Arm C vs Arm B on Extra-Ovarian FPR) | "No statistically significant difference in extra-ovarian false-positive rate was detected between non-generative masking and harmonic inpainting ($\Delta = -0.03\%$, 95% BCa CI $[-0.30\%, +0.23\%]$, $p = 0.8316$); formal equivalence was not tested." | ❌ "Equivalence confirmed in extra-ovarian space"<br>❌ "Masking produces zero extra-ovarian error" | - $\Delta \text{FPR}_{\text{extra}} = -0.0003$<br>- 95% BCa CI: $[-0.0030, +0.0023]$<br>- Jackknife SE: $0.0014$<br>- Seed SD: $0.0018$<br>- Raw BCa $p = 0.8316$<br>- Primary Bonferroni $p_{\text{adj}} = 1.0000$ | Conditioned on the 87 normal ovary scans outside the annotated ovary boundary. | Non-significant ($p_{\text{adj}} = 1.0$) |
| **CLM-06** | **Secondary Benchmark: Inpainting vs Raw** (Arm B vs Arm A on LPR) | "In secondary exploratory comparisons against un-preprocessed inputs, harmonic inpainting significantly improved in-caliper lesion sensitivity over raw scans by an estimated 4.78 percentage points." | ❌ "Inpainting universally superior"<br>❌ "Primary endpoint victory" (Arm B vs Arm A is a secondary exploratory comparison) | - $\Delta \text{LPR} = +0.0478$<br>- 95% BCa CI: $[+0.0216, +0.0763]$<br>- Jackknife SE: $0.0140$<br>- Seed SD: $0.0694$<br>- Raw BCa $p = 0.0015$<br>- Full Bonferroni $p_{\text{adj}} = 0.0325$ | Conditioned on the 193 overlap scans; evaluated within the full 21-comparison secondary family. | **Survives Full Benchmark Bonferroni Correction ($M=21$, $p_{\text{adj}} < 0.05$)** |
| **CLM-07** | **Secondary Benchmark: Masking vs Raw** (Arm C vs Arm A on Normal Ovary Errors) | "In secondary exploratory comparisons against un-preprocessed inputs, non-generative masking significantly reduced clean rejection ($\Delta = -11.49\%$, $p_{\text{adj}} = 0.0151$) and increased intra-ovarian FPR ($\Delta = +8.24\%$, $p_{\text{adj}} = 0.0105$)." | ❌ "Primary confirmatory proof"<br>❌ "Masking ruins all segmentation" | - Clean Rejection: $\Delta = -0.1149$, 95% BCa CI $[-0.1777, -0.0588]$, $p_{\text{adj}} = 0.0151$<br>- Intra FPR: $\Delta = +0.0824$, 95% BCa CI $[+0.0522, +0.1240]$, $p_{\text{adj}} = 0.0105$ | Evaluated across 87 normal ovary scans; secondary benchmark checks against raw baseline. | **Survives Full Benchmark Bonferroni Correction ($M=21$, $p_{\text{adj}} < 0.05$)** |

---

## 2. Protocol & Registration Status Disclosure

### 2.1 Formal Protocol Provenance Statement for Methods Section
> **Protocol & Multiplicity Disclosure**:  
> "The comparison and seven analysis endpoints were specified in internal protocols before model training. Bonferroni adjustment was applied during subsequent inferential analysis and was not documented as a prespecified procedure. Furthermore, the study was not prospectively registered on a public trial or study registry (such as ClinicalTrials.gov or the Open Science Framework). Consequently, all inferential findings are formally designated as planned secondary and internal-protocol analyses rather than prospectively pre-registered clinical trial endpoints."

---

## 3. Estimand Specification & Variance Separation

### 3.1 Three-Seed Scan-Level Mean Estimand
All reported point estimates represent the **three-seed scan-level mean metric**:
$$\bar{y}_i = \frac{1}{3} \sum_{s \in \{42, 1337, 2026\}} y_{s, i}$$
This quantity represents the seed-averaged performance estimate of models trained under the specified preprocessing regimes.

### 3.2 Dual Variance Decomposition
1. **Scan-Cohort Sampling Uncertainty ($\text{SE}_{\text{jack}}$ / 95% BCa CI)**:
   Quantifies the sampling error of evaluating the three fixed trained checkpoints over a finite sample of patients/scans ($N=469$, $K=454$ clusters). Computed via paired pseudo-patient cluster bootstrap ($B=2,000$).
2. **Training-Seed Variability ($s_{\text{seed}}$)**:
   Quantifies algorithmic/stochastic training variance across weight initializations and batch shuffle sequences ($s_{\text{seed}} = \sqrt{\frac{1}{2} \sum_{s} (\Delta_s - \bar{\Delta})^2}$).
   - **LPR (Arm C vs Arm B)**: $\bar{\Delta} = -0.0710$, with all three seeds showing negative differences (Seed 42: $-0.0742$, Seed 1337: $-0.0195$, Seed 2026: $-0.1194$; $s_{\text{seed}} = 0.0500$).
   - **Intra-Ovarian FPR (Arm C vs Arm B)**: $\bar{\Delta} = +0.0914$, but individual seed differences varied in sign (Seed 42: $+0.2191$, Seed 1337: $+0.1191$, Seed 2026: $-0.0640$; $s_{\text{seed}} = 0.1435$).
   - **Clean Rejection (Arm C vs Arm B)**: $\bar{\Delta} = -0.0805$, with individual seed differences reversing sign (Seed 42: $-0.3103$, Seed 1337: $-0.1264$, Seed 2026: $+0.1954$; $s_{\text{seed}} = 0.2560$).

### 3.3 Consistency of BCa P-Values and Confidence Intervals
The BCa-inverted $p$-values are internally consistent with the corresponding BCa confidence intervals by construction. The null-centred bootstrap is retained as a separate supporting analysis.
