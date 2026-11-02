# OvaryVision — Final Manuscript Consistency Audit (Revised)

**Audit Date**: 2026-11-02  
**Status**: Completed and Formally Certified  
**Target Scope**: Manuscript Tables, Statistical Claims, Estimand Terminology, and Protocol Provenance  
**Output Directory**: [`outputs/final_research_audit/statistical_inference_verification/`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/)  

---

## 1. Executive Audit Summary

Following the comprehensive audit of all inferential outputs, all scientific-reporting corrections have been rigorously applied:
1. **Unsupported Equivalence Claims Removed**: All occurrences of "Equivalence Confirmed" and "Primary Endpoint: Equivalence" have been purged. For non-significant endpoints (DSC, IoU, Boundary IoU, extra-ovarian FPR), the standardized language **"No statistically significant difference was detected; formal equivalence was not tested"** is now enforced universally. No retrospective equivalence margins were fabricated.
2. **Estimand Terminology Corrected**: All references to "ensemble" or "ensemble predictor" have been replaced with **"three-seed scan-level mean metric"** or **"seed-averaged performance estimate."** The cluster bootstrap is explicitly clarified as quantifying scan-cohort sampling uncertainty conditional on the three fixed trained models, while training-seed variability ($s_{\text{seed}}$) is preserved as an independent reporting layer.
3. **Protocol & Bonferroni Provenance Clarified**: The comparison and seven analysis endpoints were specified in internal protocols before model training. Bonferroni adjustment was applied during subsequent inferential analysis and was not documented as a prespecified procedure. Furthermore, the study was not prospectively registered on a public trial or study registry.
4. **Seed Consistency Claims Corrected**: Corrected the description of intra-ovarian FPR. The three-seed scan-level mean showed higher intra-ovarian FPR under zero-masking (+0.0914), but the direction varied across training seeds (Seed 42: +0.2191, Seed 1337: +0.1191, Seed 2026: -0.0640), indicating sensitivity to model initialization.
5. **BCa Consistency Language Clarified**: The BCa-inverted $p$-values are internally consistent with the corresponding BCa confidence intervals by construction. The null-centred bootstrap is retained as a separate supporting analysis.

---

## 2. Detailed Verification of the Pre-Registration Claim

### 2.1 Audit of Existing Project Records
A full text and git-history search was performed across all project markdown files, specifications, decisions, and commit logs:
- **`SPEC.md` / `DECISIONS.md`**: Establish architecture choices, high-level dataset scope, and Decision D01 ("Paired statistical comparison"). However, Decision D01 references DeLong covariance and classification AUROC comparisons; it does not specify the 7 segmentation endpoints or a Bonferroni family-wise adjustment.
- **`docs/CEUS_AND_SEGMENTATION_EXPERIMENT_PLAN.md` (lines 195–202)**: Authored prior to the 9-run benchmark, this document explicitly established the primary comparison of Arm C versus Arm B via paired cluster bootstrap ($B=2,000$, BCa 95% CIs) for $\Delta \text{DSC}$, $\Delta \text{BIoU}$, and $\Delta \text{LPR}$. However, it did not define the complete 7-metric family or a formal family-wise multiplicity adjustment.
- **`docs/SEGMENTATION_PRE-EXECUTION_AUDIT.md` (Sections 3.4 & 5.1)**: Formulated the normal ovary endpoints (`clean_rej`, `fpr_intra`, `fpr_extra`), bringing the total evaluation endpoints to seven.
- **External Public Registries**: No registration was submitted to ClinicalTrials.gov, OSF, or equivalent timestamped repositories.

### 2.2 Formal Finding
While internal written protocols predated the final benchmark runs:
> **"The comparison and seven analysis endpoints were specified in internal protocols before model training. Bonferroni adjustment was applied during subsequent inferential analysis and was not documented as a prespecified procedure."**

In accordance with ICMJE and scientific reporting standards:
- The family is renamed to **"Designated Primary Analysis Family (M=7)"**.
- The manuscript methods section will include the explicit disclosure statement documented in [`final_claims_register.md`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/final_claims_register.md#2-protocol--registration-status-disclosure).

---

## 3. Comprehensive Consistency Audit Across Core Dimensions

### 3.1 Decision Rule Consistency
- **Rule A ($p_{\text{adj}} < 0.05$) vs Rule B ($p_{\text{raw}} < 0.05 / M$)**:
  - Primary Family ($M=7$, threshold $\alpha/7 = 0.00714$):
    - LPR: $p_{\text{raw}} = 0.0011 < 0.00714 \iff p_{\text{adj}} = 0.0078 < 0.05$. (Significant)
    - Intra FPR: $p_{\text{raw}} < 0.001 < 0.00714 \iff p_{\text{adj}} = 0.0035 < 0.05$. (Significant on pooled estimand)
    - Clean Rejection: $p_{\text{raw}} = 0.0185 > 0.00714 \iff p_{\text{adj}} = 0.1298 \ge 0.05$. (Nominal only, fails FWER)
    - DSC, IoU, Boundary IoU, Extra FPR: $p_{\text{raw}} > 0.23 \iff p_{\text{adj}} = 1.0000 \ge 0.05$. (Non-significant)
  - Full Benchmark Family ($M=21$, threshold $\alpha/21 = 0.00238$):
    - Exactly 6 comparisons survive: C-vs-B LPR ($p_{\text{adj}} = 0.0233$), C-vs-B Intra FPR ($p_{\text{adj}} = 0.0105$), C-vs-A Clean Rej ($p_{\text{adj}} = 0.0151$), C-vs-A Intra FPR ($p_{\text{adj}} = 0.0105$), B-vs-A LPR ($p_{\text{adj}} = 0.0325$), B-vs-A Extra FPR ($p_{\text{adj}} = 0.0105$).
  - **No comparison was made between $p_{\text{adj}}$ and $\alpha / M$.** Arithmetic consistency is 100% verified.

### 3.2 Cohort Counts & Sample Conditioning
- **All Tumour Scans**: Exactly $N = 382$ scans across $K = 370$ clusters.
- **Overlap Cohort**: Exactly $N = 193$ scans with non-zero caliper-tumour intersection ($|\Omega_{\text{overlap}}| > 0$). Zero-overlap cases ($N=189$) are excluded without imputation.
- **Normal Ovary Scans**: Exactly $N = 87$ scans across $K = 84$ clusters ($Y_{\text{tumour}} \equiv 0$).
- **Total Test Scans**: Exactly $382 + 87 = 469$ scans across $K = 454$ pseudo-patient clusters.

### 3.3 Directionality & BCa Consistency Language
- The BCa-inverted $p$-values are internally consistent with the corresponding BCa confidence intervals by construction.
- The null-centred bootstrap is retained as a separate supporting analysis.

### 3.4 Disentangling Uncertainty and Seed Variability
- **Scan-Cohort Uncertainty**: Represented by leave-one-cluster-out Jackknife SE ($\text{SE}_{\text{jack}}$) and BCa 95% CIs conditional on the 3 fixed models.
- **Training-Seed Variability**: Represented by the sample standard deviation ($s_{\text{seed}}$) across Seeds 42, 1337, and 2026.
- **Reporting Rule**: The three-seed scan-level mean showed higher intra-ovarian FPR under zero-masking (+0.0914), but the direction varied across training seeds (Seed 42: +0.2191, Seed 1337: +0.1191, Seed 2026: -0.0640), indicating sensitivity to model initialization.
- For Clean Rejection (Arm C vs Arm B), scan-level $\text{SE}_{\text{jack}} = 0.0351$ while training-seed $s_{\text{seed}} = 0.2560$, explaining why it fails multiplicity correction despite nominal scan-level significance.

---

## 4. Deliverables Index

All revised deliverables are located in [`outputs/final_research_audit/statistical_inference_verification/`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/):

1. [`corrected_manuscript_table_primary_family.csv`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/corrected_manuscript_table_primary_family.csv): Corrected 7-row primary family table with updated family name, seed variability, and standardized verdicts.
2. [`corrected_manuscript_table_full_21_comparisons.csv`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/corrected_manuscript_table_full_21_comparisons.csv): Corrected 21-row benchmark table for supplementary materials.
3. [`final_claims_register.md`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/final_claims_register.md): Master register of approved scientific statements, prohibited phrases, statistical backing, and registration disclosure.
4. [`final_manuscript_consistency_audit.md`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/final_manuscript_consistency_audit.md): This comprehensive verification document.
