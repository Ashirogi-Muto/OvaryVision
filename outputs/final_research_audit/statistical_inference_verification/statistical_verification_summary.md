# Statistical Verification Summary: OvaryVision Inferential Audit (Revised)

**Date**: 2026-11-02  
**Status**: Inferential Calculations Audited, Recalculated, and Formally Reconciled  
**Directory**: `outputs/final_research_audit/statistical_inference_verification/`  

---

## 1. Executive Summary of Corrections

1. **Bonferroni Reporting Standardized**:
   - Rejection decisions are evaluated by comparing adjusted $p$-values against $\alpha = 0.05$ ($p_{\text{adj}} < 0.05$), or equivalently comparing raw $p$-values against $\alpha / M$ ($p_{\text{raw}} < 0.05 / M$).
   - The erroneous comparison of adjusted $p$-values against $\alpha / M$ has been explicitly eliminated.
   - All adjusted values have been arithmetically verified: $p_{\text{adj}} = \min(1.0, M \times p_{\text{raw}})$.
2. **Protocol & Bonferroni Provenance Statement**:
   - The comparison and seven analysis endpoints were specified in internal protocols before model training. Bonferroni adjustment was applied during subsequent inferential analysis and was not documented as a prespecified procedure.
   - The primary family is designated as **"Designated Primary Analysis Family (M=7)"**.
3. **Bootstrap P-Value Resolution Formalized**:
   - For Monte Carlo tests with $B=2,000$, zero observed extreme replicates ($\sum \mathbb{I} = 0$) bounds the $p$-value from above rather than measuring an impossible $p = 0.0000$.
   - The minimum resolvable finite-sample point estimate is $\frac{1}{B+1} \approx 0.0005$, presented formally in tables as $p < 0.001$.
4. **Null-Centered Bootstrap Imposition**:
   - Shifted bootstrap replicates by the observed difference ($\tilde{\theta}^* = \hat{\theta}^* - \hat{\theta}_{\text{obs}}$) to generate an empirical reference distribution under $H_0: \theta = 0$.
   - Preserved paired patient cluster multiplicity across both arms simultaneously.
5. **BCa Consistency Language Clarified**:
   - The BCa-inverted $p$-values are internally consistent with the corresponding BCa confidence intervals by construction.
   - The null-centred bootstrap is retained as a separate supporting analysis.
6. **Variance Disentanglement & Seed Consistency**:
   - Quantified training-seed variability ($s_{\text{seed}}$ across Seeds 42, 1337, and 2026) separately from scan-level bootstrap uncertainty ($\text{SE}_{\text{jack}}$ across $N=469$ scans).
   - Clarified that while the three-seed scan-level mean showed higher intra-ovarian FPR under zero-masking (+0.0914), the direction varied across training seeds (Seed 42: +0.2191, Seed 1337: +0.1191, Seed 2026: -0.0640), indicating sensitivity to model initialization.
7. **Removal of Unsupported Equivalence Claims**:
   - Replaced all claims of equivalence with: *"No statistically significant difference was detected; formal equivalence was not tested."*

---

## 2. Core Scientific Verdicts

1. **Designated Primary Analysis Family (Arm C vs Arm B, $M=7$)**:
   - **LPR Deficit Survives FWER Control**: $\bar{\Delta} = -7.10\%$, 95% BCa CI $[-9.69\%, -4.60\%]$, raw $p_{\text{BCa}} = 0.0011$, Bonferroni-adjusted $p_{\text{adj}} = \mathbf{0.0078} < 0.05$. All three training seeds showed negative differences (Seed 42: $-0.0742$, Seed 1337: $-0.0195$, Seed 2026: $-0.1194$).
   - **Intra-Ovarian FPR Inflation on Pooled Estimand**: $\bar{\Delta} = +9.14\%$, 95% BCa CI $[+5.32\%, +13.82\%]$, raw $p_{\text{BCa}} < 0.001$, Bonferroni-adjusted $p_{\text{adj}} = \mathbf{0.0035} < 0.05$. However, the direction varied across training seeds (Seed 42: $+0.2191$, Seed 1337: $+0.1191$, Seed 2026: $-0.0640$), indicating sensitivity to model initialization.
   - **Tumour Delineation (No Statistically Significant Difference Detected; Formal Equivalence Not Tested)**: Tumour DSC ($\bar{\Delta} = +0.19\%$, $p_{\text{adj}} = 1.0$), IoU ($\bar{\Delta} = -0.15\%$, $p_{\text{adj}} = 1.0$), and Boundary IoU ($\bar{\Delta} = +0.13\%$, $p_{\text{adj}} = 1.0$) show no statistically significant difference; formal equivalence was not tested.
   - **Clean Rejection Drop Fails FWER Control**: $\bar{\Delta} = -8.05\%$, raw $p_{\text{BCa}} = 0.0185$, but Bonferroni-adjusted $p_{\text{adj}} = \mathbf{0.1298} \ge 0.05$. It is reported as a secondary nominal trend with high training-seed variability ($s_{\text{seed}} = 25.60\%$).
2. **Benchmark-Wide Multiplicity (All Comparisons, $M=21$)**:
   - Exactly 6 comparisons survive full Bonferroni correction ($p_{\text{adj}} < 0.05 \iff p_{\text{raw}} < 0.00238$):
     1. Arm C vs Arm B on LPR ($p_{\text{adj}} = 0.0233$)
     2. Arm C vs Arm B on Intra FPR ($p_{\text{adj}} = 0.0105$)
     3. Arm C vs Arm A on Clean Rejection ($p_{\text{adj}} = 0.0151$)
     4. Arm C vs Arm A on Intra FPR ($p_{\text{adj}} = 0.0105$)
     5. Arm B vs Arm A on LPR ($p_{\text{adj}} = 0.0325$)
     6. Arm B vs Arm A on Extra FPR ($p_{\text{adj}} = 0.0105$)

---

## 3. Deliverables Index

All files reside in [`outputs/final_research_audit/statistical_inference_verification/`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/):

1. [`corrected_manuscript_table_primary_family.csv`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/corrected_manuscript_table_primary_family.csv): Corrected 7-row primary family table with updated family name, seed variability, and standardized verdicts.
2. [`corrected_manuscript_table_full_21_comparisons.csv`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/corrected_manuscript_table_full_21_comparisons.csv): Corrected 21-row benchmark table for supplementary materials.
3. [`final_claims_register.md`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/final_claims_register.md): Master register of approved scientific statements, prohibited phrases, statistical backing, and registration disclosure.
4. [`final_manuscript_consistency_audit.md`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/final_manuscript_consistency_audit.md): Complete audit report certifying the four areas of correction.
5. [`p_value_implementation_audit.md`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/p_value_implementation_audit.md): Audit answering implementation questions, documenting null centering, and BCa inversion formulas.
6. [`multiple_comparison_analysis.md`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/multiple_comparison_analysis.md): Multiplicity analysis, decision rules, variance component separation, and manuscript tables.
7. [`recalculated_inferential_results.csv`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/recalculated_inferential_results.csv): Comprehensive 21-row table with all 31 columns.
8. [`verification_execution_log.txt`](file:///home/ubuntu/major/outputs/final_research_audit/statistical_inference_verification/verification_execution_log.txt): Execution log recording verification timestamps and convergence.
