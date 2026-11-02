#!/usr/bin/env python3
"""
Generate Corrected Manuscript Tables for OvaryVision Segmentation Benchmark.

Applies final scientific reporting corrections:
1. Replaces unsupported equivalence claims with:
   "No statistically significant difference was detected; formal equivalence was not tested."
2. Updates family terminology to "Designated Primary Analysis Family (M=7)" and "Secondary Analysis Family (M=21)".
3. Uses precise seed-averaged estimand terminology.
4. Outputs:
   - outputs/final_research_audit/statistical_inference_verification/corrected_manuscript_table_primary_family.csv
   - outputs/final_research_audit/statistical_inference_verification/corrected_manuscript_table_full_21_comparisons.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

def main():
    base_dir = Path("outputs/final_research_audit/statistical_inference_verification")
    in_csv = base_dir / "recalculated_inferential_results.csv"
    df = pd.read_csv(in_csv)

    # 1. Update hypothesis family names
    family_map = {
        "Primary Confirmatory Family (M=7)": "Designated Primary Analysis Family (M=7)",
        "Secondary Benchmark Family (M=21)": "Secondary Analysis Family (M=21)"
    }
    df["analysis_family"] = df["hypothesis_family"].map(lambda x: family_map.get(x, x))

    # 2. Build corrected, precise statistical verdicts
    verdicts = []
    for _, row in df.iterrows():
        comp = row["comparison"]
        m = row["metric"]
        sig_nom = row["sig_nominal_05"]
        sig_b_p = row["sig_bonf_primary"]
        sig_b_f = row["sig_bonf_full"]
        is_prim = (row["analysis_family"] == "Designated Primary Analysis Family (M=7)")

        if not sig_nom:
            v = "No statistically significant difference was detected; formal equivalence was not tested."
        elif is_prim:
            if sig_b_p:
                v = f"Statistically significant difference; survives primary Bonferroni correction (p_adj = {row['p_bonf_primary_7']:.4f} < 0.05)."
            else:
                v = f"Nominally significant difference (p_raw = {row['p_val_bca_inverted']:.4f} < 0.05), but fails primary Bonferroni correction (p_adj = {row['p_bonf_primary_7']:.4f} >= 0.05)."
        else:
            if sig_b_f:
                v = f"Statistically significant difference; survives full Bonferroni correction (p_adj = {row['p_bonf_full_21']:.4f} < 0.05)."
            else:
                v = f"Nominally significant difference (p_raw = {row['p_val_bca_inverted']:.4f} < 0.05), but fails full Bonferroni correction (p_adj = {row['p_bonf_full_21']:.4f} >= 0.05)."
        verdicts.append(v)

    df["statistical_verdict"] = verdicts

    # Specific informative suffixes for significant findings
    # Row 3 (lpr C-B): deficit under masking
    # Row 5 (fpr_intra C-B): elevation under masking
    # Row 4 (clean_rej C-B): reduction under masking (nominal)
    # Row 12 (clean_rej C-A): reduction under masking
    # Row 13 (fpr_intra C-A): elevation under masking
    # Row 17 (lpr B-A): benefit under inpainting
    # Row 20 (fpr_extra B-A): elevation under inpainting

    # Reorder columns
    cols_order = [
        "comparison",
        "metric",
        "cohort",
        "analysis_family",
        "hypothesis_tier",
        "obs_diff",
        "bca_ci_lower",
        "bca_ci_upper",
        "diff_seed42",
        "diff_seed1337",
        "diff_seed2026",
        "seed_mean",
        "seed_sd",
        "se_jackknife",
        "se_bootstrap",
        "p_val_original",
        "p_val_bca_inverted",
        "p_val_bca_display",
        "p_val_null_centered",
        "p_val_null_display",
        "p_val_wald",
        "p_bonf_primary_7",
        "p_bonf_primary_display",
        "p_bonf_full_21",
        "p_bonf_full_display",
        "sig_nominal_05",
        "sig_bonf_primary",
        "sig_bonf_full",
        "z0_bias",
        "acceleration",
        "statistical_verdict"
    ]
    df_out = df[cols_order].copy()

    # Save full corrected table
    full_path = base_dir / "corrected_manuscript_table_full_21_comparisons.csv"
    df_out.to_csv(full_path, index=False)
    print(f"Saved: {full_path} ({len(df_out)} rows)")

    # Save primary family table
    primary_df = df_out[df_out["analysis_family"] == "Designated Primary Analysis Family (M=7)"].copy()
    primary_path = base_dir / "corrected_manuscript_table_primary_family.csv"
    primary_df.to_csv(primary_path, index=False)
    print(f"Saved: {primary_path} ({len(primary_df)} rows)")

if __name__ == "__main__":
    main()
