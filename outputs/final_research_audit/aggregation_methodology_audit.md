# Aggregation Methodology & Estimand Audit: OvaryVision 3-Seed Benchmark

**Date**: 2026-11-01  
**Auditor**: Antigravity (Advanced Agentic Coding / DeepMind)  
**Target Script**: `scripts/aggregate_three_seed_benchmark.py`  
**Data Inputs**: `outputs/benchmark_runs/segmentation_benchmark/seed{42,1337,2026}_test_evaluation/seed*_per_scan_test_metrics.csv`  

---

## 1. Aggregation Mechanism Classification

Inspection of `scripts/aggregate_three_seed_benchmark.py` (lines 171–202) reveals the exact mathematical aggregation mechanism implemented:

```python
# Compute scan-level mean across seeds cleanly per cohort subset
for arm in arms:
    for m in ["dsc", "iou", "biou"]:
        col_name = f"{arm}_{m}"
        stacked = np.stack([per_scan_dfs[s][col_name].to_numpy() for s in seeds], axis=1)
        mean_vals = np.full(num_scans, np.nan)
        sd_vals = np.full(num_scans, np.nan)
        mean_vals[is_tumour] = np.mean(stacked[is_tumour], axis=1)
        sd_vals[is_tumour] = np.std(stacked[is_tumour], axis=1, ddof=1) if len(seeds) > 1 else 0.0
        base_df[f"{col_name}_mean"] = mean_vals
        base_df[f"{col_name}_sd"] = sd_vals
```

### Classification Verdict: **Option A (Averages per-scan metrics across seeds)**

- **Option A (Actual Implementation)**: For each individual test scan $i$ and metric $m$, the metric is evaluated independently on the predictions of the three seed checkpoints ($s \in \{42, 1337, 2026\}$), producing three scalar values $m_{i, 42}, m_{i, 1337}, m_{i, 2026}$. The scan-level pooled metric is computed as their arithmetic mean:
  $$\bar{m}_i = \frac{1}{3} \sum_{s \in \{42, 1337, 2026\}} m_{i, s}$$
- **Option B (Prediction Averaging / Ensembling)**: **NOT USED**. Soft probability maps or binary segmentations $\hat{Y}_{i, s}$ were *not* averaged prior to thresholding or metric calculation. The evaluation script evaluates individual model checkpoints, not an ensemble.
- **Option C (Independent Pooling across Scan-Seeds)**: **NOT USED**. The script strictly avoids flattening the observations into $3 \times 469 = 1,407$ independent rows. It retains $N=469$ scan units, preventing artificial deflation of bootstrap standard errors.
- **Option D (Other)**: Not applicable.

---

## 2. Precise Estimand Definition & Limitations

### 2.1 The Mathematical Estimand
Let $Y_i(A, \theta_s)$ be the metric obtained on scan $i$ by a network trained under preprocessing arm $A$ with stochastic initialization/augmentation seed $s$.

The scan-level average score is:
$$\bar{Y}_i(A) = \frac{1}{S} \sum_{s=1}^S Y_i(A, \theta_s) \quad (S=3)$$

When paired cluster bootstrap is applied across the $K=454$ pseudo-patient clusters with weights $w_k$, the resulting inferential statistic estimates:
$$\theta_{\text{pooled}} = \mathbb{E}_{i \sim \mathcal{D}_{\text{patients}}} \left[ \bar{Y}_i(C) - \bar{Y}_i(B) \right] = \frac{1}{S} \sum_{s=1}^S \mathbb{E}_{i \sim \mathcal{D}_{\text{patients}}} \left[ Y_i(C, \theta_s) - Y_i(B, \theta_s) \right]$$

### 2.2 Critical Limitations of this Estimand
1. **Fixed-Committee Conditioning**: The bootstrap resampling resamples **patient scans/clusters $i$**, keeping the set of three trained models $\{\theta_{42}, \theta_{1337}, \theta_{2026}\}$ **fixed**. Consequently:
   - The BCa 95% confidence intervals quantify **patient cohort sampling variability**.
   - They **do not quantify the full predictive uncertainty of retraining a new model with a fourth seed** (e.g., Seed 9999).
2. **Non-Equivalence to Ensembling**: Averaging non-linear metrics (such as Dice: $\frac{2|X \cap Y|}{|X| + |Y|}$) across seeds is generally concave and differs from the metric of the averaged prediction:
   $$\frac{1}{S} \sum_{s=1}^S \text{DSC}(\hat{Y}_s, Y) \neq \text{DSC}\left(\frac{1}{S} \sum_{s=1}^S \hat{Y}_s, Y\right)$$
3. **Variance Compression**: Scan-level averaging reduces the observed variance of $\bar{Y}_i$ compared to any single run $Y_{i, s}$ by averaging out seed-level stochastic noise per scan.

---

## 3. Three-Tiered Reporting Structure

To eliminate ambiguity and prevent treating $N=3$ seeds as an infinite sample or collapsing seed variability, the audit establishes three separate reporting tiers:

### Tier A: Per-Seed Complete Results (`per_seed_results.csv`)
Shows every metric independently for each seed and arm:
- All 9 models preserved as distinct empirical points.
- No cross-seed averaging.

### Tier B: Scan-Level Averaging (`three_seed_pooled_*_summary.csv`)
- Retains scan-level mean scores $\bar{m}_i$.
- Evaluates paired cluster bootstrap with BCa intervals ($B=2,000$).
- Accurately represents the performance of the 3-model committee over the held-out patient population.

### Tier C: Training-Seed Variability Summary (`seed_variability_summary.csv`)
- Treats $N=3$ training seeds as the unit of variation.
- Reports Mean, Sample Standard Deviation ($s$, $ddof=1$), Range [Min, Max], and Coefficient of Variation ($CV = \frac{s}{\bar{x}} \times 100\%$).
- Avoids overconfident asymptotic inferential tests (such as paired t-tests with $df=2$).

---

## 4. Cohort Partitioning & Resampling Units

The test cohort consists of 469 scans. The table below details the exact sample counts, cluster counts, and conditional rules for every evaluated cohort:

| Cohort Name | Metric Group | Total Scans ($N$) | Pseudo-Patient Clusters ($K$) | Cluster Composition | Exclusion / Imputation Rule |
|---|---|:---:|:---:|---|---|
| **Tumour Cohort** | DSC, IoU, Boundary IoU | 382 | 370 | 67 multi-frame clusters, 303 singletons | Normal scans ($N=87$) excluded. |
| **Overlap Cohort** | Lesion Preservation Ratio (LPR) | 193 | 188 | 32 multi-frame clusters, 156 singletons | Zero-overlap scans ($N=189$) excluded; degenerate draws redrawn via rejection sampling. |
| **Normal Ovary Cohort** | Clean Rejection, Intra/Extra FPR | 87 | 84 | 15 multi-frame clusters, 69 singletons | Tumour scans ($N=382$) excluded. |

All bootstrap iterations preserve multinomial cluster draw multiplicities ($m_k \ge 0$). Zero-overlap draws for LPR are handled strictly by rejection sampling, never by constant imputation ($0$ or $1$).

