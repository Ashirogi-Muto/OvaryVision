# Paired Cluster Bootstrap & BCa Statistical Validation Report

**Date**: 2026-11-01  
**Auditor**: Antigravity (Advanced Agentic Coding / DeepMind)  
**Implementation**: `src/ovaryvision/statistics.py` (`paired_cluster_bootstrap_segmentation()`)  
**Test Suite**: `tests/test_statistics.py` (13 tests, all passing)  

---

## 1. Algorithm Verification

The paired cluster bootstrap implementation was inspected line-by-line against Efron & Tibshirani (1993) and Davison & Hinkley (1997) standards for clustered dependent data.

### 1.1 Resampling Unit & Multiplicity Preservation
- **Resampling Unit**: Pseudo-patient cluster index $k \in \{0, \dots, K-1\}$.
- **Draw Mechanism**: Multinomial sampling with replacement:
  ```python
  sampled_k = rng.choice(num_clusters, size=num_clusters, replace=True)
  multiplicities = np.bincount(sampled_k, minlength=num_clusters)
  ```
- **Integrity Check**: Cluster draws preserve integer multiplicity $m_k \ge 0$. The old set-union bug that collapsed duplicate draws was completely eradicated. Verified by unit test `test_paired_cluster_bootstrap_multiplicity_preservation`.

### 1.2 Conditional Rejection for Degenerate Sub-Cohorts
- For conditional metrics evaluated on restricted subsets (specifically Lesion Preservation Ratio on $N=193$ caliper overlap scans):
  ```python
  sampled_total_count = np.sum(multiplicities * cluster_valid_sizes)
  if sampled_total_count > 0:
      boot_diffs[b] = sampled_total_diff / sampled_total_count
      break
  ```
- **Integrity Check**: Iterations drawing zero valid overlap scans are conditionally redrawn via rejection sampling. Zero arbitrary imputation ($0$ or $1$) occurs. In the test evaluation ($B=2,000$), **zero empty draws occurred** across all runs because 188 of 454 clusters contain overlap observations.

### 1.3 Median Bias Correction $z_0$
- Computed using mid-p tie correction to prevent quantile distortion on discrete/tied metrics:
  $$\hat{p} = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\theta^*_b < \hat{\theta}) + \frac{1}{2B} \sum_{b=1}^B \mathbb{I}(\theta^*_b = \hat{\theta})$$
- Clipped to finite-sample bounds $\left[\frac{1}{B+1}, \frac{B}{B+1}\right]$ prior to probit transformation $z_0 = \Phi^{-1}(\hat{p})$.
- Verified by unit tests `test_paired_cluster_bootstrap_constant_statistic` and `test_paired_cluster_bootstrap_many_tied_replicates`.

### 1.4 Acceleration $a$ via Cluster Leave-One-Out Jackknife
- Acceleration is estimated directly at the cluster level via $O(K)$ jackknife:
  $$\hat{\theta}_{(k)} = \frac{\sum_{j \neq k} \sum_{i \in \text{valid}_j} (Y_{i, C} - Y_{i, B})}{\sum_{j \neq k} |\text{valid}_j|}$$
  $$a = \frac{\sum_{k=1}^K (\bar{\theta}_{(\cdot)} - \hat{\theta}_{(k)})^3}{6 \left[ \sum_{k=1}^K (\bar{\theta}_{(\cdot)} - \hat{\theta}_{(k)})^2 \right]^{3/2}}$$

### 1.5 Fallback Mechanism & Adjusted Quantiles
- Adjusted quantiles $\alpha_1, \alpha_2$ are computed as:
  $$\alpha_1 = \Phi\left( z_0 + \frac{z_0 + z_{\alpha/2}}{1 - a(z_0 + z_{\alpha/2})} \right), \quad \alpha_2 = \Phi\left( z_0 + \frac{z_0 + z_{1-\alpha/2}}{1 - a(z_0 + z_{1-\alpha/2})} \right)$$
- If denominators $\le 10^{-8}$, arguments non-finite, or quantiles non-monotonic ($q_1 \ge q_2$ or outside $(0, 1)$), the function automatically falls back to empirical percentiles ($\alpha/2, 1-\alpha/2$) and sets `ci_method = "percentile_fallback"`.

---

## 2. Empirical Validation Results on 3-Seed Pooled Data ($B=2,000$)

The table below presents the verified bootstrap parameters and confidence intervals for all 10 pooled comparisons across the 469 held-out test scans:

| Comparison | Metric | Cohort ($N, K$) | Observed $\Delta$ | BCa 95% CI | Percentile 95% CI | $z_0$ | Accel $a$ | Method | $p$-value | Fallback |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm C vs Arm B** | DSC | Tumour (382, 370) | $+0.0019$ | $[-0.0070, +0.0131]$ | $[-0.0080, +0.0120]$ | $+0.0803$ | $+0.0112$ | BCa | $0.776$ | No |
| **Arm C vs Arm B** | IoU | Tumour (382, 370) | $-0.0015$ | $[-0.0105, +0.0091]$ | $[-0.0112, +0.0084]$ | $+0.0589$ | $+0.0048$ | BCa | $0.728$ | No |
| **Arm C vs Arm B** | Boundary IoU | Tumour (382, 370) | $+0.0013$ | $[-0.0008, +0.0036]$ | $[-0.0009, +0.0036]$ | $+0.0175$ | $+0.0036$ | BCa | $0.256$ | No |
| **Arm C vs Arm B** | LPR | Overlap (193, 188) | $\mathbf{-0.0710}$ | $\mathbf{[-0.0969, -0.0460]}$ | $[-0.0975, -0.0467]$ | $+0.0320$ | $-0.0032$ | BCa | $\mathbf{0.0005}$ | No |
| **Arm C vs Arm B** | Clean Rejection | Normal (87, 84) | $\mathbf{-0.0805}$ | $\mathbf{[-0.1515, -0.0152]}$ | $[-0.1490, -0.0116]$ | $-0.0414$ | $+0.0003$ | BCa | $\mathbf{0.022}$ | No |
| **Arm C vs Arm B** | Intra-Ovary FPR | Normal (87, 84) | $\mathbf{+0.0914}$ | $\mathbf{[+0.0532, +0.1382]}$ | $[+0.0498, +0.1350]$ | $+0.0364$ | $+0.0173$ | BCa | $\mathbf{0.0005}$ | No |
| **Arm C vs Arm B** | Extra-Ovary FPR | Normal (87, 84) | $-0.0003$ | $[-0.0030, +0.0023]$ | $[-0.0031, +0.0023]$ | $+0.0263$ | $-0.0039$ | BCa | $0.791$ | No |
| **Arm C vs Arm A** | DSC | Tumour (382, 370) | $+0.0118$ | $[+0.0009, +0.0240]$ | $[+0.0004, +0.0234]$ | $+0.0100$ | $+0.0117$ | BCa | $0.042$ | No |
| **Arm B vs Arm A** | DSC | Tumour (382, 370) | $+0.0099$ | $[-0.0009, +0.0215]$ | $[-0.0007, +0.0218]$ | $-0.0489$ | $+0.0107$ | BCa | $0.071$ | No |
| **Arm B vs Arm A** | LPR | Overlap (193, 188) | $\mathbf{+0.0478}$ | $\mathbf{[+0.0216, +0.0763]}$ | $[+0.0209, +0.0751]$ | $+0.0364$ | $+0.0003$ | BCa | $\mathbf{0.002}$ | No |

---

## 3. Inferential Conclusions & Audit Findings

1. **Fallback Frequency**: Across all 10 pooled comparisons, `bca_fallback_used` is **false (0% fallback)**. All quantile adjustments fell cleanly within $(0, 1)$ without degenerate bounds.
2. **Empty Draws**: Zero iterations were rejected during resampling (`empty_draws_count = 0`), confirming that the multi-frame cluster pool is sufficiently dense.
3. **Difference between BCa and Percentile Intervals**: BCa adjustments slightly expand the upper tail for right-skewed differences (e.g., $\Delta \text{DSC}_{C-B}$ upper bound moves from $+0.0120$ to $+0.0131$, $z_0 = 0.0803$), correcting for non-normality and cluster leverage.
4. **Finite-Sample $p$-Value Bound**: Two-sided $p$-values are bounded below by $\frac{1}{B} = 0.0005$. Values reported as $p = 0.0005$ indicate that zero of 2,000 bootstrap resamples crossed zero.

