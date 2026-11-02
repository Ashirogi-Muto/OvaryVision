# P-Value Implementation & Inferential Audit (Revised Final Report)

**Audit Date**: 2026-11-02  
**Target Code**: `src/ovaryvision/statistics.py` (`paired_cluster_bootstrap_segmentation()`, lines 260–263)  
**Caller Script**: `scripts/aggregate_three_seed_benchmark.py` (lines 308–373)  
**Verification Script**: `scripts/verify_statistical_inference.py`  
**Test Suite**: `tests/test_statistics.py`  

---

## 1. Audit of Original Implementation & Formulation Flaws

In `src/ovaryvision/statistics.py`, the empirical $p$-value was originally calculated as:

```python
# Lines 260-263 of src/ovaryvision/statistics.py:
# Two-sided empirical p-value
p_val = 2.0 * min(float(np.mean(boot_diffs <= 0)), float(np.mean(boot_diffs >= 0)))
p_val = min(1.0, max(p_val, 1.0 / num_resamples))
```

where `boot_diffs` is the array of $B=2,000$ cluster bootstrap difference estimates:
$$\hat{\theta}^*_b = \frac{\sum_{k} m_{b, k} \text{cluster\_diff\_sums}[k]}{\sum_{k} m_{b, k} \text{cluster\_valid\_sizes}[k]}$$

### 1.1 Uncentered Bootstrap Distribution
The replicates $\hat{\theta}^*_b$ were resampled from observed data without centering under the null hypothesis:
$$\mathbb{E}[\hat{\theta}^*] \approx \hat{\theta}_{\text{obs}}$$
Evaluating $\mathbb{P}(\hat{\theta}^* \le 0)$ or $\mathbb{P}(\hat{\theta}^* \ge 0)$ measures the tail mass of the **alternative** distribution covering zero. It does not construct a null reference distribution under $H_0: \theta = 0$.

### 1.2 Mismatch Between P-Values and BCa Confidence Intervals
The original formula corresponds to the inversion of an equal-tailed **Percentile** confidence interval. However, the reported confidence intervals were **BCa (Bias-Corrected and Accelerated)** intervals, which correct for median bias ($z_0$) and jackknife skewness acceleration ($a$).
When the bootstrap distribution is asymmetric ($a \neq 0, z_0 \neq 0$), percentile quantiles diverge from BCa quantiles.

#### Concrete Inconsistency in Benchmark Data:
In Row 18 of the benchmark (`Arm C vs Arm A` on `fpr_extra`, Normal Ovary cohort $N=87$):
- **Observed Difference**: $+0.00251$
- **Reported 95% BCa CI**: $[+0.00032, +0.00590]$ (strictly positive, excluding 0)
- **Reported Original P-Value**: **$p = 0.0560$** ($p > 0.05$)!

**Root Cause**: 56 of 2,000 bootstrap replicates (2.8%) were $\le 0$. The unadjusted percentile tail yielded $2 \times 0.028 = 0.0560$, and the unadjusted percentile interval crossed zero ($[-0.000039, +0.005092]$).  
However, because $z_0 = +0.103$ and $a = +0.045$, the BCa quantile adjustment shifted the lower evaluation bound from the 2.5% quantile to the 1.5% quantile, pushing the 95% BCa interval lower bound above zero ($+0.00032$).  
Reporting $p = 0.0560$ alongside a 95% BCa CI excluding zero created a direct contradiction.

### 1.3 Distinguishing Zero Observed Extreme Draws from $p=0$
When zero of $B=2,000$ bootstrap replicates crossed zero, the unclipped percentile calculation yielded $2 \times 0.0 = 0.0$. Line 262 then applied an artificial clipping floor:
$$\max\left(0.0, \frac{1}{2000}\right) = 0.0005$$
This was an ad-hoc clipping floor artifact. A measured Monte Carlo $p$-value cannot be $0.0$, because observing zero extreme events in 2,000 draws does not prove the true null probability is zero; it simply bounds it from above. Reporting exact $p = 0.0005$ falsely implied an empirical frequency of $1/2000$. The mathematically correct presentation is an upper bound:
$$p < 0.001 \quad \text{or} \quad p < \frac{1}{B}$$

---

## 2. Null-Centered Bootstrap Formulation

To conduct a genuine bootstrap hypothesis test of $H_0: \theta = 0$ (Hall & Wilson 1991; Davison & Hinkley 1997, Sec. 4.2):

### 2.1 Null Imposition
Given $K=454$ pseudo-patient clusters, cluster total differences $S_k = \sum_{i=1}^{n_k} (y_{1, k, i} - y_{2, k, i})$, and cluster valid scan counts $n_k$:
The observed paired cluster difference is:
$$\hat{\theta}_{\text{obs}} = \frac{\sum_{k=1}^K S_k}{\sum_{k=1}^K n_k}$$
For bootstrap replicate $b$, cluster multiplicities $m_{b, k}$ are drawn via multinomial sampling with $\sum_k m_{b, k} = K$. The uncentered replicate is:
$$\hat{\theta}^*_b = \frac{\sum_{k=1}^K m_{b, k} S_k}{\sum_{k=1}^K m_{b, k} n_k}$$
We impose the null hypothesis $H_0: \theta = 0$ by shifting the bootstrap replicates by the observed estimate:
$$\tilde{\theta}^*_b = \hat{\theta}^*_b - \hat{\theta}_{\text{obs}}$$
Under this translation, $\mathbb{E}^*[\tilde{\theta}^*_b] \approx 0$. The distribution of $\tilde{\theta}^*_b$ directly models the sampling variation of $\hat{\theta} - \theta_0$ under $H_0$.

### 2.2 Preservation of Pairing and Cluster Multiplicity
Because clusters are sampled as coupled pairs across both arms simultaneously, the joint covariance structure between Arm 1 and Arm 2 across scans within each cluster is preserved identically. Scans from the same patient cluster always share the exact same integer multiplicity $m_{b, k}$ in both arms.

### 2.3 Finite-Sample Correction & Minimum Resolvable P-Value
The two-sided test statistic evaluates the proportion of null-centered replicates whose absolute magnitude equals or exceeds $|\hat{\theta}_{\text{obs}}|$. Following Davison & Hinkley (1997, Eq. 4.3) and Phipson & Smyth (2010), the unbiased add-one estimator is applied:
$$p_{\text{null}} = \frac{1 + \sum_{b=1}^B \mathbb{I}\left( |\tilde{\theta}^*_b| \ge |\hat{\theta}_{\text{obs}}| - 10^{-12} \right)}{1 + B}$$
For $B=2,000$:
- When zero extreme draws occur ($\sum \mathbb{I} = 0$):
  $$p_{\text{null}} = \frac{1 + 0}{1 + 2000} = \frac{1}{2001} \approx 0.00049975$$
- The minimum resolvable finite-sample point estimate is $\frac{1}{B+1} \approx 0.0005$.
- In manuscript tables, this resolution limit is presented as $p < 0.001$.

---

## 3. BCa Quantile Inversion

To ensure exact duality between the reported confidence intervals and the hypothesis tests, the $p$-value is derived by inverting the BCa distribution (Efron 1987; Hall 1992).

### 3.1 Mathematical Derivation
Let the BCa confidence interval for coverage $1 - 2\alpha$ be $[\hat{\theta}^*_{(\alpha_1)}, \hat{\theta}^*_{(\alpha_2)}]$, where:
$$\alpha_1 = \Phi\left( z_0 + \frac{z_0 + z_{\alpha}}{1 - a(z_0 + z_{\alpha})} \right), \quad \alpha_2 = \Phi\left( z_0 + \frac{z_0 + z_{1-\alpha}}{1 - a(z_0 + z_{1-\alpha})} \right)$$
where $z_0$ is the median bias correction, $a$ is the leave-one-cluster-out jackknife acceleration, and $z_{\alpha} = \Phi^{-1}(\alpha)$.

To find the two-sided $p$-value for $H_0: \theta = 0$, we find the significance level $\alpha$ at which the null value $\theta_0 = 0$ falls on the boundary of the confidence interval.
Let $q_0$ be the empirical proportion of bootstrap replicates $\le 0$ with mid-p rank tie correction:
$$q_0 = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\hat{\theta}^*_b < -10^{-12}) + \frac{1}{2B} \sum_{b=1}^B \mathbb{I}(|\hat{\theta}^*_b| \le 10^{-12})$$
clipped to $\left[\frac{1}{B+1}, \frac{B}{B+1}\right]$.
Inverting the BCa transformation:
$$z(q_0) = \Phi^{-1}(q_0)$$
$$w = \frac{z(q_0) - z_0}{1 + a(z(q_0) - z_0)} - z_0$$
The two-sided BCa-inverted $p$-value is:
$$p_{\text{BCa}} = 2 \times \min(\Phi(w), 1 - \Phi(w))$$
clipped to $\left[\frac{1}{B+1}, 1.0\right]$.

### 3.2 Exact Duality with BCa Confidence Intervals
Under this inversion:
- $p_{\text{BCa}} < 0.05 \iff 0 \notin [ \text{BCa}_{0.025}, \text{BCa}_{0.975} ]$.
- $p_{\text{BCa}} = 0.05 \iff 0$ lies exactly on the boundary of the 95% BCa CI.
- $p_{\text{BCa}} > 0.05 \iff 0 \in [ \text{BCa}_{0.025}, \text{BCa}_{0.975} ]$.

For Row 18 (`Arm C vs Arm A` on `fpr_extra`):
- 95% BCa CI: $[+0.00032, +0.00590]$ (excludes 0)
- $p_{\text{BCa}} = \mathbf{0.0205} < 0.05$.
The mathematical contradiction is completely resolved.

### 3.3 Primary Role of BCa Intervals
BCa confidence intervals remain the **primary presentation of inferential uncertainty** throughout the OvaryVision manuscript because they are second-order accurate ($O(1/n)$ coverage error vs $O(1/\sqrt{n})$ for standard percentile intervals) and invariant under monotonic transformations.
