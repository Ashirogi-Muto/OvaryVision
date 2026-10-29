# RESEARCH_PROTOCOL.md — Operational Experiment Protocol: OvaryVision

**Protocol Version**: 1.0.0  
**Associated Specification**: `SPEC.md` (v2.0.0)  
**Research Question**: Does non-generative masking match or exceed generative inpainting in downstream diagnostic discrimination and calibration on clean external ovarian ultrasound examinations?

---

## 1. Study Design & Cohort Flow

The study follows a standardized, paired comparative evaluation design across three image preprocessing arms:

```
                             [MMOTU 2D B-Mode Cohort]
                                   (N = 1,469)
                                        │
                         [Perceptual Near-Duplicate Audit]
                         (Cluster into Pseudo-Patient Groups)
                                        │
                         [Stratified Group-KFold Split]
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
               Training Fold      Validation Fold     Internal Test Fold
                   (70%)               (15%)               (15%)
                    │                   │                   │
         ┌──────────┴──────────┐        │                   │
         ▼          ▼          ▼        │                   │
       Arm A      Arm B      Arm C      │                   │
       (Raw)    (Inpaint)   (Masked)    │                   │
         │          │          │        │                   │
         └──────────┼──────────┘        │                   │
                    ▼                   ▼                   ▼
             Train ConvNeXt /     Optimize Temp.       Internal Test
              EfficientNet        Scaling (T)          Evaluation
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                            [External Independent Test]
                               (OvaTUS-V1 Benchmark)
                                        │
                            Paired Statistical Comparison
                         (DeLong, McNemar, Bootstrap CIs)
```

---

## 2. Experimental Preprocessing Arms

### Arm A — Raw Images (Shortcut Learning Baseline)
- **Input**: Original B-mode image as provided by the clinic/dataset.
- **Transform**: None. Measurement calipers, crosses, boundary annotations, and text are left untouched.
- **Purpose**: Quantifies how much deep networks rely on synthetic annotation cues during training.

### Arm B — Inpainting (Harmonic / Boundary Inpainting Reference)
- **Input**: Original image + binary annotation artifact mask ($M_{art}$).
- **Transformation**:
  1. $M_{art}$ is dilated by $r_{dil} = 2$ pixels using a morphological cross structuring element to prevent boundary artifacts.
  2. Inpainting fills pixels within the dilated mask using boundary-constrained fast-marching harmonic diffusion (Telea / Navier-Stokes).
  3. Strict exterior constraint: All pixels outside the dilated mask are preserved exactly:
     $$I_B(u, v) = \begin{cases} I_{inpaint}(u, v), & (u, v) \in \text{dilate}(M_{art}) \\ I_A(u, v), & (u, v) \notin \text{dilate}(M_{art}) \end{cases}$$
- **Literature Context & Provenance Disclosure**: While the motivating literature (Wang et al., 2023) proposed Mask-Guided GAN (MGGAN), MGGAN model weights were not publicly packaged with the MMOTU distribution. The operational benchmark executes harmonic boundary inpainting, providing a mathematically bounded, deterministic inpainting baseline without unverified generative pixel hallucination. Deep generative inpainting remains unexecuted.

### Arm C — Non-Generative Masking (Zero / Neutral Value Occlusion)
- **Input**: Original image + binary annotation artifact mask ($M_{art}$).
- **Transformation**:
  1. $M_{art}$ is dilated by $r_{dil} = 2$ pixels.
  2. All pixels within the dilated mask are set to constant zero (black):
     $$I_C(u, v) = \begin{cases} 0, & (u, v) \in \text{dilate}(M_{art}) \\ I_A(u, v), & (u, v) \notin \text{dilate}(M_{art}) \end{cases}$$
- **Zero Risk Guarantee**: No synthetic anatomical structures or texture patterns are hallucinated.

### Controlled Synthetic Caliper Benchmark (Known Ground Truth)
- **Purpose**: Because real ultrasound scans have unobserved anatomical tissue beneath calipers, this benchmark corrupts audited clean scans with known digital caliper overlays (`src/ovaryvision/synthetic_artifacts.py`).
- **Generated Geometry**: Crosshairs ($s = 12\text{px}$), connecting dashed measurement lines ($\text{dash} = 6\text{px}, \text{gap} = 4\text{px}$), and callout text (e.g. "D1: 28.4mm") with exact containment ($I_{clean} \neq I_{corrupted} \implies M_{art} = 1$).
- **Tri-Partition Reporting**:
  - Global: MAE, PSNR, SSIM across the entire scan.
  - In-Mask: MAE, PSNR, SSIM evaluated strictly over $\{ (u, v) \mid M_{art}(u, v) = 1 \}$.
  - Exterior: MAE, PSNR, SSIM evaluated strictly over $\{ (u, v) \mid M_{art}(u, v) = 0 \}$ (auditing zero boundary leakage).
  - Local Texture Variance ($\Delta \text{LTV}$): $|\sigma^2_{in} - \sigma^2_{band}|$ against surrounding 10-pixel parenchyma band.
- **Scientific Limitation Notice**: Synthetic benchmark results evaluate geometric filling capability on known digital overlays; they do not prove that generative inpainters preserve micro-pathology (such as papillary projections) on real clinical scans. Software tests are distinct from clinical results.

---

## 3. Training & Optimization Fairness Protocol

To ensure rigorous neutrality, model training parameters are identical across all arms:

1. **Backbone Architectures**:
   - `convnext_small`: Inverted bottleneck CNN with large $7 \times 7$ kernels (approved benchmark standard per Section 15.1; `convnext_tiny` permitted strictly for fast smoke-tests).
   - `efficientnet_b3`: Compound-scaled CNN with depthwise separable convolutions (approved benchmark standard per Section 15.1; `efficientnet_b0` permitted strictly for fast smoke-tests).
   - Initial weights: Official ImageNet-1K pre-trained weights.
2. **Loss Function**: Class-Weighted Focal Loss ($\gamma = 2.0$, $\alpha_c$ computed from training fold class frequencies).
3. **Optimizer**: AdamW ($\eta = 10^{-4}$, weight decay $10^{-4}$) with Cosine Annealing learning rate schedule.
4. **Data Augmentation**: Standard spatial transforms (horizontal flip $p=0.5$, subtle affine rotation $\pm 10^\circ$, affine scaling $[0.95, 1.05]$). No cutmix or random erasing is applied inside the lesion to prevent confounding with Arm C.
5. **Early Stopping**: Monitored on **Validation Macro-F1** (patience = 10 epochs, maximum 50 epochs).

---

## 4. Probability Calibration Protocol

1. **Uncalibrated Output**: Raw logits $z_i \in \mathbb{R}^2$ from the best validation checkpoint.
2. **Temperature Parameter Optimization**:
   - Find $T^* > 0$ that minimizes negative log-likelihood on the **Validation set**:
     $$T^* = \arg\min_T \left( -\sum_{i=1}^{N_{val}} \log \sigma(z_i / T) \right)$$
3. **Application**: Apply $T^*$ to test set logits: $\hat{p}_i = \sigma(z_i / T^*)$.
4. **Strict Isolation**: The external test set is never used to fit or select $T^*$.

---

## 5. Statistical Hypothesis Testing Protocol

- **Primary Hypothesis Test**: Non-inferiority test of Arm C vs. Arm B on test Macro-AUC:
  $$H_0: \text{AUC}_C - \text{AUC}_B \le \delta \quad \text{vs.} \quad H_1: \text{AUC}_C - \text{AUC}_B > \delta$$
  where $\delta = -0.02$ is a provisional non-inferiority margin. A mandatory sensitivity analysis must evaluate margins $\delta \in \{-0.01, -0.02, -0.03\}$.
- **Primary Statistical Estimator**: Stratified 1,000-sample bootstrap paired difference confidence intervals ($95\%$ BCa / percentile method) computed across identical test image sets.
- **Secondary Asymptotic Check**: DeLong's test for paired ROC curves on the identical test cohort.
- **Paired Classification Decisions**: McNemar's test on binary 0/1 outcome concordance matrices at threshold $\tau = 0.5$ and optimal $\tau^*$.
- **External Evaluation Gate**: External binary evaluation on OvaTUS-V1 is **strictly blocked** until patient-matched surgical pathology records are verified, because OvaTUS-V1 provides 6 IOTA morphological categories rather than confirmed benign/malignant histology.
- **Multi-Seed Replication**: All experiments are executed across 3 fixed seeds ($42, 1337, 2026$) minimum, with 5 seeds recommended. Reports must present mean $\pm$ standard deviation across seeds.

