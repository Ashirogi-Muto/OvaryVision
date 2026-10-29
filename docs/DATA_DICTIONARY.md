# DATA_DICTIONARY.md — Data Schemas & Field Contracts: OvaryVision

**Version**: 1.0.0  
**Specification Reference**: `SPEC.md` Section 2 & 4  

This document formalizes the exact column specifications, data types, constraints, and semantics for all manifests, audit reports, predictions, and experiment records used across the OvaryVision benchmark.

---

## 1. Image Manifest Schema (`data/manifests/inventory.csv`)

The central image manifest tracks every raw and preprocessed ultrasound image in the project.

| Field Name | Type | Required? | Description / Allowed Values | Constraint / Provenance |
|---|---|---|---|---|
| `image_id` | String | Yes | Unique identifier for the scan (e.g., `MMOTU_2D_0012`) | Unique primary key across datasets |
| `dataset_id` | String | Yes | Source dataset: `MMOTU_2D`, `MMOTU_CEUS`, `OvaTUS_V1` | Enum: `MMOTU_2D`, `MMOTU_CEUS`, `OvaTUS_V1` |
| `relative_image_path` | String | Yes | Relative path from project root to image file | Must exist on disk |
| `sha256_checksum` | String | Yes | SHA-256 hexadecimal digest of raw image bytes | 64-character lowercase hex string |
| `width` | Integer | Yes | Pixel width of image (e.g., `800`) | $\ge 64$ |
| `height` | Integer | Yes | Pixel height of image (e.g., `400`) | $\ge 64$ |
| `channels` | Integer | Yes | Color channels (typically `1` for grayscale, `3` for RGB) | `1` or `3` |
| `file_size_bytes` | Integer | Yes | Exact file size in bytes | $> 0$ |
| `diagnostic_label` | Integer | Yes | Binary classification label: `0` = Benign, `1` = Malignant | `0` or `1` (or `-1` if unlabeled) |
| `fine_grained_class` | String | No | Detailed histology (e.g., `serous_cystadenoma`) | Unresolved in public release |
| `patient_id` | String | No | Clinical patient identifier | *Unresolved in public MMOTU* |
| `study_id` | String | No | Clinical ultrasound examination/study identifier | *Unresolved in public MMOTU* |
| `pseudo_patient_cluster` | String | No | Perceptual-hash cluster ID for near-duplicates | Generated via dHash ($d_H \le 4$) |
| `artifact_mask_path` | String | No | Path to binary mask isolating calipers/annotations | $1 = \text{artifact}, 0 = \text{tissue}$ |
| `tumor_mask_path` | String | No | Path to binary mask isolating tumor lesion boundary | $1 = \text{tumor}, 0 = \text{background}$ |
| `has_visible_artifacts` | Boolean | Yes | Flag indicating whether scan has clinician calipers | True / False |

> [!CRITICAL]
> Under no circumstances may `artifact_mask_path` point to the same file as `tumor_mask_path`. Conflating the two invalidates the research benchmark.

---

## 2. Split Manifest Schema (`data/manifests/split_seed{seed}.csv`)

Defines reproducible, group-aware cross-validation and benchmark partitions.

| Field Name | Type | Required? | Description / Allowed Values |
|---|---|---|---|
| `image_id` | String | Yes | Corresponds to `image_id` in Image Manifest |
| `split_role` | String | Yes | Partition assignment: `train`, `val`, `test_internal`, `test_external` |
| `split_seed` | Integer | Yes | Seed used for pseudo-random partition generation (e.g., `42`, `1337`, `2026`) |
| `pseudo_patient_cluster` | String | No | Cluster ID ensuring frames from same sweep remain in single fold |
| `diagnostic_label` | Integer | Yes | Binary label `0` or `1` |

---

## 3. External Dataset Label Mapping Contract (OvaTUS-V1)

**Status**: **BLOCKED (UNVERIFIED MAPPING)**  
Authoritative inspection of the OvaTUS-V1 dataset (SigM Lab, HUST / NHOG Vietnam; 439 images) reveals that images are categorized according to **six IOTA (International Ovarian Tumor Analysis) morphological categories**, **not** histopathological diagnoses:

| OvaTUS-V1 Raw Class Name | IOTA Morphological Definition | Malignancy Risk Profile | Binary Benchmark Mapping Status |
|---|---|---|---|
| `Unilocular cyst` | Single-chamber cyst without solid components | < 1% (Almost universally benign) | **UNVERIFIED / PROVISIONAL** (Cannot assume without histology) |
| `Dermoid cyst` | Acoustic shadow, hyperechoic lines/dots (teratoma) | < 2% (Typically benign mature teratoma) | **UNVERIFIED / PROVISIONAL** (Cannot assume without histology) |
| `Multilocular cyst` | Multiple chambers without solid components | ~10% (Low-to-moderate risk) | **BLOCKED** (Heterogeneous risk; cannot map to 0 or 1) |
| `Unilocular-solid cyst` | Single cyst with solid papillary projection | ~30% – 40% (Substantial risk) | **BLOCKED** (High borderline/malignancy probability) |
| `Multilocular-solid cyst`| Multiple cysts with solid components | ~40% – 60% (High risk) | **BLOCKED** (Frequently malignant or borderline) |
| `Solid tumor` | Purely solid mass ($\ge 80\%$ solid tissue) | > 65% (Very high risk) | **BLOCKED** (Cannot map to 1 without histology) |

> [!CAUTION]
> **Scientific Integrity Gate**:
> In clinical gynecological ultrasonography, IOTA descriptors represent acoustic morphology, not confirmed histology. A "solid tumor" may be a benign fibroma/thecoma or a malignant adenocarcinoma; a "multilocular cyst" may be a benign cystadenoma or a borderline mucinous neoplasm.
> 
> **Resolution Protocol**:
> 1. Binary benign/malignant evaluation of OvaTUS-V1 is **strictly blocked** until patient-matched histopathological surgical outcome records are obtained and audited.
> 2. Alternative Approved Evaluation Design: If histological ground truth is absent, OvaTUS-V1 may be evaluated as a **morphological transfer benchmark** (evaluating how inpainting affects IOTA morphological classification accuracy or feature stability), or internal MMOTU evaluation will serve as the primary endpoint.

---

## 4. Prediction Manifest Schema (`outputs/metrics/preds_{exp_id}.csv`)

Serialized prediction outputs generated during evaluation.

| Field Name | Type | Required? | Description |
|---|---|---|---|
| `experiment_id` | String | Yes | Unique ID for the experiment run |
| `image_id` | String | Yes | Unique scan identifier |
| `preprocessing_arm` | String | Yes | `arm_a_raw`, `arm_b_inpainted`, or `arm_c_masked` |
| `backbone` | String | Yes | `convnext_tiny` or `efficientnet_b0` |
| `split_seed` | Integer | Yes | Seed used for training |
| `ground_truth` | Integer | Yes | `0` (Benign) or `1` (Malignant) |
| `raw_logit_benign` | Float | Yes | Uncalibrated logit for class 0 |
| `raw_logit_malignant` | Float | Yes | Uncalibrated logit for class 1 |
| `calibrated_prob_malignant` | Float | Yes | Probability after temperature scaling: $\sigma(z_1 / T)$ |
| `predicted_class` | Integer | Yes | Binary decision at threshold $\tau = 0.5$ |

---

## 5. Experiment Registry Record (`docs/EXPERIMENT_TRACKER.md`)

Fields recorded in the central experiment registry for full reproducibility:

| Field Name | Type | Description |
|---|---|---|
| `experiment_id` | String | Format: `EXP_{DATE}_{ARM}_{BACKBONE}_S{SEED}` |
| `timestamp_utc` | String | ISO 8601 UTC timestamp |
| `preprocessing_arm` | String | `arm_a`, `arm_b`, `arm_c` |
| `backbone` | String | Network backbone architecture name |
| `split_seed` | Integer | Random split seed |
| `git_commit_hash` | String | Git commit hash at execution time (or `no_git_repo`) |
| `config_hash` | String | SHA-256 digest of loaded experiment YAML configuration |
| `val_macro_f1` | Float | Best validation macro-F1 achieved |
| `test_macro_auc` | Float | Test set Area Under ROC Curve |
| `test_ece` | Float | Expected Calibration Error on test set |
| `optimal_temperature` | Float | Fitted temperature scaling parameter $T^*$ |
| `status` | String | `PLANNED`, `RUNNING`, `COMPLETED`, `FAILED` |
| `output_checkpoint` | String | Relative path to saved checkpoint weights |


---

## 6. Synthetic Benchmark & Image Integrity Schemas

### 6.1 Synthetic Artifact Metadata Record (`{case_id}_metadata.json`)

| Field Name | Type | Description |
|---|---|---|
| `case_id` | String | Unique synthetic benchmark case ID (e.g. `SYNTH_CASE_0001`) |
| `clean_path` | String | Absolute path to uncorrupted reference PNG |
| `corrupted_path` | String | Absolute path to overlay-corrupted PNG |
| `mask_path` | String | Absolute path to binary artifact mask PNG (scaled to 0/255) |
| `provenance.timestamp_utc` | String | ISO 8601 UTC creation timestamp |
| `provenance.clean_sha256` | String | SHA-256 digest of clean image array |
| `provenance.corrupted_sha256` | String | SHA-256 digest of corrupted image array |
| `provenance.mask_sha256` | String | SHA-256 digest of binary mask array |
| `provenance.mask_pixel_count` | Integer | Total number of pixels marked as artifact |
| `provenance.mask_fraction` | Float | Fraction of total image pixels covered by artifact mask |
| `provenance.artifacts_drawn` | List[Dict] | Geometries, coordinates, and distances for calipers drawn |

### 6.2 Image Integrity Report Schema (`ImageIntegrityReport`)

| Field Name | Type | Partition | Description |
|---|---|---|---|
| `global_mae` | Float | Full Image | Mean Absolute Error across all image pixels |
| `global_psnr` | Float | Full Image | Peak Signal-to-Noise Ratio (dB) across all pixels |
| `global_ssim` | Float | Full Image | Mean Structural Similarity Index across all pixels |
| `in_mask_mae` | Float | $\{ (u, v) \mid M = 1 \}$ | MAE evaluated strictly over artifact coordinates |
| `in_mask_psnr` | Float | $\{ (u, v) \mid M = 1 \}$ | PSNR (dB) evaluated strictly over artifact coordinates |
| `in_mask_ssim` | Float | $\{ (u, v) \mid M = 1 \}$ | SSIM evaluated strictly over artifact coordinates |
| `exterior_mae` | Float | $\{ (u, v) \mid M = 0 \}$ | MAE evaluated strictly over unmasked exterior tissue |
| `exterior_psnr` | Float | $\{ (u, v) \mid M = 0 \}$ | PSNR (dB) on exterior (expected: $\infty$ if no leakage) |
| `exterior_ssim` | Float | $\{ (u, v) \mid M = 0 \}$ | SSIM on exterior tissue |
| `local_texture_variance.var_in_mask` | Float | In-Mask | Variance of pixel intensities inside $M_{art}$ |
| `local_texture_variance.var_surrounding_band` | Float | $\partial M_{art}$ | Variance in surrounding 10-pixel parenchyma band |
| `local_texture_variance.delta_ltv` | Float | Difference | $|\sigma^2_{in} - \sigma^2_{band}|$ (variance divergence) |
| `lpips` | Float / Null | Full Image | Optional Learned Perceptual Patch Similarity (GPU only) |
| `scientific_limitations` | List[String] | Metadata | Explicit disclaimers regarding synthetic-to-real transfer |
