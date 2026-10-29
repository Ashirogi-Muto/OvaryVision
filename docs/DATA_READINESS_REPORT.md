# DATA_READINESS_REPORT.md — OvaryVision Dataset Readiness and Forensic Audit

**Date**: 2026-10-29  
**Auditor**: Autonomous Senior ML Engineer & Methodology Lead  
**Audit Scope**: MMOTU (2D B-mode cohort) and OvaTUS-V1 (External Benchmark)  
**Overall Readiness Status**: **INTERNAL COHORT READY FOR ARM A; GATES B & D ENFORCED**

---

## 1. Executive Summary

This report establishes the empirical foundation and data readiness for the OvaryVision research project. A forensic audit was conducted on the ingested MMOTU cohort and the external OvaTUS-V1 dataset candidate to assess sample counts, integrity, near-duplicate clustering, class distributions, and mask provenance.

| Evaluation Dimension | MMOTU Internal Cohort | OvaTUS-V1 External Benchmark |
|---|---|---|
| **Primary Source** | Figshare Record 25058690 / Kaggle `orvile/mmotu-ovarian-ultrasound-images-dataset` | National Hospital of Obstetrics and Gynecology (NHOG), Hanoi / SigM Lab HUST |
| **Total Ingested Images** | 1,469 valid B-mode ultrasound scans | 439 images across Samsung Medison W80i & GE Voluson S6 |
| **Corrupt / Zero-Byte Files**| 0 files (100% readable) | N/A (Restricted / Agreement Required) |
| **Exact Duplicates** | 0 files | N/A |
| **Near-Duplicate Clusters** | 136 perceptual clusters (290 frames, $d_H \le 4$) | Unknown (Single-center acquisition, potential video sweep risk) |
| **Diagnostic Target** | 8 fine-grained histological classes $\rightarrow$ Binary Benign/Malignant | 6 IOTA morphological categories |
| **Binary Pathology Labels** | Verified (1,416 Benign / 53 Malignant) | **ABSENT** in public release |
| **Annotation Masks** | Tumor ROI segmentation masks (`_binary.PNG`) | Tumor boundary annotations |
| **Clinician Artifact Masks**| **ABSENT** in public release (Gate B Active) | Absent |
| **Readiness Status** | **READY for Arm A (Raw baseline)**; Synthetic Arm B/C Ready | **BLOCKED for Binary CAD** (Gate D Active) |

---

## 2. MMOTU Cohort Forensic Audit

### 2.1 Acquisition and Physical Integrity
- **Dataset Path**: `data/raw/mmotu/images/`
- **Scanned File Count**: 1,469 images.
- **Integrity Validation**: All 1,469 files read successfully via PIL and OpenCV with valid dimensions ($483 \times 542$ to $959 \times 537$, 3 RGB channels).
- **Exact Checksums**: 1,469 distinct SHA-256 hashes. No byte-level identical duplicates exist.

### 2.2 Near-Duplicate Perceptual Clustering (dHash Analysis)
Ultrasound examinations typically consist of video sweep captures or burst cine loops. Adjacent frames share identical acoustic speckle patterns and anatomical structures, creating severe cross-validation leakage if allocated to different splits.
- **Methodology**: Difference Hashing (dHash, 64-bit fingerprint) evaluated with a Hamming distance threshold $d_H \le 4$.
- **Audit Findings**:
  - Identified **136 near-duplicate clusters** spanning **290 images** (19.7% of the dataset).
  - Cluster sizes range from 2 to 4 near-duplicate frames.
  - The remaining 1,179 images were isolated as unique singletons.
  - Total independent cluster units: **1,315 clusters**.
- **Action Taken**: Pseudo-patient cluster IDs (`NEAR_DUP_0001` through `NEAR_DUP_0136`) were recorded in `data/manifests/mmotu_master_manifest.csv`. All split generation algorithms enforce cluster-level atomicity, guaranteeing that no cluster is ever split across train, validation, or test partitions.

### 2.3 Diagnostic Class Distribution & Binary Mapping
The official MMOTU metadata files (`train_cls.txt`, `val_cls.txt`) provide 8 fine-grained morphological and histological categories:

| Class ID | Fine-Grained Diagnosis | Category | Total Count | Proportion | Binary Label |
|:---:|:---|:---|:---:|:---:|:---:|
| 0 | Chocolate Cyst (Endometrioma) | Benign | 336 | 22.87% | 0 (Benign) |
| 1 | Serous Cystadenoma | Benign | 219 | 14.91% | 0 (Benign) |
| 2 | Teratoma (Dermoid Cyst) | Benign | 336 | 22.87% | 0 (Benign) |
| 3 | Theca Cell Tumor (Thecoma) | Benign | 88 | 5.99% | 0 (Benign) |
| 4 | Simple Cyst | Benign | 66 | 4.49% | 0 (Benign) |
| 5 | Normal Ovary | Non-neoplastic | 267 | 18.18% | 0 (Benign) |
| 6 | Mucinous Cystadenoma | Benign | 104 | 7.08% | 0 (Benign) |
| 7 | High-Grade Serous Carcinoma | Malignant | 53 | 3.61% | 1 (Malignant) |
| **Total** | | | **1,469** | **100.0%** | **1,416 B / 53 M** |

#### Critical Methodological Note on Binary Imbalance:
- **Prevalence**: 3.61% positive (malignant) prevalence ($26.7 : 1$ class imbalance).
- **Handling**: Standard Cross-Entropy loss will collapse toward majority-class predictions. All neural network pipelines strictly utilize **Class-Weighted Focal Loss ($\gamma = 2.0$)** with automated inverse class frequency weighting ($\alpha_0 = 0.519, \alpha_1 = 13.858$).
- **Primary Metrics**: Precision-Recall AUC (PR-AUC), ROC-AUC, Balanced Accuracy, and Brier Score are mandatory. Accuracy alone is non-informative.

---

## 3. Scientific Gate B Audit: Annotation Mask Provenance

### 3.1 Mask Forensic Discovery
In the MMOTU public release (Zenodo 21128657, Figshare 25058690, Kaggle `orvile/mmotu-ovarian-ultrasound-images-dataset`), files ending with `_binary.PNG` were audited.
- **Finding**: These masks correspond strictly to **tumor region-of-interest (ROI) segmentation boundaries**, covering between 2.1% and 34.8% of the ultrasound field of view.
- **Absence of Caliper Masks**: The public release does **NOT** contain clinician annotation artifact masks ($M_{art}$ representing measurement crosshairs, calipers, ellipses, or text overlays).
- **Literature Context**: Chen et al. (*Bioengineering* 2023) created a custom symbol-inpainting dataset (MGGAN), but the symbol masks were held as internal author assets and not deposited in the public distribution.

### 3.2 Methodological Enforcement
- **GATE B RULE**: The project strictly prohibits substituting tumor segmentation masks for clinician annotation masks. Inpainting or zero-masking a 30% tumor region would evaluate tumor removal, not artifact removal.
- **Current Operational Scope**:
  1. **Arm A (Raw Baseline)**: Fully executable on real MMOTU images.
  2. **Arm B (Inpainted) & Arm C (Masked)**: Executed on the controlled synthetic caliper benchmark (`data/interim/synthetic_benchmark`), where ground-truth artifact masks and uncorrupted reference tissues are mathematically verified.
  3. Real-image Arm B and Arm C remain conditionally blocked pending either (a) procurement of author annotation masks, or (b) deployment and manual radiologist verification of a dedicated high-precision caliper detection network.

---

## 4. Scientific Gate D Audit: OvaTUS-V1 External Benchmark

### 4.1 Dataset Inspection
- **Source**: National Hospital of Obstetrics and Gynecology (NHOG), Hanoi / SigM Lab HUST (`https://sigmlab.com/datasets/Ovarian/`).
- **Release Content**: 439 transvaginal/transabdominal ultrasound images acquired on Samsung Medison W80i and GE Voluson S6 systems.
- **Label Structure**: Annotations provide **6 IOTA morphological categories**:
  1. Solid tumor ($n = 65$)
  2. Multilobular cyst ($n = 137$)
  3. Unilocular cyst ($n = 76$)
  4. Dermoid cyst ($n = 100$)
  5. Multilobular-solid cyst ($n = 36$)
  6. Unilocular-solid cyst ($n = 25$)

### 4.2 Pathology Label Absence
- **Finding**: While the descriptive paper mentions postoperative histopathology, the publicly accessible dataset distribution does **NOT** provide patient-matched benign vs. malignant pathology labels.
- **Scientific Risk**: Assigning binary labels based solely on morphological appearance (e.g., assuming solid = malignant, unilocular = benign) is clinically invalid and scientifically fraudulent. Multilobular cysts and solid ovarian masses can be either benign (fibromas, thecomas, cystadenomas) or malignant.
- **GATE D RULE**: External binary CAD evaluation on OvaTUS-V1 is **STRICTLY BLOCKED** until authenticated surgical histopathology records are linked to the specific image identifiers.

---

## 5. Master Manifest and Split Manifest Verification

### 5.1 Master Manifest
- **Path**: `data/manifests/mmotu_master_manifest.csv`
- **Total Records**: 1,469 entries.
- **Fields**: `image_id`, `dataset_id`, `relative_image_path`, `sha256_checksum`, `width`, `height`, `channels`, `file_size_bytes`, `diagnostic_label`, `fine_grained_class`, `pseudo_patient_cluster`, `tumor_mask_path`, `has_visible_artifacts`.

### 5.2 Group-Stratified Multi-Seed Splits
Three independent split manifests were generated using the group-stratified splitting engine:
1. `data/manifests/mmotu_split_seed42.csv`
2. `data/manifests/mmotu_split_seed1337.csv`
3. `data/manifests/mmotu_split_seed2026.csv`

#### Partition Properties (Exact across all three seeds):
- **Training Cohort**: 1,029 images (992 Benign, 37 Malignant; 70.05%)
- **Validation Cohort**: 220 images (212 Benign, 8 Malignant; 14.98%)
- **Internal Test Cohort**: 220 images (212 Benign, 8 Malignant; 14.98%)
- **Cluster Isolation**: Verified. 0 near-duplicate pairs cross any split partition.
- **Target Balance**: Verified. Positive prevalence is strictly preserved at 3.6% ($\pm 0.05\%$) across all folds.

---

## 6. Actionable Next Steps

1. **Remote GPU Execution for Arm A**:
   - Dispatch ConvNeXt-Small and EfficientNet-B3 training across seeds 42, 1337, and 2026 using `mmotu_split_seed*.csv`.
   - Retrieve prediction logits, test AUROC, and calibration metrics.
2. **Post-Hoc Probability Calibration**:
   - Fit optimal temperature $T^*$ strictly on the validation fold logits.
   - Evaluate whether probability calibration improves on the test fold without altering discrimination (AUROC).
3. **Controlled Synthetic Benchmark**:
   - Maintain the synthetic benchmark as the primary quantitative proof for RQ2 and RQ3 ($\Delta \text{LTV}$, SSIM, exterior boundary preservation).

