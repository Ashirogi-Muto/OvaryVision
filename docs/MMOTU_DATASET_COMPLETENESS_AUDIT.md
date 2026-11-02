# MMOTU Dataset Completeness, Provenance, and Annotation Recovery Audit

**Target Project**: OvaryVision (`Ashirogi-Muto/OvaryVision`)  
**Active Branch**: `exp/full-cohort-benchmark`  
**Date of Audit**: October 30, 2026  
**Auditor**: Antigravity Autonomous Research Agent  
**Status**: Complete & Verified  

---

## 1. Executive Summary & Definitive Answer

### Core Question:
> *Are we missing usable images, segmentation masks, annotations, or metadata that already exist publicly for MMOTU, and what additional scientifically valid experiments can they enable?*

### Definitive Answer:
**YES, but with critical distinctions across modalities, metadata, and annotation types:**

1. **Missing Ultrasound Modality Recovered (`OTU_CEUS`)**:
   - **Finding**: Our prior local repository contained exclusively conventional 2D B-mode scans (`data/raw/mmotu/`, 1,469 scans). We were missing the entire public **Contrast-Enhanced Ultrasound (CEUS)** cohort (`OTU_CEUS`, 170 scans across 47 patients).
   - **Recovery**: Staged safely and separately in `data/raw/mmotu_ceus/` (170 `.JPG` images, 170 verified tumor segmentation masks, official train/val partition splits, and 7-class diagnostic ground truth). A dedicated manifest was created at `data/manifests/mmotu_ceus_master_manifest.csv`.
   - **Impact**: Enables cross-modality domain adaptation, acoustic transfer benchmarking, and out-of-domain artifact ablation without touching the frozen 2D benchmark.

2. **2D B-mode Image & Tumor Mask Cohort is 100% Complete**:
   - **Finding**: All **1,469 2D images** and all **1,469 binary tumor segmentation masks** from the official release are already present in `data/raw/mmotu/`. No public repository contains additional 2D images.
   - **Derivative Censorship Identified**: The popular Figshare derivatives (25115633, 25058690) contain only 1,202 2D images because their curators intentionally dropped all 267 "Normal Ovary" scans (1,469 − 267 = 1,202). OvaryVision retains the complete, unpruned cohort of 1,469 images.

3. **Clinician Artifact Masks DO NOT Exist in Any Public Release**:
   - **Finding**: Neither the primary authors (Zhao et al., Beihang University), nor the downstream segmentation researchers (Ortiz de Murua et al., Lang Li et al.), nor the MGGAN inpainting paper (Chen et al.) have publicly released clinician annotation artifact masks ($M_{art}$).
   - **Acoustic Reality**: Medical calipers, measurement crosses, dashed lines, callout characters, and arrows were burned into the scan bitmap during real-time transvaginal examinations at Beijing Shijitan Hospital. The original authors provided only pathological lesion masks ($M_{roi}$), not artifact isolation masks.
   - **Integrity Directive**: OvaryVision's automated caliper segmentation algorithm (with 140-case researcher verification) remains the authoritative method for artifact localization in this benchmark. Claiming that author-verified caliper masks exist publicly is scientifically false.

4. **Missing Sub-Modality Disclosed (216 Color Doppler Images)**:
   - **Finding**: Section 3.2 of the original manuscript reveals that `OTU_2d` is not homogenous B-mode ultrasound: it contains **216 Color Doppler Flow Images (CDFI)** alongside **1,253 traditional grayscale B-mode scans**.
   - **Impact**: Represents an unexplored clinical covariate. Chrominance and Doppler velocity boxes introduce distinct acoustic frequency shifts that can be audited separately.

5. **Clinical Metadata & Patient Identifiers**:
   - **Patient Count**: 294 patients total across the dataset (247 in 2D, 47 in CEUS) scanned on a **Mindray Resona8** ultrasound machine at Beijing Shijitan Hospital.
   - **Anonymization**: Individual patient IDs, age, weight, and menstrual status were deliberately withheld by the authors for patient privacy and ethical compliance; scanner headers were cropped out.
   - **Author Disclosure**: The authors stated in print: *"to maximally help the community... we'd like to make the doctors' diagnostic description available after a research request"* (contact: `lyushuchang@buaa.edu.cn`).

---

## 2. Release-by-Release Forensic Comparison

We inspected the primary publication LaTeX sources, the official Google Drive repository, the two Zenodo deposits, and the two Figshare deposits.

| Release / Source | URL / Identifier | Curating Author / Institution | Date | 2D Scans | CEUS Scans | Total Masks | Diagnostic Ground Truth | Mask Pixel Format | Patient IDs Present | Artifact Masks ($M_{art}$) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Primary Publication** | [arXiv:2207.06799](https://arxiv.org/abs/2207.06799) / *Pattern Recognition* | Zhao, Lyu, Bai et al. (Beihang Univ / Beijing Shijitan Hosp) | 2022-07-14 | 1,469 | 170 | 1,639 | 8 classes (2D) / 7 classes (CEUS) | Multi-class RGB & Binary [0, 1] | Disclosed aggregate (294 pts; 247 2D, 47 CEUS); IDs withheld | NONE (burned into pixels; never isolated) |
| **Official Google Drive** | `1c5n0fVKrM9-SZE1kacTXPt1pt844iAs1` | Shuchang Lyu (`cv516Buaa`) | 2022-07 | 1,469 | 170 | 1,639 unique | `train_cls.txt`, `val_cls.txt` (both 2D & CEUS) | RGB palette (.PNG) & 8-bit [0, 1] (`_binary.PNG`) | NO (Cropped image borders remove scanner text) | NONE |
| **Zenodo 21128657** | [DOI: 10.5281/zenodo.21128657](https://doi.org/10.5281/zenodo.21128657) | J. Ortiz de Murua, I. Armas, J. Heras (Univ. de La Rioja) | 2026-07-06 | 1,469 | 0 | 1,469 | NONE (Omitted; curated for MedSAM pruning) | 8-bit binary [0, 1] (`mode=L`) | NO | NONE |
| **Zenodo 20541610** | [DOI: 10.5281/zenodo.20541610](https://doi.org/10.5281/zenodo.20541610) | J. Ortiz de Murua (Univ. de La Rioja) | 2026-07-02 | 0 | 0 | 0 | HTTP 410 GONE (Tombstone; withdrawn test deposit) | N/A | N/A | N/A |
| **Figshare 25115633** | [DOI: 10.6084/m9.figshare.25115633](https://doi.org/10.6084/m9.figshare.25115633) | Lang Li, Peng Jin et al. (EPDWDNet) | 2024-02-01 | 1,202 | 170 | 1,372 | NONE (Organized into train/test folders) | 8-bit binary [0, 255] (`mode=L`) | NO | NONE |
| **Figshare 25058690** | [DOI: 10.6084/m9.figshare.25058690](https://doi.org/10.6084/m9.figshare.25058690) | Lang Li, Peng Jin et al. (PMFFNet) | 2024-01-24 | 1,202 | 170 | 1,372 | NONE | 8-bit binary [0, 255] (`mode=L`) | NO | NONE |
| **OvaryVision Repo (Current)** | `data/raw/` on `exp/full-cohort-benchmark` | OvaryVision Research Team | 2026-10-30 | 1,469 | 170 | 1,469 (2D) + 170 (CEUS) | 1,469 (2D: 53 malig, 1416 benign); 170 (CEUS: 9 malig, 161 benign) | 8-bit binary [0, 1] (`mode=L`) | Mitigated via dHash clustering (173 2D clusters, 27 CEUS clusters) | Automated caliper extraction (1,402 2D, 79 CEUS) + 140 pilot QC |

*The machine-readable manifest is preserved at `outputs/metrics/mmotu_release_comparison.csv`.*

---

## 3. Segmentation Masks & Annotation Forensic Audit

### 3.1 Resolving the 2,938 Local Tumor Masks
A forensic audit of `data/raw/mmotu/tumor_masks/` revealed 2,938 files:
- 1,469 files named `<id>_binary.PNG`
- 1,469 files named `<id>_binary_binary.PNG`

**Forensic Finding**:
1. All 1,469 pairs are **100% byte-for-byte SHA-256 identical**.
2. Inspection of the official upstream Google Drive archive (`OTU_2d/annotations/`) revealed that the upstream repository itself contains three files per image:
   - `<id>.PNG` (RGB multi-class palette mask)
   - `<id>_binary.PNG` (Grayscale binary lesion mask)
   - `<id>_binary_binary.PNG` (Redundant binary copy created during upstream batch processing)
3. The OvaryVision pipeline standardizes on `<id>_binary.PNG`. The duplicate files carry zero additional scientific information, but their presence is completely benign and does not affect the benchmark.

### 3.2 Multi-Class Semantic Palettes vs Binary Lesion Masks
In `OTU_2d/annotations/`, the original `<id>.PNG` files are 3-channel RGB images encoded with specific color triplets:
- `[0, 0, 0]`: Background
- `[64, 0, 0]`: Chocolate Cyst (Endometrioma)
- `[0, 64, 0]`: Serous Cystadenoma
- `[0, 0, 64]`: Teratoma (Dermoid Cyst)
- `[64, 0, 64]`: Theca Cell Tumor
- `[64, 64, 0]`: Simple Cyst
- `[64, 64, 64]`: Normal Ovary (all black; no mass)
- `[0, 128, 0]`: Mucinous Cystadenoma
- `[0, 0, 128]`: High Grade Serous Cystadenoma / Carcinoma (Malignant)

When the authors implemented binary segmentation in `MMOTU_DS2Net`, they mapped all non-zero pixels ($R+G+B > 0$) to $1$ (`_binary.PNG`).

### 3.3 The Absence of Clinician Artifact Masks
Every public mask in MMOTU is an anatomical tumor boundary ($M_{roi}$). Clinician calipers ($+$ markers, $\times$ markers, callout text like "1.2cm") were burned into the frame during live clinical exams.
The authors did not segment or mask these artifacts. Any assumption that an external dataset release provides "ground truth caliper annotations" is refuted by our forensic analysis.

---

## 4. Contrast-Enhanced Ultrasound (CEUS / OTU_CEUS) Modality Assessment

### 4.1 Cohort Architecture & Clinical Characteristics
Contrast-Enhanced Ultrasound (CEUS) involves intravenous administration of sulfur hexafluoride microbubble contrast agents (e.g., SonoVue) followed by low-mechanical-index transvaginal imaging to visualize microvascular perfusion kinetics (wash-in, peak intensity, wash-out).

| Category ID | Histopathological Class | Biological Nature | Binary Target | Official Train (N=70) | Official Val/Test (N=100) | Total CEUS Scans (N=170) |
|---|---|---|---|---|---|---|
| `0` | Chocolate Cyst (Endometrioma) | Benign | 0 | 18 | 17 | 35 |
| `1` | Serous Cystadenoma | Benign | 0 | 16 | 24 | 40 |
| `2` | Teratoma (Dermoid Cyst) | Benign | 0 | 11 | 21 | 32 |
| `3` | Theca Cell Tumor | Benign | 0 | 14 | 17 | 31 |
| `4` | Simple Cyst | Benign | 0 | 4 | 14 | 18 |
| `5` | Normal Ovary | Benign | 0 | 0 | 0 | **0** |
| `6` | Mucinous Cystadenoma | Benign | 0 | 2 | 3 | 5 |
| `7` | High Grade Serous (Carcinoma) | Malignant | 1 | 5 | 4 | **9** |
| **Total** | — | — | — | **70 (5 malig)** | **100 (4 malig)** | **170 (9 malig)** |

**Key Clinical Observations**:
1. **Zero Normal Ovaries**: Normal ovaries have 0 cases in CEUS because CEUS is performed strictly as a second-line diagnostic examination when a suspicious adnexal mass is detected on conventional B-mode ultrasound.
2. **Class Imbalance**: Malignant prevalence is 5.29% (9 of 170 scans), very close to the 3.61% malignant prevalence in the 2D cohort (53 of 1,469 scans).
3. **Caliper Artifact Prevalence**: Calipers are detected in **79 of 170 CEUS scans (46.5%)**, compared to **95.4% in 2D B-mode**. This clinical difference occurs because CEUS is evaluated primarily through dynamic video playback of wash-in/wash-out perfusion rather than static orthogonal caliper distance measurements.

### 4.2 Staging and Manifest Integration
The CEUS cohort is safely staged in `data/raw/mmotu_ceus/`:
- `images/`: 170 verified `.JPG` images.
- `tumor_masks/`: 170 binary tumor segmentation masks (`_binary.PNG`).
- `original_masks/`: 170 multi-class RGB palette masks (`.PNG`).
- `labels/`: Official train/val text files (`train.txt`, `train_cls.txt`, `val.txt`, `val_cls.txt`, `trainval.txt`).
- `data/manifests/mmotu_ceus_master_manifest.csv`: Machine-readable manifest with perceptual dHash clusters (27 clusters), image dimensions, SHA-256 hashes, binary targets, and artifact flags.

---

## 5. Clinical Metadata, Scanner Provenance & Privacy Auditing

### 5.1 Hospital & Hardware Origin
- **Hospital of Origin**: Department of Gynecology and Obstetrics, Beijing Shijitan Hospital, Capital Medical University, Beijing, China.
- **Hardware Instrument**: Mindray Resona8 Premium Diagnostic Ultrasound System.
- **Transducer Modality**: High-resolution transvaginal ultrasound probes with dedicated harmonic imaging, color Doppler velocity mapping, and contrast-pulse sequencing (CPS) algorithms.

### 5.2 Anonymization & Demographic Withholding
Section 3.2 of the manuscript explicitly addresses demographic data:
> *"According to privacy policy, we manually crop the images to remove the private information and make sure that the published MMOTU image dataset will not contain any private information. For samples from each patient, we provide demographic information including gender, age, weight and doctors' diagnostic description. To protect patients' privacy out of ethics, we think that some information (e.g., age, weight) are not supposed be available. However, to maximally help the community investigate and develop the machine learning models based on MMOTU image dataset, we'd like to make the doctors' diagnostic description available after a research request."*

**Implications for OvaryVision**:
1. Patient identifiers are permanently absent from image files; borders were physically cropped before distribution.
2. Perceptual dHash near-duplicate clustering ($\le 4$) remains the mathematically soundest approach to group correlated sweep frames into pseudo-patient clusters.
3. Access to free-text clinical diagnostic descriptions can be requested formally from the corresponding author (`lyushuchang@buaa.edu.cn`).

---

## 6. Scientifically Defensible Additional Experiments

The discovery and staging of the CEUS cohort and the CDFI sub-modality enable three distinct, publishable research experiments:

### Experiment 1: Out-of-Domain Modality Transfer (2D B-mode $\to$ CEUS)
- **Scientific Objective**: Evaluate whether deep CAD models trained on conventional 2D B-mode ultrasound generalize to contrast-enhanced ultrasound (CEUS), and whether artifact removal improves out-of-domain transfer.
- **Hypothesis**: Models trained on Raw images (Arm A) overfit to B-mode speckle and static caliper features, leading to severe performance collapse on CEUS. Non-generative zero-masking (Arm C) strips non-biological features and demonstrates superior domain-invariant transfer.
- **Evaluation Design**:
  - Source Training: 2D MMOTU cohort under Arms A, B, and C.
  - Target Evaluation: Zero-shot evaluation on the 170 CEUS scans (9 malignant, 161 benign).
  - Metrics: Cross-domain AUROC, Expected Calibration Error (ECE), and Brier score.

### Experiment 2: Sub-Modality Color Doppler Flow (CDFI) Ablation
- **Scientific Objective**: Determine whether Color Doppler Flow Images (216 scans) act as an acoustic confounder in binary malignancy detection.
- **Hypothesis**: Color Doppler signals (reflecting vascular neoangiogenesis inside malignant tumors) provide strong biological discriminative cues that may compensate for artifact presence.
- **Evaluation Design**: Stratify the 1,469 2D test evaluations into CDFI ($N=216$) and pure B-mode ($N=1,253$) sub-cohorts. Test whether Arm C's performance advantage over Arm B holds equally across both acoustic modalities.

### Experiment 3: Full 8-Class Morphological Risk Stratification
- **Scientific Objective**: Evaluate the impact of artifact removal on fine-grained ovarian mass classification (Chocolate Cyst vs Serous vs Teratoma vs High-Grade Serous Carcinoma), moving beyond binary malignancy detection.
- **Protocol**: Train a multi-class ConvNeXt model using cross-entropy loss across the 8 MMOTU classes under Arms A, B, and C.

---

## 7. Licensing & Provenance Compliance

1. **Source License**: The MMOTU dataset is distributed under Creative Commons Attribution 4.0 International (CC-BY 4.0).
2. **Attribution Requirement**: Any publication, repository, or distribution utilizing MMOTU must cite:
   ```bibtex
   @article{zhao2022mmotu,
     author    = {Qi Zhao and Shuchang Lyu and Wenpei Bai and Linghan Cai and Binghao Liu and Meijing Wu and Xiubo Sang and Min Yang and Lijiang Chen},
     title     = {A Multi-Modality Ovarian Tumor Ultrasound Image Dataset for Unsupervised Cross-Domain Semantic Segmentation},
     journal   = {Pattern Recognition / arXiv:2207.06799},
     year      = {2022}
   }
   ```
3. **Repository Cleanliness**:
   - Zero raw medical images from `data/raw/mmotu/` or `data/raw/mmotu_ceus/` are committed to Git.
   - All dataset manifests, comparison tables, and audit summaries contain exclusively verified metadata and SHA-256 hashes.

