# data_protocol.md — Data Ingestion, Mask Contracts & Privacy SOP: OvaryVision

This Standard Operating Procedure (SOP) governs data handling, ingestion pipelines, mask provenance, and privacy safeguards for the OvaryVision project.

---

## 1. Directory Structure Conventions

All datasets must be placed in `data/` adhering to the following structure:

```text
data/
├── raw/
│   ├── mmotu/
│   │   ├── images/              # Raw B-mode images (e.g., .jpg, .png)
│   │   ├── tumor_masks/         # Expert radiologist tumor ROI masks
│   │   └── labels.csv           # Image-level diagnostic annotations
│   └── ovatus_v1/               # External benchmark images & metadata
├── interim/
│   ├── artifact_masks/          # Verified clinician annotation artifact masks
│   └── near_duplicate_clusters/ # Perceptual hash grouping tables
├── processed/
│   ├── arm_a_raw/               # Normalized raw scans
│   ├── arm_b_inpainted/         # Inpainted scans (Harmonic/PDE Inpainting)
│   └── arm_c_masked/            # Non-generative zero-masked scans
└── manifests/
    ├── mmotu_inventory.csv      # Complete image inventory
    └── split_seed42.csv         # Train / Val / Test split assignments
```

---

## 2. Ingestion SOP for MMOTU

1. **Obtain Data**: Procure the official 2D B-mode subset (`OTU_2D`) from Zenodo (DOI: 10.5281/zenodo.21128657) or Figshare.
2. **Directory Placement**:
   - Place original images in `data/raw/mmotu/images/`.
   - Place tumor segmentation masks in `data/raw/mmotu/tumor_masks/`.
   - Place labels in `data/raw/mmotu/labels.csv`.
3. **Run Audit**:
   ```bash
   python3 -m ovaryvision.cli audit-data --data-dir data/raw/mmotu/images/ --output-dir outputs/metrics/
   ```
4. **Audit Review**: Confirm that zero files failed to load, check total image count (nominal 1,469 B-mode images), and inspect the near-duplicate cluster report.

---

## 3. Strict Mask Separation Contract

> [!CRITICAL]
> **Tumor ROI Mask $\neq$ Annotation Artifact Mask**
> 
> - **Tumor ROI Mask ($M_{roi}$)**: Delineates the clinical pathological lesion. Provided in official MMOTU releases. Used solely for spatial localization or reference.
> - **Annotation Artifact Mask ($M_{art}$)**: Delineates measurement calipers, crosses, lines, and text added by ultrasound machines or doctors.
> 
> *Constraint*: Applying Arm B (Inpainting) or Arm C (Zero-Masking) using $M_{roi}$ will destroy the entire lesion area, completely invalidating the benchmark.

### Annotation Mask Verification Protocol:
Before running preprocessing:
1. Verify that the artifact mask has pixel values $\in \{0, 1\}$.
2. Check that artifact area constitutes a small fraction of the image (typically $< 5\%$ of pixel area), whereas tumor masks typically span $10\% - 60\%$.
3. If an image contains no calipers, $M_{art} = \mathbf{0}$. The image passes through Arms A, B, and C identically.

---

## 4. Ingestion & Compatibility SOP for OvaTUS-V1

OvaTUS-V1 contains images from Samsung Medison W80i and GE Voluson S6 scanners.

1. **Access Authorization**:
   - Access is restricted under clinical data use agreements.
   - Contact the publishing institution (National Hospital of Obstetrics and Gynecology, Hanoi).
   - Once access is formally granted, place files in `data/raw/ovatus_v1/`.
2. **Label Compatibility Audit**:
   - OvaTUS-V1 classes must map to binary labels according to `docs/DATA_DICTIONARY.md` Section 3:
     - Class 0 (Benign): Simple Cyst, Endometrioma, Mature Teratoma, Hemorrhagic Cyst, Fibroma.
     - Class 1 (Malignant): Cystadenocarcinoma.
   - Run label validation:
     ```bash
     python3 -m ovaryvision.cli audit-data --data-dir data/raw/ovatus_v1/images/ --manifest data/raw/ovatus_v1/metadata.csv
     ```
   - If any unmapped classes are detected, the audit marks the external cohort as blocked until clinical resolution.

---

## 5. Data Privacy & Confidentiality Safeguards

1. **No Medical Scans in Git**: `data/raw/*`, `data/interim/*`, and `data/processed/*` are strictly excluded in `.gitignore`.
2. **Zero Protected Health Information (PHI)**:
   - File metadata must not contain patient names, national IDs, or clinic record numbers.
   - Any ultrasound images containing visible burned-in text displaying patient names or hospital identifiers must be flagged and masked before inclusion.
3. **No Credential Exposure**: Never commit `.env` or cloud storage access keys.

