# MMOTU_INGESTION_GUIDE.md — MMOTU Acquisition, Placement & Ingestion SOP

**Version**: 1.0.0  
**Associated Specification**: `SPEC.md` Section 2.1 & 4  
**Target Dataset**: Multi-Modality Ovarian Tumor Ultrasound (MMOTU)  

This Standard Operating Procedure (SOP) defines the protocol for obtaining, verifying, placing, and auditing the MMOTU dataset for the OvaryVision benchmark.

---

## 1. Authoritative Acquisition Sources

The MMOTU dataset was introduced by Zhao et al. (Beijing Shijitan Hospital, Capital Medical University). All data files must be procured strictly from authoritative, documented academic repositories:

| Resource | Identifier / URL | License / Access Terms |
|---|---|---|
| **Primary Paper** | Zhao et al., IEEE ISBI / [arXiv:2207.06799](https://arxiv.org/abs/2207.06799) (2022) | Open Access |
| **Zenodo Repository** | [DOI: 10.5281/zenodo.21128657](https://zenodo.org/records/21128657) | Creative Commons Attribution 4.0 International |
| **Figshare Repository**| [DOI: 10.6084/m9.figshare.25058690](https://figshare.com/articles/dataset/_zip/25058690) | CC-BY 4.0 / Apache 2.0 |
| **Official GitHub Code**| [cv516Buaa/MMOTU_DS2Net](https://github.com/cv516Buaa/MMOTU_DS2Net) | Open Source Research Code |

> [!IMPORTANT]
> **No Automatic Downloads**:
> The project infrastructure intentionally does **not** execute automated background downloads of large medical archives. The researcher must explicitly download the archive from the verified repository or supply a local file path.

---

## 2. Expected Dataset Contents & Hierarchy

The official MMOTU dataset package contains two modalities:
1. `OTU_2D`: 1,469 conventional B-mode ultrasound images in PNG/JPG format.
2. `OTU_CEUS`: 170 contrast-enhanced ultrasound images (excluded from the baseline 2D B-mode benchmark to maintain modality consistency).

### Expected Directory Layout on Local Host:
Extract or place files into the project repository adhering strictly to this layout:

```text
/home/ubuntu/major/data/
├── raw/
│   └── mmotu/
│       ├── images/              # 1,469 B-mode images (e.g. OTU_2D_*.jpg/.png)
│       ├── tumor_masks/         # Radiologist pixel-wise tumor ROI masks
│       └── labels.csv           # Image IDs and binary/8-class diagnostic labels
└── interim/
    └── artifact_masks/          # Clinician annotation artifact masks (Chen et al. / Caliper Detector)
```

> [!CRITICAL]
> **Tumor ROI Masks $\neq$ Annotation Artifact Masks**:
> - The masks included in the public MMOTU distribution are **Tumor ROI Segmentation Masks** ($M_{roi}$). They delineate the ovarian mass boundary.
> - They are **NOT** clinician annotation artifact masks ($M_{art}$).
> - Storing tumor masks in `data/interim/artifact_masks/` or using them for Arm B (Inpainting) or Arm C (Masking) will destroy the lesion and invalidate the benchmark.

---

## 3. Step-by-Step Ingestion & Audit Workflow

### Step 1: Place Archive or Extracted Files
- Option A: Place the downloaded archive at `/home/ubuntu/major/data/raw/mmotu.zip`.
- Option B: Extract images directly into `/home/ubuntu/major/data/raw/mmotu/images/`.

### Step 2: Run Opt-In Extraction (if using archive)
```bash
python3 scripts/ingest_mmotu.py --archive data/raw/mmotu.zip --confirm
```

### Step 3: Execute Forensic Image Audit
Run the automated image inventory and duplicate auditor:
```bash
python3 -m ovaryvision.cli audit-data --data-dir data/raw/mmotu/images/ --output-dir outputs/metrics/
```
Or using the script:
```bash
python3 scripts/ingest_mmotu.py --audit-only --data-dir data/raw/mmotu/images/
```

### Step 4: Audit Verification Checklist
The audit output ([`outputs/metrics/data_audit_report.json`](file:///home/ubuntu/major/outputs/metrics/data_audit_report.json)) and inventory ([`data/manifests/data_inventory.csv`](file:///home/ubuntu/major/data/manifests/data_inventory.csv)) must confirm:
- [ ] Exactly 1,469 valid 2D B-mode images discovered.
- [ ] 0 corrupt or unreadable image files.
- [ ] Exact duplicate clusters identified and recorded.
- [ ] Near-duplicate clusters identified via dHash ($d_H \le 4$).
- [ ] Patient IDs marked as absent (`patient_ids_available: false`).

---

## 4. Known Blockers & Post-Ingestion Policy

1. **Patient Leakage (BLK-01)**:
   - Because public MMOTU lacks patient IDs, the split manifest generator will cluster frames by perceptual dHash to prevent identical sweep frames from crossing folds.
   - All reports must disclose that pseudo-patient clustering does not guarantee complete patient-level independence.
2. **Artifact Mask Procurement (BLK-02)**:
   - Post-ingestion, Arm B (Inpainting) and Arm C (Masking) remain blocked until verified clinician caliper masks are procured from Chen et al. (*Bioengineering* 2023) or extracted using `src/ovaryvision/caliper_detection.py`.
3. **External Benchmark Policy (OvaTUS-V1)**:
   - Do **NOT** download OvaTUS-V1 without institutional approval from the National Hospital of Obstetrics and Gynecology (Hanoi).
   - Binary evaluation on OvaTUS-V1 remains blocked until histological ground-truth records are verified.

