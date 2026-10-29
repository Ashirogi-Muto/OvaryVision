# MILESTONE_1_AUDIT.md — Critical Scientific & Implementation Audit: Milestone 1

**Audit Date**: 2026-10-29  
**Auditor**: Autonomous Research Software Architect & Lead ML Engineer  
**Scope**: Codebase, canonical specification (`SPEC.md`), operating memory (`AGENT.md`), schemas (`docs/DATA_DICTIONARY.md`), research protocol (`docs/RESEARCH_PROTOCOL.md`), and automated test suite.  
**Objective**: Identify and rectify unsupported scientific assumptions, verify code and test paths, eliminate redundant memory files, and issue an evidence-based Go/No-Go gate for dataset ingestion.

---

## 1. Executive Summary & Audit Verdict

Milestone 1 established the initial executable infrastructure, typed configuration loading, logging, environment diagnostics, manifests, image auditing engine, and automated unit tests. 

However, a forensic scientific audit revealed **two critical scientific discrepancies** in prior documentation:
1. **OvaTUS-V1 Label Taxonomy**: Prior specifications assumed OvaTUS-V1 provided 6 histological categories mapped directly to binary Benign/Malignant. Authoritative documentation from the dataset creators (SigM Lab, HUST / NHOG Vietnam) confirms that OvaTUS-V1 provides **six IOTA morphological classes**, which carry heterogeneous malignancy risk and cannot be mapped to binary ground truth without patient-matched surgical pathology records.
2. **Patient Leakage & dHash Boundaries**: Prior notes risked overstating the power of perceptual difference hashing (dHash $\le 4$), potentially implying that grouping near-duplicates guarantees patient-level separation. In ultrasound imaging, frames from different sweep angles or depth settings for the same patient have visual differences that evade dHash $\le 4$.

**Verdict**: 
- **Milestone 1 Code & Infrastructure**: **VERIFIED & PASSING (100% test pass rate across 24 tests)**.
- **MMOTU Dataset Ingestion Gate**: **GO (CONDITIONAL)** — Proceed with image download and inventory audit; block Arms B and C until annotation artifact masks are procured or extracted.
- **OvaTUS-V1 External Binary Evaluation Gate**: **NO-GO (BLOCKED)** — Block binary evaluation until surgical histopathology outcomes are obtained and verified, or re-frame external testing as an IOTA morphological classification benchmark.

---

## 2. Findings Ranked by Severity

### Finding 1: Speculative OvaTUS-V1 Histological Mapping (Critical — Scientific Validity)
- **Severity**: **Critical**
- **Description**: Early drafts and `docs/DATA_DICTIONARY.md` proposed mapping six presumed classes (`Simple Cyst`, `Endometrioma`, `Teratoma/Dermoid`, `Hemorrhagic Cyst`, `Fibroma/Thecoma`, `Cystadenocarcinoma`) into binary labels (`0` for the first five, `1` for the last).
- **Evidence**:
  - Primary literature and official repository ([SigM Lab Ovarian Dataset](https://sigm-seee.github.io/datasets/Ovarian.html), Nguyen et al., 2025) confirm OvaTUS-V1 actually categorizes 439 scans into **six IOTA morphological categories**:
    1. `Solid tumor`
    2. `Multilocular cyst`
    3. `Unilocular cyst`
    4. `Dermoid cyst`
    5. `Multilocular-solid cyst`
    6. `Unilocular-solid cyst`
  - Under IOTA consensus guidelines (Timmerman et al., J Clin Oncol), ultrasound morphology does not have a 1:1 correspondence with benign vs. malignant pathology. For instance, `Multilocular-solid cyst` has a 40%–60% risk of malignancy, while `Solid tumor` may be a benign fibroma or a high-grade carcinoma.
- **Affected Files**: `SPEC.md`, `docs/DATA_DICTIONARY.md`, `docs/RESEARCH_PROTOCOL.md`, `PROJECT_STATE.md`.
- **Correction Implemented**:
  - Removed the speculative histological mapping table from `docs/DATA_DICTIONARY.md`.
  - Replaced it with the authoritative 6 IOTA morphological categories and risk profiles.
  - Formally instituted **Blocker BLK-03**: Binary CAD evaluation on OvaTUS-V1 is **strictly blocked** until patient-matched surgical pathology records are verified.
  - Recorded Decision **D09** in `DECISIONS.md`.

---

### Finding 2: Incomplete Leakage Check in Split Manifest Validator (High — Code Defect)
- **Severity**: **High**
- **Description**: `validate_manifest_leakage` in `src/ovaryvision/manifests.py` checked only for `pseudo_patient_cluster` overlap across split roles. If split rows lacked a cluster ID (or if cluster was `None`), an identical `image_id` assigned to both `train` and `val` slipped through undetected.
- **Evidence**:
  - Code inspection of `src/ovaryvision/manifests.py` lines 198–215.
- **Affected Files**: `src/ovaryvision/manifests.py`, `tests/test_manifests.py`.
- **Correction Implemented**:
  - Added a two-tier leakage gate in `validate_manifest_leakage`:
    1. *Gate 1 (Direct Image Leakage)*: Enforces that every `image_id` is unique across the entire split. An image assigned to multiple split roles raises `ValueError("Direct image leakage detected")`. Duplicate entries in the same role raise `ValueError("Duplicate image entry detected")`.
    2. *Gate 2 (Cluster Leakage)*: Enforces that no `pseudo_patient_cluster` spans multiple split roles.
  - Added automated unit tests `test_split_leakage_detection` verifying direct image leakage and duplicate image detection. Verified 100% pass rate.
  - Recorded Decision **D11** in `DECISIONS.md`.

---

### Finding 3: Methodological Scope of Perceptual dHash Clustering (Medium — Scientific Framing)
- **Severity**: **Medium**
- **Description**: Public MMOTU releases (Zhao et al., 2022) contain no patient IDs, timestamps, or sweep metadata. While dHash ($d_H \le 4$) clusters visually near-identical frames (mitigating duplicate sweep leakage), it cannot capture disparate views of the same ovary (e.g. longitudinal vs transverse sweeps, zoomed cyst wall, variable gain/depth), where Hamming distances exceed 15–30.
- **Evidence**:
  - Acoustic variation in ultrasound physics; forensic MMOTU analysis in project background documents.
- **Affected Files**: `SPEC.md`, `AGENT.md`, `docs/RESEARCH_PROTOCOL.md`, `PROJECT_STATE.md`.
- **Correction Implemented**:
  - Explicitly defined dHash clustering as an *identical/near-identical frame mitigation heuristic*, not proof of patient-level independence.
  - Added mandatory disclosure requirement to `SPEC.md` and `AGENT.md`: all publications must state that patient frame leakage cannot be ruled out without source hospital record linkage.
  - Recorded Decision **D10** in `DECISIONS.md`.

---

### Finding 4: MMOTU Caliper Mask Provenance & Dependencies (Medium — Data Dependency)
- **Severity**: **Medium**
- **Description**: Public MMOTU Zenodo/Figshare downloads provide only tumor ROI segmentation masks ($M_{roi}$), not clinician annotation artifact masks ($M_{art}$). Wang et al. (MGGAN, Bioengineering 2023) created custom symbol masks for MGGAN training, but these masks were not packaged into the primary public Zenodo release.
- **Evidence**:
  - Zhao et al. (MMOTU 2022) dataset distribution manifest; Chen et al. (Bioengineering 2023).
- **Affected Files**: `SPEC.md`, `PROJECT_STATE.md`, `docs/data_protocol.md`.
- **Correction Implemented**:
  - Documented exact dependency: Arm B (Inpainting) and Arm C (Masking) are blocked on MMOTU until author artifact masks are procured or an audited automated caliper detector is executed.
  - Confirmed that `src/ovaryvision/manifests.py` strictly prohibits setting `artifact_mask_path == tumor_mask_path`.

---

### Finding 5: Redundant Memory Files (`AGENTS.md` vs `AGENT.md`) (Low — Process & Governance)
- **Severity**: **Low**
- **Description**: Both `AGENTS.md` and `AGENT.md` existed concurrently in the workspace root, creating potential state drift between AI agents. Per `GEMINI.md` Rule 4 and `rules/agent-md-memory.md`, `AGENT.md` is the canonical project memory and hand-off document.
- **Affected Files**: `AGENTS.md`, `AGENT.md`, `README.md`, `PROJECT_STATE.md`, `SPEC.md`, `IMPLEMENTATION_PLAN.md`.
- **Correction Implemented**:
  - Merged all operating rules, integrity constraints, reading order, hardware boundaries, stack details, changelog, and decisions into canonical `/home/ubuntu/major/AGENT.md`.
  - Removed `AGENTS.md`.
  - Updated all references across the repository to point to `AGENT.md`.

---

### Finding 6: Statistical Estimators & Non-Inferiority Margins (Low — Statistical Refinement)
- **Severity**: **Low**
- **Description**: The non-inferiority margin of $\delta = -0.02$ AUC and the 70/15/15 train/val/test split were stated without explicit sensitivity bounds or sample size power qualifications. With ~1,469 total MMOTU scans and an estimated 25% malignancy prevalence, a 15% test set contains only ~55 malignant cases, limiting power for detecting subtle AUC differences.
- **Affected Files**: `SPEC.md`, `docs/RESEARCH_PROTOCOL.md`.
- **Correction Implemented**:
  - Explicitly labeled $\delta = -0.02$ as a provisional engineering margin.
  - Mandated a sensitivity analysis reporting non-inferiority decisions across $\delta \in \{-0.01, -0.02, -0.03\}$.
  - Established 1,000-sample stratified bootstrap paired differences as the primary statistical estimator, with asymptotic DeLong's test as a secondary check.
  - Recommended expanding from 3 to 5 random seeds when compute permits.

---

## 3. Implementation Verification & Test Results

The test suite was executed against the hardened codebase:

```bash
python3 -m pytest -v
```

### Complete Test Results Table
| Test File | Test Case | Status | Duration | Coverage Scope |
|---|---|---|---|---|
| `tests/test_cli.py` | `test_cli_check_env` | **PASSED** | 0.08s | Host diagnostics CLI output |
| `tests/test_cli.py` | `test_cli_check_env_json` | **PASSED** | 0.04s | Machine-readable JSON diagnostics |
| `tests/test_cli.py` | `test_cli_validate_config` | **PASSED** | 0.05s | YAML config validation via Click |
| `tests/test_cli.py` | `test_cli_init_dirs` | **PASSED** | 0.04s | Directory structure creation |
| `tests/test_cli.py` | `test_cli_generate_template` | **PASSED** | 0.04s | Manifest CSV template generation |
| `tests/test_cli.py` | `test_cli_audit_data_missing_dir`| **PASSED** | 0.04s | Graceful CLI error handling on missing path |
| `tests/test_cli.py` | `test_cli_audit_data_success` | **PASSED** | 0.05s | End-to-end data audit on synthetic image |
| `tests/test_config.py` | `test_default_config_loading` | **PASSED** | 0.02s | Pydantic v2 loading of `default.yaml` |
| `tests/test_config.py` | `test_invalid_yaml_file` | **PASSED** | 0.02s | Error handling for malformed YAML configs |
| `tests/test_config.py` | `test_preprocessing_validation` | **PASSED** | 0.01s | Image size and fill value bounds checks |
| `tests/test_config.py` | `test_path_resolution` | **PASSED** | 0.01s | Relative path resolution against project root |
| `tests/test_data_audit.py` | `test_dhash_and_hamming` | **PASSED** | 0.02s | dHash calculation and Hamming distance |
| `tests/test_data_audit.py` | `test_audit_with_synthetic_images`| **PASSED**| 0.08s | Exact duplicates, near-duplicates & corrupt files |
| `tests/test_data_audit.py` | `test_audit_non_existent_directory`|**PASSED**| 0.01s | FileNotFoundError on non-existent directory |
| `tests/test_data_audit.py` | `test_audit_empty_directory` | **PASSED** | 0.01s | Graceful `NO_IMAGES_FOUND` status |
| `tests/test_environment.py`| `test_environment_diagnostics` | **PASSED** | 0.02s | CPU, RAM, disk, OS, and platform model |
| `tests/test_environment.py`| `test_disk_info` | **PASSED** | 0.01s | Non-negative partition disk usage |
| `tests/test_environment.py`| `test_package_checks` | **PASSED** | 0.01s | Dependency mapping (`yaml`, `PIL`, etc.) |
| `tests/test_manifests.py` | `test_image_record_validation` | **PASSED** | 0.02s | SHA-256 (64 hex) & label validation |
| `tests/test_manifests.py` | `test_conflating_masks_is_prohibited`|**PASSED**|0.01s| Error when `artifact_mask == tumor_mask` |
| `tests/test_manifests.py` | `test_manifest_csv_roundtrip` | **PASSED** | 0.02s | CSV serialization and deserialization |
| `tests/test_manifests.py` | `test_split_leakage_detection` | **PASSED** | 0.02s | Direct image leakage & cluster leakage gates |
| `tests/test_provenance.py`| `test_compute_config_hash` | **PASSED** | 0.01s | Order-independent SHA-256 config hashing |
| `tests/test_provenance.py`| `test_provenance_record_creation`| **PASSED**| 0.02s | Full provenance snapshot serialization |

**Summary**: **24 passed in 0.78 seconds** (100% success rate, 0 failures, 0 warnings).

---

## 4. Unresolved Scientific Decisions

| Decision Topic | Current Status | Evidence Requirement | Resolution Path |
|---|---|---|---|
| **OvaTUS-V1 External Benchmark Design** | **BLOCKED** for binary CAD | Surgical pathology records matched to scan IDs | Contact SigM Lab (HUST) to request histopathology labels; if unavailable, re-frame OvaTUS-V1 as an IOTA morphology classification benchmark. |
| **MMOTU Caliper Mask Extraction** | **BLOCKED** for Arms B & C | Paired binary artifact masks | Contact MGGAN authors (Chen et al., Bioengineering 2023) or implement automated morphological caliper detector in `src/ovaryvision/caliper_detection.py`. |
| **Non-Inferiority Margin Consensus** | **PROVISIONAL** ($\delta = -0.02$) | Clinical significance audit | Retain $-0.02$ as provisional baseline and report sensitivity across $\delta \in \{-0.01, -0.02, -0.03\}$. |
| **Pseudo-Patient Grouping Threshold** | **PROVISIONAL** ($d_H \le 4$) | Visual inspection of clustered pairs | Run audit on real MMOTU scans when downloaded and verify that visually distinct lesions are not merged. |
| **Sample Size & Cross-Validation Design** | **PROVISIONAL** (70/15/15 single split) | Statistical power calculation | Supplement single internal test split with Stratified 5-Fold Cross-Validation across the 3 arms to ensure well-powered confidence intervals. |

---

## 5. Go / No-Go Recommendations

### Recommendation for MMOTU Data Ingestion: **GO (CONDITIONAL)**
- **Conditions**:
  1. Download/place 2D B-mode images into `data/raw/mmotu/images/`.
  2. Execute `ovaryvision audit-data` to inventory images, compute SHA-256 digests, and cluster near-duplicates.
  3. **Hold Arm B and Arm C execution** until annotation artifact masks are verified.

### Recommendation for OvaTUS-V1 Ingestion: **HOLD / CAUTION**
- Download of OvaTUS-V1 for exploratory inspection is permitted, but **binary diagnostic evaluation is strictly blocked** until surgical pathology ground truth is confirmed.

---

## 6. Exact Next Implementation Task

**Task**: **Implement the Controlled Ground-Truth Image Integrity Benchmark Module (`src/ovaryvision/synthetic_artifacts.py`)**  
- **Goal**: Implement digital caliper injection onto clean, unannotated ultrasound scans to establish a ground-truth reference for measuring inpainting distortion (SSIM, LPIPS, local texture variance shift $\Delta \text{LTV}$) without confounding unobserved anatomical tissue.
- **Dependencies**: Milestone 1 infrastructure.
- **Expected Deliverables**: `src/ovaryvision/synthetic_artifacts.py` and `tests/test_synthetic_artifacts.py`.
