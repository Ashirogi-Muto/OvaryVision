# OVATUS_ACCESS_REQUEST.md — Formal Research Data Request: OvaTUS-V1

> [!WARNING]
> **TRANSMISSION STATUS: DRAFT DOCUMENT ONLY — NOT YET SENT**  
> This access request has been drafted per the research protocol requirements. It has **NOT** been transmitted to the authors and will only be sent upon explicit authorization and review by the Principal Investigator.

---

**Date**: 2026-10-29  
**To**: Prof. Le Thi Lan (`lan.lethi1@hust.edu.vn`), Dr. Pham Thi Loan, and the SigM Lab Research Team  
**Affiliation**: Signal, Information, and Multimedia Content Processing Laboratory (SigM Lab),  
School of Electrical and Electronic Engineering, Hanoi University of Science and Technology (HUST),  
in collaboration with the National Hospital of Obstetrics and Gynecology (NHOG), Hanoi, Vietnam  
**From**: OvaryVision Research Team, School of Computer Science and Engineering, IILM University, Greater Noida  
**Subject**: Formal Academic Access & Label Request for OvaTUS-V1 Ovarian Ultrasound Dataset (Grant KC-4.0-45/19-25)

---

### 1. Statement of Research Purpose

Dear Prof. Le Thi Lan and the SigM Research Team,

We are conducting an independent academic medical-imaging study titled **OvaryVision**, investigating the diagnostic safety, image integrity, and probability calibration of deep convolutional neural networks (ConvNeXt-Small and EfficientNet-B3) applied to ovarian ultrasound computer-aided diagnosis (CAD). Specifically, our investigation focuses on whether non-generative zero-masking of clinician annotation symbols (calipers, measurement bars, text) can match generative symbol inpainting without introducing synthetic texture alterations or impairing diagnostic calibration.

We have reviewed your landmark contributions on the **OvaTUS-V1** benchmark dataset collected at NHOG Hanoi. To rigorously evaluate cross-center external generalizability across ultrasound hardware systems (Samsung Medison W80i and GE Voluson S6), we formally request academic research access to the OvaTUS-V1 dataset and its linked clinical records.

---

### 2. Specific Data Items Requested

To ensure our external validation complies with medical imaging standards (STARD-AI / TRIPOD-AI), we specifically request access to:

1. **Ultrasound Imaging Data**:
   - Access to the 439 transvaginal/transabdominal ultrasound scans across both Samsung Medison W80i and GE Voluson S6 hardware platforms.
2. **Histopathological Ground Truth Labels**:
   - Patient-matched post-operative surgical histopathology outcomes (confirmed **Benign vs. Borderline vs. Malignant** classification).
   - Specific pathological histology subtypes where available (e.g., serous cystadenoma, mucinous cystadenoma, mature cystic teratoma, endometrioma, thecoma, high-grade serous carcinoma, clear cell carcinoma).
   - *Rationale*: The publicly documented taxonomy provides 6 IOTA morphological categories (Solid, Multilocular, Unilocular, Dermoid, etc.). While valuable for morphological segmentation, morphology cannot be substituted for surgical histology when benchmarking diagnostic benign/malignant classification.
3. **De-Identified Examination / Subject Identifiers**:
   - De-identified subject/patient keys or examination identifiers to ensure that multi-frame cine sweeps or multi-angle acquisitions from the same patient are strictly grouped together, preventing train/test data leakage.
4. **Annotation Artifact Metadata**:
   - Documentation or masks indicating whether scans contain clinician measurement calipers, calipers, or demographic text overlays, or if they have undergone prior digital sanitization.

---

### 3. Ethical Compliance and Data Use Commitment

The OvaryVision research team formally commits to the following terms:

- **Strict Academic & Non-Commercial Use**: The data will be utilized solely for academic research and peer-reviewed scientific publication.
- **Zero Redistribution**: No image files, clinical metadata, or derivative patient data will be redistributed, publicly hosted, or transferred to third parties.
- **Privacy & Anonymity**: All patient privacy protections will be preserved. No attempts will be made to re-identify any individual.
- **Full Attribution & Citation**: All resulting publications will cite the foundational grant (*"Research and development of a computer-aided support system for ovarian cancer diagnosis using ultrasound images"*, Grant No. KC-4.0-45/19-25) and all relevant SigM Lab publications (including works on OvaTUS, CSA-SegNet, and related publications).
- **Compliance with Institutional Agreements**: We are prepared to execute a formal Data Use Agreement (DUA) or Material Transfer Agreement (MTA) with HUST and NHOG as required.

---

### 4. Current External Benchmark Status in OvaryVision

Pending receipt of confirmed surgical histopathology labels, the OvaryVision project enforces the following methodological protocol:
- **Gate D Policy**: Direct binary CAD classification on OvaTUS-V1 remains **formally blocked**.
- **Exploratory Alternative**: If only morphological labels can be authorized, OvaTUS-V1 will be evaluated strictly as an independent **IOTA 6-category morphological transfer benchmark**, and will not be conflated with binary pathology-confirmed diagnosis.

We welcome the opportunity to discuss collaborative research arrangements.

Sincerely,  
**The OvaryVision Research Team**  
School of Computer Science and Engineering, IILM University, Greater Noida  
Contact: `tanishaqbansal`

