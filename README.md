# OvaryVision: Preprocessing Benchmark for Ovarian Ultrasound CAD

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests: 69 Passing](https://img.shields.io/badge/tests-69%20passing-brightgreen.svg)]()
[![Hardware: CPU Demo + GPU Benchmark](https://img.shields.io/badge/hardware-CPU%20Demo%20%7C%20CUDA%20GPU-orange.svg)]()

A rigorous, data-centric medical-imaging research benchmark investigating whether **non-generative artifact masking** (zero-occlusion of clinician annotations) matches or outperforms **generative inpainting** (biharmonic / deep inpainters) in downstream diagnostic performance and probability calibration on ovarian ultrasound computer-aided diagnosis (CAD).

---

## 1. Research Overview

- **The Problem**: Clinical transvaginal ultrasound scans frequently feature clinician-added measurement calipers, crosses, and text overlays placed directly over or adjacent to tumors. Deep neural networks exploit these synthetic cues as predictive shortcuts (the *Clever Hans* effect). In MMOTU, **98.1% of malignant scans contain calipers**, creating an overwhelming shortcut for uncleaned models.
- **The Core Research Question**: Does non-generative zero-masking match or exceed generative inpainting in diagnostic discrimination (AUROC) and calibration (ECE/Brier) on clean ultrasound scans, while avoiding parenchymal hallucination and reducing computational overhead?
- **Experimental Arms**:
  - **Arm A (Raw Baseline)**: Scans evaluated with clinician annotations intact.
  - **Arm B (Harmonic Inpainting)**: Strict in-mask biharmonic Poisson inpainting with a 2-pixel boundary dilation rule ($MAE_{ext} = 0.000000, PSNR_{ext} = \infty$).
  - **Arm C (Non-Generative Masking)**: Annotation regions zero-filled ($I=0$) with a 2-pixel boundary dilation rule ($MAE_{ext} = 0.000000, PSNR_{ext} = \infty$).
- **Benchmark Architectures**: `ConvNeXt-Small` and `EfficientNet-B3` across 3 repeated group-stratified Monte Carlo seeds (42, 1337, 2026).
- **Primary Metrics**: Test AUROC, Expected Calibration Error (ECE), Calibrated Brier Score ($T^*$), Exterior MAE/PSNR, Structural Similarity (SSIM), and Local Texture Variance ($\Delta \text{LTV}$).

---

## 2. Quick-Start Demo (CPU-Only, No Data Download Required)

You can run the complete tri-partition synthetic benchmark demo on any standard CPU in **less than 2 seconds** without downloading clinical datasets or requiring a GPU:

```bash
# 1. Clone repository
git clone https://github.com/Ashirogi-Muto/OvaryVision.git
cd OvaryVision

# 2. Install dependencies
pip install -e .

# 3. Run self-contained CPU Quick-Start Demo
python3 scripts/run_quickstart_demo.py
```

### Expected Output

```text
==============================================================================
      OVARYVISION: CPU QUICK-START RESEARCH PROTOTYPE DEMO
==============================================================================
Demonstrating 3-Arm Comparative Inpainting vs Masking Benchmark on CPU...

[1/3] Synthetic Ultrasound Scans Generated:
      Resolution        : 512x384 px
      Artifact Footprint: 227 px (0.115% of frame)

[2/3] Comparative Tri-Partition Image Integrity Evaluation:
------------------------------------------------------------------------------
Evaluation Metric                Arm A (Raw)     Arm B (Inpainted) Arm C (Zero-Mask)
------------------------------------------------------------------------------
Exterior MAE (Outside Mask)      0.000000        0.000000         0.000000       
Exterior PSNR (Conservation)     inf             inf              inf            
In-Mask Reconstruction MAE       28.60           10.52            116.19         
Structural Similarity (SSIM)     0.9939          0.9973           0.9904         
In-Mask Texture Variance (ΔLTV)  2584.1          215.1            460.7          
------------------------------------------------------------------------------

[3/3] Key Research Observations:
  • Strict Exterior Conservation: Both Arm B and Arm C preserve 100% of exterior parenchyma (MAE = 0.000000).
  • Speckle Preservation: Harmonic inpainting (Arm B) reconstructs smooth speckle (MAE=10.5),
    whereas Zero-masking (Arm C) creates high-contrast black steps (ΔLTV=460.7).
  • Deep CAD Impact: In real MMOTU evaluations, Arm C's artificial black edges systematically elevated
    predicted malignancy probability on benign scans (+0.005 to +0.033) and degraded Brier calibration.

[✓] Demo executed successfully on CPU in <2 seconds without external dependencies.
==============================================================================
```

---

## 3. Running Automated Tests

The test suite validates configuration, data auditing, two-tier leakage detection, synthetic artifact generation, harmonic inpainting, temperature calibration, and statistical non-inferiority testing:

```bash
pytest -v tests/
```

All **69 tests pass** in under 2 seconds.

---

## 4. Current Research Prototype Findings

### 4.1 Arm A Baseline Benchmark (Full MMOTU Test Set, $N=220$)

Trained on NVIDIA RTX PRO 6000 Blackwell Server Edition hardware:
- **EfficientNet-B3**: Test AUROC = **$0.8931 \pm 0.0443$** | Calibrated ECE = **$0.1681 \pm 0.0189$** | Calibrated Brier = **$0.0816 \pm 0.0106$** ($T^* = 0.4500$).
- **ConvNeXt-Small**: Test AUROC = **$0.8803 \pm 0.0685$** | Calibrated ECE = **$0.2993 \pm 0.2432$** | Calibrated Brier = **$0.1631 \pm 0.1188$** ($T^* = 2.6133$).

### 4.2 Gate B Real-Image Pilot Study ($N=140$ QC-Passed Scans)

Evaluated across Arms A, B, and C with 2-pixel boundary dilation:
- **Exterior Conservation**: $MAE_{ext} = 0.000000$ and $PSNR_{ext} = \infty$ across 100% of images.
- **Texture Variance**: Arm B In-Mask $\Delta \text{LTV} = 1,311.4 \pm 958.5$; Arm C In-Mask $\Delta \text{LTV} = 3,821.2 \pm 2,092.2$ (2.91-fold higher speckle variance deviation under zero-masking).
- **Probability Drift on Held-Out Test Scans**: Zero-masking (Arm C) systematically elevated predicted malignancy on benign scans ($\Delta p \in [+0.0017, +0.0328]$, $95\%$ bootstrap CIs strictly positive), while Harmonic Inpainting (Arm B) maintained near-zero drift ($\Delta p \in [-0.0021, +0.0019]$).
- **Calibration Impact**: Calibrated Brier score was preserved under Arm B ($0.0004$–$0.0042$) and degraded under Arm C ($0.0006$–$0.0093$).
- **Specificity**: Remained **100.0%** at the 0.5 decision threshold across all arms.

See [`docs/GATE_B_PILOT_RESULTS.md`](docs/GATE_B_PILOT_RESULTS.md) for complete tabular metrics and confidence intervals.

---

## 5. Dataset Instructions & Access Conditions

Medical ultrasound datasets are subject to clinical ethics agreements and are **never committed to this repository**:

### 1. MMOTU (Multi-Modality Ovarian Tumor Ultrasound Benchmark)
- **Source**: Zhao et al., *MMOTU DS2Net* ([Official GitHub](https://github.com/cv516Buaa/MMOTU_DS2Net) / [Zenodo Record 21128657](https://zenodo.org/records/21128657)).
- **Content**: 1,469 2D transvaginal ultrasound images with binary high-grade serous carcinoma vs. benign labels and tumor segmentation masks.
- **Placement**: Place downloaded images in `data/raw/mmotu/images/` (`{id}.JPG`) and tumor masks in `data/raw/mmotu/tumor_masks/` (`{id}.PNG`).
- **Audit**: Run `python3 -m ovaryvision.cli audit-data --data-dir data/raw/mmotu/images/` to verify checksums and perceptual cluster assignments.

### 2. OvaTUS-V1 (Independent External Generalization Benchmark)
- **Source**: SigM Lab, Hanoi University of Science and Technology ([Official Portal](https://sigmlab.com/datasets/Ovarian/)).
- **Access Status**: Formal data request drafted ([`docs/OVATUS_ACCESS_REQUEST.md`](docs/OVATUS_ACCESS_REQUEST.md)) requesting patient-matched post-operative surgical histopathology. External binary evaluation is blocked until histology labels are confirmed.

---

## 6. GPU Training Workflow

To reproduce model training on an external GPU environment (Google Colab / Kaggle / Cloud CUDA):

```bash
# 1. Install GPU dependencies
pip install torch torchvision timm --index-url https://download.pytorch.org/whl/cu121

# 2. Train baseline Arm A model
python3 scripts/gpu_train.py \
    --manifest data/manifests/mmotu_split_seed42.csv \
    --data-dir data/raw/mmotu/images \
    --arm arm_a_raw \
    --backbone convnext_small \
    --seed 42 \
    --batch-size 32 \
    --epochs 25 \
    --output-dir outputs/benchmark_runs
```

---

## 7. Project Structure

```text
OvaryVision/
├── SPEC.md                    # Canonical research & technical specification
├── README.md                  # Project overview, quick-start & reproduction
├── pyproject.toml             # PEP 621 packaging metadata
├── configs/
│   ├── default.yaml           # Default runtime parameters
│   └── experiment.yaml        # 3-arm benchmark configuration
├── docs/
│   ├── PROJECT_STATUS.md      # Prototype freeze status, findings & roadmap
│   ├── RESEARCH_PROTOCOL.md   # Experimental design & metric formulations
│   ├── GATE_B_PILOT_RESULTS.md# Real-image pilot benchmark findings
│   ├── ARM_A_RESULTS_AUDIT.md # Baseline evaluation forensic audit
│   ├── OVATUS_ACCESS_REQUEST.md # External data request draft (not sent)
│   ├── EXPERIMENT_TRACKER.md  # Continuous experiment registry
│   └── DATA_DICTIONARY.md     # Manifest schemas & field definitions
├── src/ovaryvision/
│   ├── annotation_workflow.py # Caliper extraction & QC filter engine
│   ├── calibration.py         # Validation temperature scaling & ECE/Brier
│   ├── cli.py                 # Click CLI commands
│   ├── config.py              # Pydantic v2 typed configuration
│   ├── data_audit.py          # Checksums & dHash near-duplicate clustering
│   ├── manifests.py           # Two-tier leakage validation gate
│   ├── preprocessing.py       # 3-arm inpainting & zero-masking engine
│   ├── splits.py              # Group-stratified train/val/test splitter
│   ├── statistics.py          # Paired DeLong test, bootstrap CIs & non-inferiority
│   └── synthetic_artifacts.py # Deterministic synthetic artifact engine
├── scripts/
│   ├── run_quickstart_demo.py # 2-second CPU quick-start demo
│   ├── evaluate_synthetic_benchmark.py # Tri-partition evaluation
│   ├── generate_figures.py    # Publication vector SVG/PNG figures
│   ├── generate_tables.py     # LaTeX manuscript table generator
│   └── gpu_train.py           # PyTorch GPU benchmark training
├── outputs/
│   ├── figures/               # Publication figures & LaTeX tables
│   └── metrics/               # Summary CSV/JSON benchmark results
└── tests/                     # 69 automated unit & integration tests
```

---

## 8. Citation & Ethics

```bibtex
@article{ovaryvision2026,
  title={Diagnostic Safety and Calibration Impact of Clinician Annotation Masking versus Inpainting in Ovarian Ultrasound CAD},
  author={OvaryVision Research Team},
  year={2026}
}
```

This research utilizes de-identified retrospective ultrasound data for academic, non-commercial purposes in compliance with medical data governance principles.
