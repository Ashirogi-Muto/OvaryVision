# architecture.md — System Architecture & Component Design: OvaryVision

This document describes the software architecture, data pipelines, module interactions, and compute boundaries of the OvaryVision benchmark system.

---

## 1. High-Level System Architecture

The codebase is structured around a decoupled, data-centric research workflow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LOCAL OCI CPU ORCHESTRATOR                         │
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │  config.py   │────▶│environment.py│────▶│logging_utils.│                 │
│  └──────┬───────┘     └──────────────┘     └──────┬───────┘                 │
│         │                                         │                         │
│         ▼                                         ▼                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │ data_audit.py│────▶│ manifests.py │────▶│provenance.py │                 │
│  └──────────────┘     └──────┬───────┘     └──────────────┘                 │
│                              │                                              │
│                              │ Generate Splitted Manifests                  │
│                              ▼                                              │
│                   data/manifests/split_seed*.csv                            │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ Artifact & Config Sync
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL GPU EXECUTION (COLAB / KAGGLE)               │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    3-Arm Preprocessing Engine                         │  │
│  │  Arm A: Raw          Arm B: MGGAN Inpainting     Arm C: Zero-Masking  │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     PyTorch Training Engine                           │  │
│  │  Backbones: ConvNeXt-Tiny, EfficientNet-B0                            │  │
│  │  Loss: Class-Weighted Focal Loss (gamma=2.0)                          │  │
│  │  Optimizer: AdamW + Cosine Annealing LR                               │  │
│  │  Model Selection: Best Val Macro-F1                                   │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │               Validation Temperature Scaling (T*)                     │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                   Evaluation & Prediction Serialization               │  │
│  │  Predictions CSV, Logits, Calibrated Probs                            │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │ Sync Results Back to OCI
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LOCAL OCI CPU ORCHESTRATOR                         │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                   Statistical Analysis & Reporting                    │  │
│  │  - Paired DeLong's ROC Test                                           │  │
│  │  - McNemar's Concordance Test                                         │  │
│  │  - Stratified 1,000-Bootstrap 95% CIs for Delta AUC (Arm C - Arm B)   │  │
│  │  - Tri-Partition Texture Auditing (LTV, Exterior PSNR)                │  │
│  │  - SVG Figures and LaTeX Summary Tables                               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Directory Layout

The repository enforces standard Python packaging under `src/ovaryvision/`:

```text
/home/ubuntu/major/
├── configs/
│   ├── default.yaml            # Base default configuration
│   └── experiment.yaml         # Complete 3-arm benchmark experiment settings
├── data/
│   ├── raw/                    # Untouched external & raw downloads (gitignored)
│   ├── interim/                # Extracted masks & near-duplicate clusters
│   ├── processed/              # Normalized, arm-preprocessed cached images
│   └── manifests/              # Versioned CSV manifests (git-tracked)
├── docs/
│   ├── RESEARCH_PROTOCOL.md    # Experimental design, arms, calibration
│   ├── DATA_DICTIONARY.md      # Manifest & prediction schemas
│   ├── EXPERIMENT_TRACKER.md   # Durable run registry
│   ├── architecture.md         # Component diagrams & data flows
│   └── data_protocol.md        # Ingestion, mask contracts, privacy rules
├── logs/                       # Rotating application logs (10MB x 5 rotation)
├── outputs/
│   ├── checkpoints/            # Model weight checkpoints (gitignored)
│   ├── figures/                # Publication SVG/PNG plots
│   ├── metrics/                # Prediction CSVs & evaluation JSONs
│   └── logs/                   # Experiment training logs
├── scripts/
│   └── run_audit.py            # CLI helper runner for data audit
├── src/
│   └── ovaryvision/
│       ├── __init__.py         # Package initialization & version
│       ├── cli.py              # Click command-line interface entry point
│       ├── config.py           # Pydantic configuration schemas & YAML parser
│       ├── logging_utils.py    # Rotating file & stream logger
│       ├── environment.py      # System diagnostic scanner & hardware checks
│       ├── manifests.py        # Typed manifests & CSV/JSON IO
│       ├── provenance.py       # Deterministic execution provenance tracker
│       └── data_audit.py       # Image inventory & near-duplicate auditor
└── tests/
    ├── test_config.py          # Config loading & validation tests
    ├── test_environment.py     # Hardware diagnostic tests
    ├── test_manifests.py       # Manifest schema & validation tests
    ├── test_provenance.py      # Execution hashing tests
    ├── test_data_audit.py      # Image inventory & duplicate detection tests
    └── test_cli.py             # CLI command end-to-end tests
```

---

## 3. Data Flow & Provenance Tracking

1. **Configuration Loading**: `config.py` loads YAML configurations, validates types, and sets defaults using Pydantic v2.
2. **Environment Diagnostics**: `environment.py` verifies available compute resources, CPU architectures, disk capacities, and warns if attempting to train without CUDA.
3. **Data Ingestion & Audit**: `data_audit.py` scans raw image directories, computes SHA-256 digests, calculates perceptual difference hashes (dHash), and outputs a structured audit report.
4. **Manifest Creation**: `manifests.py` generates verifiable CSV manifests linking image paths, labels, and separate artifact/tumor masks.
5. **Execution Provenance**: `provenance.py` captures configuration hashes, environment versions, and timestamps into deterministic experiment receipts.

