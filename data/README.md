# Data Directory — OvaryVision

This directory stores all raw, interim, and processed ultrasound imaging data and manifests.

## Structure

- `raw/`: Untouched datasets (e.g. `data/raw/mmotu/`, `data/raw/ovatus_v1/`). Excluded from Git.
- `interim/`: Intermediate artifacts such as extracted annotation masks and near-duplicate cluster lists. Excluded from Git.
- `processed/`: Preprocessed, normalized images organized by arm (`arm_a_raw/`, `arm_b_inpainted/`, `arm_c_masked/`). Excluded from Git.
- `manifests/`: Machine-readable, versioned CSV manifests linking image IDs, paths, hashes, labels, and split assignments. Tracked in Git.

*Note: Raw medical data is never committed to version control.*

