"""Manifest models, CSV/JSON serialization, and split leakage validation."""

import csv
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ImageRecord(BaseModel):
    """Schema for individual ultrasound image records in the central manifest."""
    image_id: str
    dataset_id: str = "MMOTU_2D"  # MMOTU_2D, MMOTU_CEUS, OvaTUS_V1
    relative_image_path: str
    sha256_checksum: str
    width: int
    height: int
    channels: int = 1
    file_size_bytes: int
    diagnostic_label: int = -1  # 0: Benign, 1: Malignant, -1: Unlabeled
    fine_grained_class: Optional[str] = None
    patient_id: Optional[str] = None
    study_id: Optional[str] = None
    pseudo_patient_cluster: Optional[str] = None
    artifact_mask_path: Optional[str] = None
    tumor_mask_path: Optional[str] = None
    has_visible_artifacts: bool = False

    @field_validator("sha256_checksum")
    @classmethod
    def validate_sha256(cls, v: str) -> str:
        clean = v.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", clean):
            raise ValueError(f"sha256_checksum must be a 64-character lowercase hex string, got '{v}'")
        return clean

    @field_validator("diagnostic_label")
    @classmethod
    def validate_label(cls, v: int) -> int:
        if v not in (-1, 0, 1):
            raise ValueError(f"diagnostic_label must be 0 (Benign), 1 (Malignant), or -1 (Unlabeled), got {v}")
        return v

    @model_validator(mode="after")
    def validate_distinct_masks(self) -> "ImageRecord":
        """Strictly prohibit confusing tumor ROI masks with annotation artifact masks."""
        art = self.artifact_mask_path
        tumor = self.tumor_mask_path
        if art and tumor and art.strip() == tumor.strip():
            raise ValueError(
                f"Severe integrity violation: artifact_mask_path and tumor_mask_path cannot point to the same file '{art}'. "
                "Tumor segmentation masks must never be conflated with clinician annotation artifact masks."
            )
        return self


class SplitRecord(BaseModel):
    """Schema for cross-validation and benchmark split assignments."""
    image_id: str
    split_role: str  # 'train', 'val', 'test_internal', 'test_external'
    split_seed: int
    pseudo_patient_cluster: Optional[str] = None
    diagnostic_label: int

    @field_validator("split_role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"train", "val", "test_internal", "test_external"}
        if v not in allowed:
            raise ValueError(f"split_role must be one of {allowed}, got '{v}'")
        return v


class PredictionRecord(BaseModel):
    """Schema for model inference and evaluation outputs."""
    experiment_id: str
    image_id: str
    preprocessing_arm: str  # arm_a_raw, arm_b_inpainted, arm_c_masked
    backbone: str
    split_seed: int
    ground_truth: int
    raw_logit_benign: float
    raw_logit_malignant: float
    calibrated_prob_malignant: float
    predicted_class: int


def load_image_manifest(manifest_path: Path | str) -> List[ImageRecord]:
    """Load image records from a CSV or JSON file."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    records: List[ImageRecord] = []
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    records.append(ImageRecord(**item))
            elif isinstance(data, dict) and "records" in data:
                for item in data["records"]:
                    records.append(ImageRecord(**item))
    else:  # CSV format
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Type conversions from string
                processed_row: Dict[str, Any] = dict(row)
                processed_row["width"] = int(row.get("width", 0))
                processed_row["height"] = int(row.get("height", 0))
                processed_row["channels"] = int(row.get("channels", 1))
                processed_row["file_size_bytes"] = int(row.get("file_size_bytes", 0))
                processed_row["diagnostic_label"] = int(row.get("diagnostic_label", -1))
                
                # Handle boolean flag
                has_art = str(row.get("has_visible_artifacts", "false")).lower()
                processed_row["has_visible_artifacts"] = has_art in ("true", "1", "yes")

                # Handle empty strings as None
                for opt_field in ["fine_grained_class", "patient_id", "study_id", "pseudo_patient_cluster", "artifact_mask_path", "tumor_mask_path"]:
                    val = processed_row.get(opt_field)
                    if val is None or (isinstance(val, str) and not val.strip()):
                        processed_row[opt_field] = None

                records.append(ImageRecord(**processed_row))

    return records


def save_image_manifest(records: List[ImageRecord], output_path: Path | str) -> None:
    """Save image records to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "dataset_id",
        "relative_image_path",
        "sha256_checksum",
        "width",
        "height",
        "channels",
        "file_size_bytes",
        "diagnostic_label",
        "fine_grained_class",
        "patient_id",
        "study_id",
        "pseudo_patient_cluster",
        "artifact_mask_path",
        "tumor_mask_path",
        "has_visible_artifacts",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.model_dump())


def load_split_manifest(manifest_path: Path | str) -> List[SplitRecord]:
    """Load train/val/test split assignments from CSV."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    records: List[SplitRecord] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                SplitRecord(
                    image_id=row["image_id"],
                    split_role=row["split_role"],
                    split_seed=int(row["split_seed"]),
                    pseudo_patient_cluster=row.get("pseudo_patient_cluster") or None,
                    diagnostic_label=int(row["diagnostic_label"]),
                )
            )
    return records


def save_split_manifest(records: List[SplitRecord], output_path: Path | str) -> None:
    """Save train/val/test split assignments to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image_id", "split_role", "split_seed", "pseudo_patient_cluster", "diagnostic_label"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.model_dump())


def validate_manifest_leakage(records: List[SplitRecord]) -> bool:
    """Verify that no image_id or pseudo_patient_cluster appears across multiple split roles.

    Enforces two strict leakage gates:
    1. Direct Image Leakage: Each image_id must appear at most once across the entire split.
    2. Cluster Leakage: No pseudo_patient_cluster may span more than one split role (e.g. train and test).

    Returns True if leak-free, raises ValueError if leakage is detected.
    """
    seen_images: Dict[str, str] = {}
    cluster_roles: Dict[str, str] = {}

    for r in records:
        # Gate 1: Check direct image-level duplication and split leakage
        if r.image_id in seen_images:
            prev_role = seen_images[r.image_id]
            if prev_role != r.split_role:
                raise ValueError(
                    f"Direct image leakage detected! Image ID '{r.image_id}' is allocated to both "
                    f"'{prev_role}' and '{r.split_role}'. Split is invalid."
                )
            else:
                raise ValueError(
                    f"Duplicate image entry detected! Image ID '{r.image_id}' appears multiple times "
                    f"in split role '{r.split_role}'."
                )
        seen_images[r.image_id] = r.split_role

        # Gate 2: Check pseudo-patient cluster leakage
        cluster = r.pseudo_patient_cluster
        if not cluster:
            continue
        role = r.split_role
        if cluster in cluster_roles and cluster_roles[cluster] != role:
            raise ValueError(
                f"Cluster data leakage detected! Pseudo-patient cluster '{cluster}' is allocated to both "
                f"'{cluster_roles[cluster]}' and '{role}'. Split is invalid."
            )
        cluster_roles[cluster] = role

    return True


def generate_manifest_template_csv(output_path: Path | str) -> None:
    """Generate an empty CSV manifest template with header row and example comments."""
    sample = [
        ImageRecord(
            image_id="SAMPLE_0001",
            dataset_id="MMOTU_2D",
            relative_image_path="data/raw/mmotu/images/sample_0001.jpg",
            sha256_checksum="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            width=800,
            height=400,
            channels=1,
            file_size_bytes=102400,
            diagnostic_label=0,
            fine_grained_class=None,
            patient_id=None,
            study_id=None,
            pseudo_patient_cluster="CLUSTER_0001",
            artifact_mask_path="data/interim/artifact_masks/sample_0001_mask.png",
            tumor_mask_path="data/raw/mmotu/tumor_masks/sample_0001_tumor.png",
            has_visible_artifacts=True,
        )
    ]
    save_image_manifest(sample, output_path)

