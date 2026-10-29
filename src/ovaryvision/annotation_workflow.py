"""Annotation workflow and QC engine for real-image clinician artifact masks (Gate B).

Provides reproducible extraction, validation, and metadata tracking for clinician measurement
calipers, crosshairs, dashed lines, and annotation text on ovarian ultrasound scans.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from scipy.ndimage import binary_opening, label

from ovaryvision.logging_utils import get_logger

logger = get_logger("annotation_workflow")


class ArtifactQCMetrics(BaseModel):
    """Quality control metrics for an annotated artifact mask."""
    image_id: str
    has_artifact: bool
    pixel_count: int
    mask_ratio: float
    tumor_overlap_ratio: float
    detected_categories: List[str] = Field(default_factory=list)
    qc_status: Literal["PASSED", "FLAGGED_EMPTY", "FLAGGED_EXCESSIVE_AREA", "FLAGGED_TUMOR_OVERLAP"]
    review_status: Literal["researcher_annotated", "pending_clinician_review", "rejected_qc"]
    qc_notes: str = ""


class ArtifactAnnotationManifestRecord(BaseModel):
    """Schema for records in the real artifact manifest."""
    image_id: str
    relative_image_path: str
    artifact_mask_path: Optional[str]
    has_artifact: bool
    artifact_categories: str
    pixel_count: int
    mask_ratio: float
    tumor_overlap_ratio: float
    qc_status: str
    review_status: str
    annotator_provenance: str


def extract_caliper_candidate_mask(
    image_array: np.ndarray,
    intensity_threshold: int = 240,
    min_line_length: int = 12,
) -> Tuple[np.ndarray, List[str]]:
    """Detect orthogonal measurement crosshair lines and caliper text candidates.

    Args:
        image_array: Grayscale 2D image array of shape (H, W).
        intensity_threshold: Minimum brightness for clinician overlays (default 240).
        min_line_length: Structuring element dimension for horizontal/vertical lines (default 12).

    Returns:
        Tuple of (binary_mask, detected_categories).
    """
    if image_array.ndim == 3:
        gray = np.mean(image_array, axis=2).astype(np.uint8)
    else:
        gray = image_array.astype(np.uint8)

    # 1. High-intensity thresholding
    bright_pixels = gray >= intensity_threshold

    # 2. Orthogonal line kernels (caliper crosshairs and measurement bars)
    h_kernel = np.ones((1, min_line_length), dtype=bool)
    v_kernel = np.ones((min_line_length, 1), dtype=bool)

    h_lines = binary_opening(bright_pixels, structure=h_kernel)
    v_lines = binary_opening(bright_pixels, structure=v_kernel)
    caliper_lines = h_lines | v_lines

    # 3. Small clustered characters / text near crosshairs (connected components)
    labeled, num_features = label(bright_pixels)
    text_mask = np.zeros_like(bright_pixels)

    categories = []
    if np.sum(caliper_lines) > 0:
        categories.append("caliper_crosshair")

    # Filter connected components: caliper ticks and characters (typically 15 to 400 pixels)
    for feat_id in range(1, num_features + 1):
        feat_mask = labeled == feat_id
        feat_size = int(np.sum(feat_mask))
        if 15 <= feat_size <= 450:
            # Check aspect ratio or proximity to lines
            text_mask |= feat_mask

    if np.sum(text_mask) > 0:
        categories.append("annotation_text")

    combined_mask = (caliper_lines | text_mask).astype(np.uint8)
    return combined_mask, categories


def evaluate_artifact_qc(
    image_id: str,
    artifact_mask: np.ndarray,
    tumor_mask: Optional[np.ndarray] = None,
    detected_categories: Optional[List[str]] = None,
    max_area_ratio: float = 0.05,
    max_tumor_overlap: float = 0.50,
) -> ArtifactQCMetrics:
    """Evaluate quality control constraints on a candidate artifact mask.

    Flags:
    - Empty masks (pixel_count == 0)
    - Excessive area (mask_ratio > max_area_ratio)
    - Potential tumor overlap (tumor_overlap_ratio > max_tumor_overlap)
    """
    h, w = artifact_mask.shape[:2]
    total_pixels = h * w
    pixel_count = int(np.sum(artifact_mask > 0))
    mask_ratio = float(pixel_count / total_pixels) if total_pixels > 0 else 0.0

    tumor_overlap_ratio = 0.0
    if tumor_mask is not None and pixel_count > 0:
        overlap = np.sum((artifact_mask > 0) & (tumor_mask > 0))
        tumor_overlap_ratio = float(overlap / pixel_count)

    categories = detected_categories or []

    # QC Rule Evaluation
    if pixel_count == 0:
        status = "FLAGGED_EMPTY"
        notes = "No artifact pixels identified in candidate region."
        rev_status = "rejected_qc"
    elif mask_ratio > max_area_ratio:
        status = "FLAGGED_EXCESSIVE_AREA"
        notes = f"Mask area ({mask_ratio*100:.2f}%) exceeds safety ceiling ({max_area_ratio*100:.2f}%)."
        rev_status = "rejected_qc"
    elif tumor_overlap_ratio > max_tumor_overlap:
        status = "FLAGGED_TUMOR_OVERLAP"
        notes = f"Significant overlap with tumor ROI ({tumor_overlap_ratio*100:.1f}% > {max_tumor_overlap*100:.1f}%). Potential lesion corruption."
        rev_status = "pending_clinician_review"
    else:
        status = "PASSED"
        notes = "Within normal caliper geometric parameters."
        rev_status = "researcher_annotated"

    return ArtifactQCMetrics(
        image_id=image_id,
        has_artifact=(pixel_count > 0),
        pixel_count=pixel_count,
        mask_ratio=mask_ratio,
        tumor_overlap_ratio=tumor_overlap_ratio,
        detected_categories=categories,
        qc_status=status,
        review_status=rev_status,
        qc_notes=notes,
    )


def generate_qc_overlay_image(
    raw_image: np.ndarray,
    artifact_mask: np.ndarray,
    tumor_mask: Optional[np.ndarray] = None,
    dilated_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """Generate a 4-panel visual QC comparison image.

    Panels:
    1. Raw Ultrasound Image
    2. Candidate Artifact Mask Overlay (Cyan)
    3. Dilated Artifact Mask Overlay (Red / 2px boundary)
    4. Tumor ROI Boundary Overlay (Yellow)
    """
    h, w = raw_image.shape[:2]
    if raw_image.ndim == 2:
        base_rgb = np.stack([raw_image] * 3, axis=-1)
    else:
        base_rgb = raw_image.copy()

    # Panel 1: Raw
    p1 = Image.fromarray(base_rgb)

    # Panel 2: Artifact Overlay (Cyan: 0, 255, 255)
    ov2 = base_rgb.copy()
    art_idx = artifact_mask > 0
    ov2[art_idx] = [0, 255, 255]
    p2 = Image.fromarray(ov2)

    # Panel 3: Dilated Overlay (Red: 255, 0, 0)
    ov3 = base_rgb.copy()
    if dilated_mask is not None:
        dil_idx = dilated_mask > 0
        ov3[dil_idx] = [255, 60, 60]
    p3 = Image.fromarray(ov3)

    # Panel 4: Tumor Boundary Overlay (Yellow: 255, 255, 0)
    ov4 = base_rgb.copy()
    if tumor_mask is not None:
        tum_idx = tumor_mask > 0
        ov4[tum_idx] = [255, 255, 0]
    p4 = Image.fromarray(ov4)

    # Assemble horizontally
    total_w = w * 4
    composite = Image.new("RGB", (total_w, h + 30), color=(30, 30, 30))
    composite.paste(p1, (0, 30))
    composite.paste(p2, (w, 30))
    composite.paste(p3, (w * 2, 30))
    composite.paste(p4, (w * 3, 30))

    draw = ImageDraw.Draw(composite)
    draw.text((10, 8), "Panel 1: Raw Scan", fill=(220, 220, 220))
    draw.text((w + 10, 8), "Panel 2: Artifact (Cyan)", fill=(0, 255, 255))
    draw.text((w * 2 + 10, 8), "Panel 3: Dilated (Red)", fill=(255, 100, 100))
    draw.text((w * 3 + 10, 8), "Panel 4: Tumor ROI (Yellow)", fill=(255, 255, 0))

    return composite

