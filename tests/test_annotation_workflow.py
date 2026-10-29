"""Unit tests for the clinician artifact annotation workflow and QC engine (Gate B)."""

import numpy as np
import pytest
from PIL import Image

from ovaryvision.annotation_workflow import (
    extract_caliper_candidate_mask,
    evaluate_artifact_qc,
    generate_qc_overlay_image,
)


def test_extract_caliper_candidate_mask_on_crosshair() -> None:
    """Verify that orthogonal crosshairs are correctly extracted as caliper artifacts."""
    img = np.zeros((100, 100), dtype=np.uint8)
    # Draw a 20px horizontal line and 20px vertical line at intensity 255
    img[50, 40:60] = 255
    img[40:60, 50] = 255

    mask, categories = extract_caliper_candidate_mask(img, intensity_threshold=240, min_line_length=12)
    assert mask.shape == (100, 100)
    assert np.sum(mask) > 0
    assert "caliper_crosshair" in categories


def test_evaluate_artifact_qc_empty_mask() -> None:
    """Verify that completely empty artifact masks are flagged as FLAGGED_EMPTY."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    qc = evaluate_artifact_qc("IMG_001", mask)
    assert qc.qc_status == "FLAGGED_EMPTY"
    assert qc.review_status == "rejected_qc"
    assert qc.has_artifact is False


def test_evaluate_artifact_qc_excessive_area() -> None:
    """Verify that masks covering more than 5% of the frame are flagged."""
    mask = np.ones((100, 100), dtype=np.uint8)  # 100% of frame
    qc = evaluate_artifact_qc("IMG_002", mask, max_area_ratio=0.05)
    assert qc.qc_status == "FLAGGED_EXCESSIVE_AREA"
    assert qc.review_status == "rejected_qc"


def test_evaluate_artifact_qc_tumor_overlap() -> None:
    """Verify that masks heavily overlapping the tumor ROI are flagged for clinician review."""
    artifact_mask = np.zeros((100, 100), dtype=np.uint8)
    artifact_mask[45:55, 45:55] = 1  # 100 pixels

    tumor_mask = np.zeros((100, 100), dtype=np.uint8)
    tumor_mask[40:60, 40:60] = 1  # completely encloses artifact

    qc = evaluate_artifact_qc(
        "IMG_003",
        artifact_mask,
        tumor_mask=tumor_mask,
        max_tumor_overlap=0.50,
    )
    assert qc.qc_status == "FLAGGED_TUMOR_OVERLAP"
    assert qc.review_status == "pending_clinician_review"
    assert qc.tumor_overlap_ratio == 1.0


def test_generate_qc_overlay_image() -> None:
    """Verify generation of the 4-panel visual QC comparison image."""
    raw = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 25] = 1

    composite = generate_qc_overlay_image(raw, mask)
    assert isinstance(composite, Image.Image)
    assert composite.size == (200, 80)  # 4 * 50 = 200 width, 50 + 30 = 80 height

