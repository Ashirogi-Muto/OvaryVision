"""Automated tests for controlled synthetic-artifact injection and integrity metrics.

NOTE: All synthetic images and fixtures in this file are software verification tests,
NOT clinical evidence or experimental benchmark results.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from ovaryvision.synthetic_artifacts import (
    SyntheticArtifactConfig,
    inject_synthetic_artifacts,
    export_synthetic_case,
    compute_mae,
    compute_psnr,
    compute_ssim,
    compute_local_texture_variance,
    compute_lpips,
    evaluate_reconstruction_integrity,
)


@pytest.fixture
def clean_synthetic_ultrasound():
    """Create a software test fixture simulating a 2D grayscale ultrasound patch.

    This is purely a software unit test fixture, not real clinical patient data.
    """
    rng = np.random.default_rng(12345)
    # Background speckle: Rayleigh-like distribution typical of acoustic speckle
    speckle = rng.rayleigh(scale=30.0, size=(128, 128))
    # Simulated hypoechoic cyst region in center
    yy, xx = np.mgrid[:128, :128]
    cyst_mask = ((xx - 64) ** 2 + (yy - 64) ** 2) < 25 ** 2
    speckle[cyst_mask] = rng.rayleigh(scale=8.0, size=cyst_mask.sum())
    patch = np.clip(speckle, 0, 255).astype(np.uint8)
    return patch


def test_deterministic_output_fixed_seed(clean_synthetic_ultrasound):
    """Verify that same random seed produces bit-identical artifacts and masks."""
    cfg1 = SyntheticArtifactConfig(seed=42)
    cfg2 = SyntheticArtifactConfig(seed=42)
    cfg3 = SyntheticArtifactConfig(seed=999)

    res1 = inject_synthetic_artifacts(clean_synthetic_ultrasound, cfg1)
    res2 = inject_synthetic_artifacts(clean_synthetic_ultrasound, cfg2)
    res3 = inject_synthetic_artifacts(clean_synthetic_ultrasound, cfg3)

    # Identical seed produces identical arrays
    np.testing.assert_array_equal(res1.corrupted_image, res2.corrupted_image)
    np.testing.assert_array_equal(res1.artifact_mask, res2.artifact_mask)
    assert res1.provenance["clean_sha256"] == res2.provenance["clean_sha256"]
    assert res1.provenance["corrupted_sha256"] == res2.provenance["corrupted_sha256"]

    # Different seed produces different coordinates and checksum
    assert res1.provenance["corrupted_sha256"] != res3.provenance["corrupted_sha256"]


def test_artifact_containment(clean_synthetic_ultrasound):
    """Verify the invariant: every pixel modified by artifact injection is inside the mask."""
    cfg = SyntheticArtifactConfig(seed=42, opacity=1.0)
    res = inject_synthetic_artifacts(clean_synthetic_ultrasound, cfg)

    clean = res.clean_reference
    corrupted = res.corrupted_image
    mask = res.artifact_mask

    diff = (clean != corrupted)
    # Every modified pixel MUST have mask == 1
    assert np.all(mask[diff] == 1), "Found modified pixels outside the artifact mask!"
    assert np.sum(diff) > 0, "No pixels were modified by artifact injection."


def test_source_array_immutability(clean_synthetic_ultrasound):
    """Verify that the source image array is not mutated in-place."""
    orig_copy = clean_synthetic_ultrasound.copy()
    cfg = SyntheticArtifactConfig(seed=42)
    res = inject_synthetic_artifacts(clean_synthetic_ultrasound, cfg)

    # Original array passed in remains identical
    np.testing.assert_array_equal(clean_synthetic_ultrasound, orig_copy)
    # Reference in result matches original
    np.testing.assert_array_equal(res.clean_reference, orig_copy)


def test_output_dimensions_match_2d_and_3d():
    """Verify 2D and 3D image arrays maintain matching dimensions and expected mask shapes."""
    # 2D Grayscale
    img_2d = np.full((80, 100), 100, dtype=np.uint8)
    res_2d = inject_synthetic_artifacts(img_2d, SyntheticArtifactConfig(seed=1))
    assert res_2d.corrupted_image.shape == (80, 100)
    assert res_2d.corrupted_image.dtype == np.uint8
    assert res_2d.artifact_mask.shape == (80, 100)
    assert res_2d.artifact_mask.dtype == np.uint8

    # 3D RGB
    img_3d = np.full((90, 110, 3), 120, dtype=np.uint8)
    res_3d = inject_synthetic_artifacts(img_3d, SyntheticArtifactConfig(seed=2))
    assert res_3d.corrupted_image.shape == (90, 110, 3)
    assert res_3d.corrupted_image.dtype == np.uint8
    # Mask is always 2D (H, W)
    assert res_3d.artifact_mask.shape == (90, 110)
    assert res_3d.artifact_mask.dtype == np.uint8


def test_invalid_inputs():
    """Verify that invalid inputs raise ValueError with clear error messages."""
    # Non-numpy
    with pytest.raises(ValueError, match="must be a numpy ndarray"):
        inject_synthetic_artifacts("not_an_array")

    # Non-uint8
    float_img = np.zeros((50, 50), dtype=np.float32)
    with pytest.raises(ValueError, match="dtype must be uint8"):
        inject_synthetic_artifacts(float_img)

    # Dimensions too small
    tiny_img = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        inject_synthetic_artifacts(tiny_img)

    # 1D or 4D
    with pytest.raises(ValueError, match="must be 2D"):
        inject_synthetic_artifacts(np.zeros((100,), dtype=np.uint8))


def test_export_synthetic_case(clean_synthetic_ultrasound, tmp_path: Path):
    """Verify export_synthetic_case saves clean, corrupted, mask, and metadata files."""
    cfg = SyntheticArtifactConfig(seed=42)
    res = inject_synthetic_artifacts(clean_synthetic_ultrasound, cfg)

    out_dir = tmp_path / "synthetic_export"
    paths = export_synthetic_case(res, out_dir, case_id="CASE_TEST_001")

    assert paths["clean"].is_file()
    assert paths["corrupted"].is_file()
    assert paths["mask"].is_file()
    assert paths["metadata"].is_file()

    with open(paths["metadata"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["case_id"] == "CASE_TEST_001"
    assert "mask_pixel_count" in meta["provenance"]
    assert meta["provenance"]["mask_pixel_count"] > 0


def test_integrity_metrics_identical_images():
    """Verify metrics on identical reference and reconstruction yield perfect scores."""
    arr = np.full((64, 64), 100, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:30, 20:30] = 1

    report = evaluate_reconstruction_integrity(arr, arr, mask)
    assert report.global_mae == 0.0
    assert report.global_psnr == float("inf")
    assert report.global_ssim == 1.0
    assert report.exterior_mae == 0.0
    assert report.exterior_psnr == float("inf")
    assert report.exterior_ssim == 1.0


def test_integrity_metrics_tri_partition():
    """Verify that modifying in-mask pixels strictly affects in-mask metrics while exterior remains perfect."""
    clean = np.full((100, 100), 100, dtype=np.uint8)
    corrupted = clean.copy()

    # Modify strictly inside a 10x10 square
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:50, 40:50] = 1
    corrupted[40:50, 40:50] = 200  # Shift intensity inside mask

    report = evaluate_reconstruction_integrity(clean, corrupted, mask)

    # In-mask metrics reflect error
    assert report.in_mask_mae == 100.0
    assert report.in_mask_psnr < 50.0
    assert report.in_mask_ssim < 1.0

    # Exterior pixel-wise metrics MUST remain perfect (zero boundary leakage)
    assert report.exterior_mae == 0.0
    assert report.exterior_psnr == float("inf")
    # Exterior SSIM is > 0.98 (near-perfect, with minor boundary window overlap within win_size//2 of mask)
    assert report.exterior_ssim > 0.98


def test_local_texture_variance():
    """Verify Local Texture Variance detects variance shifts between mask and parenchyma band."""
    img = np.zeros((100, 100), dtype=np.uint8)
    # Mask area is flat constant (var = 0)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1

    # Surrounding area has noisy speckle
    rng = np.random.default_rng(42)
    noise = rng.normal(120, 25, size=(100, 100)).astype(np.uint8)
    img = np.where(mask == 1, 0, noise)

    ltv = compute_local_texture_variance(img, mask, dilation_radius=5)
    assert ltv.var_in_mask == 0.0
    assert ltv.var_surrounding_band > 100.0
    assert ltv.delta_ltv > 100.0
    assert ltv.band_pixel_count > 0


def test_lazy_lpips():
    """Verify that compute_lpips returns None or a float without raising an exception."""
    a = np.zeros((64, 64), dtype=np.uint8)
    res = compute_lpips(a, a)
    # On CPU host without lpips, returns None
    assert res is None or isinstance(res, float)
