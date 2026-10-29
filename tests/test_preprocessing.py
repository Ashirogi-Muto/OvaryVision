"""Tests for the three preprocessing arms: Arm A (Raw), Arm B (Inpainted), Arm C (Masked)."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from ovaryvision.preprocessing import (
    HarmonicDiffusionInpainter,
    apply_arm_a,
    apply_arm_b,
    apply_arm_c,
    dilate_binary_mask,
    process_single_image,
)


@pytest.fixture
def sample_image_and_mask() -> tuple[np.ndarray, np.ndarray]:
    """Create a 100x100 synthetic ultrasound patch with a central cross caliper mask."""
    # Textured background (mean 120, std 20)
    rng = np.random.default_rng(42)
    img = np.clip(rng.normal(120, 20, (100, 100)), 0, 255).astype(np.uint8)

    # 10x10 cross in the center
    mask = np.zeros((100, 100), dtype=bool)
    mask[45:55, 50] = True
    mask[50, 45:55] = True

    # Burn white caliper onto image
    img[mask] = 255
    return img, mask


def test_dilate_binary_mask():
    """Verify mask dilation behavior."""
    single_pixel_mask = np.zeros((10, 10), dtype=bool)
    single_pixel_mask[5, 5] = True

    # Radius 0: identity
    r0 = dilate_binary_mask(single_pixel_mask, radius=0)
    assert np.sum(r0) == 1
    assert r0[5, 5]

    # Radius 1: cross (5 pixels total)
    r1 = dilate_binary_mask(single_pixel_mask, radius=1)
    assert np.sum(r1) == 5
    assert r1[5, 5] and r1[4, 5] and r1[6, 5] and r1[5, 4] and r1[5, 6]

    # Radius 2: diamond cross
    r2 = dilate_binary_mask(single_pixel_mask, radius=2)
    assert np.sum(r2) > 5


def test_arm_a_raw_identity(sample_image_and_mask):
    """Verify Arm A returns an exact duplicate without modifying original array."""
    img, _ = sample_image_and_mask
    original_copy = img.copy()
    raw = apply_arm_a(img)

    assert np.array_equal(raw, original_copy)
    raw[0, 0] = 0  # Modify result
    assert img[0, 0] == original_copy[0, 0]  # Original untouched


def test_arm_b_inpainting_strict_exterior_conservation(sample_image_and_mask):
    """Verify Arm B reconstructs mask interior while maintaining 100% exterior pixel invariance."""
    img, mask = sample_image_and_mask
    dilated_mask = dilate_binary_mask(mask, radius=2)

    inpainted = apply_arm_b(img, mask, inpainter=HarmonicDiffusionInpainter(), dilation_radius=2)

    # 1. Interior pixels must be changed from the bright caliper (255)
    assert not np.array_equal(inpainted[dilated_mask], img[dilated_mask])
    # The caliper was 255; the inpainted region should have blended closer to background mean ~120
    assert np.mean(inpainted[dilated_mask]) < 240

    # 2. Strict exterior invariance: all pixels outside dilated mask MUST be identical
    assert np.array_equal(inpainted[~dilated_mask], img[~dilated_mask])


def test_arm_c_non_generative_masking(sample_image_and_mask):
    """Verify Arm C sets mask interior to constant zero and preserves exterior unchanged."""
    img, mask = sample_image_and_mask
    dilated_mask = dilate_binary_mask(mask, radius=2)

    masked = apply_arm_c(img, mask, fill_value=0, dilation_radius=2)

    # 1. All interior pixels must be exactly zero
    assert np.all(masked[dilated_mask] == 0)

    # 2. Exterior pixels must be completely identical
    assert np.array_equal(masked[~dilated_mask], img[~dilated_mask])


def test_empty_or_none_mask_handling(sample_image_and_mask):
    """Verify that None or empty mask yields clean pass-through for all arms."""
    img, _ = sample_image_and_mask
    empty_mask = np.zeros_like(img, dtype=bool)

    assert np.array_equal(apply_arm_b(img, None), img)
    assert np.array_equal(apply_arm_b(img, empty_mask), img)
    assert np.array_equal(apply_arm_c(img, None), img)
    assert np.array_equal(apply_arm_c(img, empty_mask), img)


def test_process_single_image_file_roundtrip(tmp_path: Path, sample_image_and_mask):
    """Verify file-based processing for all three arms."""
    img_arr, mask_arr = sample_image_and_mask

    img_file = tmp_path / "test_scan.png"
    mask_file = tmp_path / "test_mask.png"

    Image.fromarray(img_arr).save(img_file)
    Image.fromarray((mask_arr * 255).astype(np.uint8)).save(mask_file)

    out_a = tmp_path / "processed_a.png"
    out_b = tmp_path / "processed_b.png"
    out_c = tmp_path / "processed_c.png"

    process_single_image(img_file, mask_file, arm="arm_a_raw", output_path=out_a)
    process_single_image(img_file, mask_file, arm="arm_b_inpainted", output_path=out_b)
    process_single_image(img_file, mask_file, arm="arm_c_masked", output_path=out_c)

    assert out_a.is_file()
    assert out_b.is_file()
    assert out_c.is_file()

    # Load and check values
    arr_c = np.array(Image.open(out_c))
    dilated = dilate_binary_mask(mask_arr, radius=2)
    assert np.all(arr_c[dilated] == 0)

