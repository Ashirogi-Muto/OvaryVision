"""Three-arm ultrasound image preprocessing engine: Arm A (Raw), Arm B (Inpainted), Arm C (Masked)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, generate_binary_structure

from ovaryvision.logging_utils import get_logger

logger = get_logger("preprocessing")


def dilate_binary_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Dilate binary artifact mask using a 2D cross-structuring element to avoid boundary bleeding.

    Args:
        mask: 2D binary or boolean mask array (H, W). Non-zero elements are treated as mask.
        radius: Number of morphological dilation iterations (default 2px).

    Returns:
        Dilated boolean mask array of shape (H, W).
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-dimensional (H, W), got shape {mask.shape}")
    if radius < 0:
        raise ValueError(f"Dilation radius must be non-negative, got {radius}")
    if radius == 0:
        return (mask > 0).astype(bool)

    struct = generate_binary_structure(2, 1)  # 4-connected cross
    dilated = binary_dilation(mask > 0, structure=struct, iterations=radius)
    return dilated


class BaseInpainter(ABC):
    """Abstract interface for image inpainting algorithms."""

    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint regions where mask is True.

        Args:
            image: Image array (H, W) or (H, W, C) in uint8 [0, 255] or float [0, 1].
            mask: 2D boolean mask (H, W) where True indicates pixels to inpaint.

        Returns:
            Inpainted image array with the same shape and dtype as input image.
        """
        pass


class HarmonicDiffusionInpainter(BaseInpainter):
    """Pure CPU-friendly harmonic boundary diffusion inpainter using Jacobi relaxation.

    Solves Laplace's equation \u2207\u00b2I = 0 inside the mask with Dirichlet boundary
    conditions taken from the exterior boundary. Well-suited for thin calipers, crosses,
    and annotation lines without requiring external C++/GPU dependencies.
    """

    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-3) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D (H, W), got {mask.shape}")
        if not np.any(mask):
            return image.copy()

        is_uint8 = np.issubdtype(image.dtype, np.integer)
        work_img = image.astype(np.float64)
        bool_mask = mask.astype(bool)

        if work_img.ndim == 2:
            reconstructed = self._inpaint_2d_channel(work_img, bool_mask)
        elif work_img.ndim == 3:
            reconstructed = np.zeros_like(work_img)
            for c in range(work_img.shape[2]):
                reconstructed[:, :, c] = self._inpaint_2d_channel(work_img[:, :, c], bool_mask)
        else:
            raise ValueError(f"Image must be 2D or 3D, got shape {image.shape}")

        if is_uint8:
            return np.clip(np.round(reconstructed), 0, 255).astype(image.dtype)
        return reconstructed.astype(image.dtype)

    def _inpaint_2d_channel(self, channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = channel.shape
        u = channel.copy()

        # Initialize mask region with boundary average
        boundary = dilate_binary_mask(mask, radius=1) & (~mask)
        init_val = np.mean(channel[boundary]) if np.any(boundary) else np.mean(channel)
        u[mask] = init_val

        # 4-connected Jacobi relaxation
        for _ in range(self.max_iterations):
            u_old = u.copy()
            # Neighbor shift averages
            north = np.roll(u, -1, axis=0)
            south = np.roll(u, 1, axis=0)
            west = np.roll(u, -1, axis=1)
            east = np.roll(u, 1, axis=1)

            # Recompute interior of mask
            laplace_avg = 0.25 * (north + south + west + east)
            u[mask] = laplace_avg[mask]

            max_diff = np.max(np.abs(u[mask] - u_old[mask]))
            if max_diff < self.tolerance:
                break

        return u


class OpenCVInpainter(BaseInpainter):
    """OpenCV Telea or Navier-Stokes inpainter (lazily imported if cv2 is installed)."""

    def __init__(self, method: str = "telea", inpaint_radius: int = 3) -> None:
        self.method = method.lower()
        self.inpaint_radius = inpaint_radius
        self._fallback = HarmonicDiffusionInpainter()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError:
            logger.warning("cv2 is not installed. Falling back to HarmonicDiffusionInpainter.")
            return self._fallback.inpaint(image, mask)

        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        flag = cv2.INPAINT_TELEA if self.method == "telea" else cv2.INPAINT_NS
        
        is_uint8 = np.issubdtype(image.dtype, np.integer)
        img_uint8 = image if is_uint8 else np.clip(image * 255, 0, 255).astype(np.uint8)

        inpainted = cv2.inpaint(img_uint8, mask_uint8, self.inpaint_radius, flag)

        if not is_uint8:
            return inpainted.astype(image.dtype) / 255.0
        return inpainted


class PrecomputedInpainter(BaseInpainter):
    """Inpainter using precomputed reconstruction arrays (e.g. from MGGAN or LaMa)."""

    def __init__(self, precomputed_image: np.ndarray) -> None:
        self.precomputed_image = precomputed_image

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.precomputed_image.shape != image.shape:
            raise ValueError(
                f"Precomputed image shape {self.precomputed_image.shape} does not match "
                f"input image shape {image.shape}"
            )
        return self.precomputed_image.copy()


def apply_arm_a(image: np.ndarray) -> np.ndarray:
    """Arm A (Raw): Return original image array unchanged.

    Preserves all clinician annotation artifacts as the baseline for shortcut learning.
    """
    return image.copy()


def apply_arm_b(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    inpainter: Optional[BaseInpainter] = None,
    dilation_radius: int = 2,
) -> np.ndarray:
    """Arm B (Generative Inpainting): Reconstruct dilated mask region while strictly conserving exterior tissue.

    Enforces the exterior constraint:
        I_B(u, v) = I_inpaint(u, v) if (u, v) in dilate(M_art) else I_A(u, v)

    Args:
        image: Original input image array (H, W) or (H, W, C).
        mask: Binary annotation artifact mask. If None or all-zero, returns unmodified image.
        inpainter: Inpainter implementation. Defaults to HarmonicDiffusionInpainter.
        dilation_radius: Morphological dilation radius in pixels (default 2px).

    Returns:
        Inpainted image with 100% exterior tissue conservation.
    """
    if mask is None or not np.any(mask):
        return image.copy()

    if inpainter is None:
        inpainter = HarmonicDiffusionInpainter()

    # 1. Dilate mask to encompass anti-aliased annotation edges
    dilated_mask = dilate_binary_mask(mask, radius=dilation_radius)

    # 2. Compute inpainting
    reconstruction = inpainter.inpaint(image, dilated_mask)

    # 3. Enforce strict exterior invariance
    result = image.copy()
    if image.ndim == 2:
        result[dilated_mask] = reconstruction[dilated_mask]
    else:
        for c in range(image.shape[2]):
            result[dilated_mask, c] = reconstruction[dilated_mask, c]

    return result


def apply_arm_c(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    fill_value: int = 0,
    dilation_radius: int = 2,
) -> np.ndarray:
    """Arm C (Non-Generative Masking): Occlude dilated mask region with a constant neutral value.

    Enforces the occlusion rule:
        I_C(u, v) = fill_value if (u, v) in dilate(M_art) else I_A(u, v)

    Args:
        image: Original input image array (H, W) or (H, W, C).
        mask: Binary annotation artifact mask. If None or all-zero, returns unmodified image.
        fill_value: Constant replacement value (default 0 for black pixels).
        dilation_radius: Morphological dilation radius in pixels (default 2px).

    Returns:
        Masked image with zero hallucinated synthetic textures.
    """
    if mask is None or not np.any(mask):
        return image.copy()

    dilated_mask = dilate_binary_mask(mask, radius=dilation_radius)
    result = image.copy()

    if image.ndim == 2:
        result[dilated_mask] = fill_value
    else:
        for c in range(image.shape[2]):
            result[dilated_mask, c] = fill_value

    return result


def process_single_image(
    image_path: Path | str,
    mask_path: Optional[Path | str],
    arm: str,
    output_path: Path | str,
    inpainter: Optional[BaseInpainter] = None,
    dilation_radius: int = 2,
    fill_value: int = 0,
) -> Path:
    """Load image and optional mask, apply the designated preprocessing arm, and save the result.

    Args:
        image_path: Path to source image.
        mask_path: Optional path to binary artifact mask.
        arm: Preprocessing arm identifier ('arm_a_raw', 'arm_b_inpainted', 'arm_c_masked').
        output_path: Destination path for preprocessed image.
        inpainter: Optional custom inpainter for Arm B.
        dilation_radius: Mask dilation radius (default 2px).
        fill_value: Neutral fill value for Arm C (default 0).

    Returns:
        Resolved output path.
    """
    img_p = Path(image_path).resolve()
    if not img_p.is_file():
        raise FileNotFoundError(f"Source image not found: {img_p}")

    with Image.open(img_p) as pil_img:
        img_arr = np.array(pil_img)

    mask_arr: Optional[np.ndarray] = None
    if mask_path is not None:
        m_p = Path(mask_path).resolve()
        if m_p.is_file():
            with Image.open(m_p) as pil_mask:
                raw_mask = np.array(pil_mask.convert("L"))
                mask_arr = raw_mask > 0

    arm_norm = arm.lower().strip()
    if arm_norm in ("arm_a_raw", "raw", "a"):
        processed = apply_arm_a(img_arr)
    elif arm_norm in ("arm_b_inpainted", "inpainted", "b"):
        processed = apply_arm_b(img_arr, mask_arr, inpainter=inpainter, dilation_radius=dilation_radius)
    elif arm_norm in ("arm_c_masked", "masked", "c"):
        processed = apply_arm_c(img_arr, mask_arr, fill_value=fill_value, dilation_radius=dilation_radius)
    else:
        raise ValueError(f"Unknown preprocessing arm '{arm}'. Must be arm_a_raw, arm_b_inpainted, or arm_c_masked.")

    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_pil = Image.fromarray(processed)
    out_pil.save(out_p)

    return out_p

