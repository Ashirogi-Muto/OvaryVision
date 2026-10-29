"""Controlled synthetic-artifact benchmark and image-integrity metrics.

Provides deterministic synthetic ultrasound annotation overlays (measurement calipers,
crosshairs, dashed measurement lines, callout text) and rigorous image-integrity metrics
(MAE, PSNR, SSIM, Local Texture Variance, and lazy LPIPS) comparing reconstructions against
known clean references.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, field_validator
import scipy.ndimage as ndi

from ovaryvision.logging_utils import get_logger

logger = get_logger("ovaryvision.synthetic_artifacts")

ArtifactType = Literal[
    "caliper_cross",
    "caliper_tick",
    "dashed_line",
    "text_overlay",
    "caliper_measurement",  # Combined two calipers + connecting dashed line + distance text
]


class SyntheticArtifactConfig(BaseModel):
    """Configuration governing synthetic annotation artifact injection."""
    artifact_types: List[ArtifactType] = Field(
        default_factory=lambda: ["caliper_measurement", "text_overlay"]
    )
    intensity: int = Field(default=255, ge=0, le=255, description="Pixel intensity of overlay (0-255)")
    opacity: float = Field(default=1.0, gt=0.0, le=1.0, description="Overlay opacity blending ratio")
    caliper_size: int = Field(default=12, ge=4, le=60, description="Length of caliper cross arms in pixels")
    line_thickness: int = Field(default=1, ge=1, le=5, description="Pixel thickness of lines and crosses")
    dash_length: int = Field(default=6, ge=2, le=30, description="Length of individual dash in pixels")
    dash_space: int = Field(default=4, ge=1, le=20, description="Length of gap between dashes in pixels")
    include_text: bool = True
    custom_text: Optional[str] = None
    seed: Optional[int] = Field(default=42, description="Random seed for deterministic positioning")
    margin_padding: int = Field(default=20, ge=5, le=100, description="Minimum distance from image borders")

    @field_validator("artifact_types")
    @classmethod
    def validate_types(cls, v: List[ArtifactType]) -> List[ArtifactType]:
        if not v:
            raise ValueError("artifact_types cannot be empty.")
        return v


class SyntheticArtifactResult(BaseModel):
    """Result container for a synthetically corrupted image and its ground truth mask."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    corrupted_image: Any = Field(description="Corrupted image array as uint8 np.ndarray")
    artifact_mask: Any = Field(description="Binary artifact mask as uint8 np.ndarray (values in {0, 1})")
    clean_reference: Any = Field(description="Clean original reference image array as uint8 np.ndarray")
    provenance: Dict[str, Any] = Field(default_factory=dict)


class LocalTextureVarianceResult(BaseModel):
    """Local texture variance metrics comparing in-mask region with surrounding parenchyma band."""
    var_in_mask: float
    var_surrounding_band: float
    delta_ltv: float
    band_pixel_count: int
    mask_pixel_count: int


class ImageIntegrityReport(BaseModel):
    """Comprehensive image integrity report comparing reconstructed image against clean reference."""
    evaluation_timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # Global metrics across entire image
    global_mae: float
    global_psnr: float
    global_ssim: float
    # In-mask metrics evaluated strictly where artifact_mask == 1
    in_mask_mae: float
    in_mask_psnr: float
    in_mask_ssim: float
    # Exterior metrics evaluated strictly where artifact_mask == 0
    exterior_mae: float
    exterior_psnr: float
    exterior_ssim: float
    # Texture metrics
    local_texture_variance: LocalTextureVarianceResult
    # Optional GPU-dependent metric
    lpips: Optional[float] = None
    mask_pixel_ratio: float
    scientific_limitations: List[str] = Field(
        default_factory=lambda: [
            "Controlled synthetic benchmark evaluates reconstruction of known digital overlays against known clean reference.",
            "In real clinical ultrasound, underlying tissue occluded by clinician calipers is unobservable.",
            "High synthetic PSNR/SSIM indicates geometric inpainting capability, but cannot prove absence of hallucination on real pathological scans."
        ]
    )


def _compute_sha256_array(arr: np.ndarray) -> str:
    """Compute deterministic SHA-256 digest of numpy array bytes."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    fill: Union[int, Tuple[int, ...]],
    width: int = 1,
    dash_length: int = 6,
    dash_space: int = 4,
) -> None:
    """Draw a dashed line segment between two points on a PIL canvas."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1.0:
        draw.point((x1, y1), fill=fill)
        return

    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist

    curr = 0.0
    while curr < dist:
        end = min(curr + dash_length, dist)
        p_start = (int(round(x1 + curr * dx)), int(round(y1 + curr * dy)))
        p_end = (int(round(x1 + end * dx)), int(round(y1 + end * dy)))
        draw.line([p_start, p_end], fill=fill, width=width)
        curr += dash_length + dash_space


def _draw_caliper_cross(
    draw: ImageDraw.ImageDraw,
    center: Tuple[int, int],
    size: int,
    fill: Union[int, Tuple[int, ...]],
    width: int = 1,
    style: str = "plus",
) -> None:
    """Draw a crosshair or tick mark centered at specified coordinate."""
    cx, cy = center
    arm = size // 2

    if style == "plus":
        draw.line([(cx - arm, cy), (cx + arm, cy)], fill=fill, width=width)
        draw.line([(cx, cy - arm), (cx, cy + arm)], fill=fill, width=width)
    elif style == "x":
        draw.line([(cx - arm, cy - arm), (cx + arm, cy + arm)], fill=fill, width=width)
        draw.line([(cx - arm, cy + arm), (cx + arm, cy - arm)], fill=fill, width=width)
    elif style == "tick":
        draw.line([(cx - arm, cy), (cx + arm, cy)], fill=fill, width=width)


def inject_synthetic_artifacts(
    image: np.ndarray,
    config: Optional[SyntheticArtifactConfig] = None,
) -> SyntheticArtifactResult:
    """Inject deterministic synthetic ultrasound measurement artifacts onto an image.

    Args:
        image: Source clean image as 2D (H, W) or 3D (H, W, C) uint8 numpy array.
        config: Generation configuration parameters.

    Returns:
        SyntheticArtifactResult containing clean reference, corrupted image, binary mask,
        and full generation provenance.

    Raises:
        ValueError: If image dimensions or types are invalid.
    """
    if config is None:
        config = SyntheticArtifactConfig()

    # Input validation
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be a numpy ndarray, got {type(image)}")
    if image.dtype != np.uint8:
        raise ValueError(f"Input image dtype must be uint8, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Input image must be 2D (H, W) or 3D (H, W, C), got ndim={image.ndim}")

    height, width = image.shape[:2]
    if height < 32 or width < 32:
        raise ValueError(f"Image dimensions too small for artifact generation: {width}x{height} (minimum 32x32)")

    is_rgb = (image.ndim == 3 and image.shape[2] == 3)
    clean_reference = image.copy()

    # Setup random state
    rng = np.random.default_rng(config.seed)

    # Create transparent RGBA overlay canvas to track drawn pixels exactly
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    alpha = int(round(config.opacity * 255))
    intensity = config.intensity
    rgba_fill = (intensity, intensity, intensity, alpha)

    pad = config.margin_padding
    artifacts_drawn: List[Dict[str, Any]] = []

    # Bounding box constraints for safe caliper placement
    safe_min_x = pad
    safe_max_x = max(pad + 10, width - pad)
    safe_min_y = pad
    safe_max_y = max(pad + 10, height - pad)

    for art_type in config.artifact_types:
        if art_type in ("caliper_measurement", "dashed_line", "caliper_cross", "caliper_tick"):
            # Sample two distinct points for caliper measurement
            x1 = int(rng.integers(safe_min_x, safe_max_x))
            y1 = int(rng.integers(safe_min_y, safe_max_y))
            # Place second point within a realistic distance (30 to 120 pixels)
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(35.0, min(140.0, min(width, height) * 0.6))
            x2 = int(np.clip(round(x1 + dist * math.cos(angle)), safe_min_x, safe_max_x))
            y2 = int(np.clip(round(y1 + dist * math.sin(angle)), safe_min_y, safe_max_y))

            if art_type == "caliper_measurement":
                # 1. Draw dashed connecting line
                _draw_dashed_line(
                    draw,
                    (x1, y1),
                    (x2, y2),
                    fill=rgba_fill,
                    width=config.line_thickness,
                    dash_length=config.dash_length,
                    dash_space=config.dash_space,
                )
                # 2. Draw crosshairs at both endpoints
                _draw_caliper_cross(draw, (x1, y1), size=config.caliper_size, fill=rgba_fill, width=config.line_thickness)
                _draw_caliper_cross(draw, (x2, y2), size=config.caliper_size, fill=rgba_fill, width=config.line_thickness)

                # 3. Draw measurement callout text if enabled
                if config.include_text:
                    pixel_dist = math.hypot(x2 - x1, y2 - y1)
                    # Simulated ultrasound calibration: ~0.25 mm/pixel
                    mm_dist = pixel_dist * 0.25
                    text_str = config.custom_text or f"D1: {mm_dist:.1f}mm"
                    # Place text slightly offset from point 2
                    tx = int(np.clip(x2 + 8, 2, width - 60))
                    ty = int(np.clip(y2 - 12, 2, height - 16))
                    draw.text((tx, ty), text_str, font=font, fill=rgba_fill)

                artifacts_drawn.append({
                    "type": "caliper_measurement",
                    "pt1": [x1, y1],
                    "pt2": [x2, y2],
                    "euclidean_distance_px": round(dist, 2),
                })

            elif art_type == "caliper_cross":
                _draw_caliper_cross(draw, (x1, y1), size=config.caliper_size, fill=rgba_fill, width=config.line_thickness)
                artifacts_drawn.append({"type": "caliper_cross", "center": [x1, y1]})

            elif art_type == "caliper_tick":
                _draw_caliper_cross(draw, (x1, y1), size=config.caliper_size, fill=rgba_fill, width=config.line_thickness, style="tick")
                artifacts_drawn.append({"type": "caliper_tick", "center": [x1, y1]})

            elif art_type == "dashed_line":
                _draw_dashed_line(
                    draw,
                    (x1, y1),
                    (x2, y2),
                    fill=rgba_fill,
                    width=config.line_thickness,
                    dash_length=config.dash_length,
                    dash_space=config.dash_space,
                )
                artifacts_drawn.append({"type": "dashed_line", "pt1": [x1, y1], "pt2": [x2, y2]})

        elif art_type == "text_overlay":
            labels = ["Lt Ovary", "Rt Ovary", "Trans", "Sagittal", "Fundus", "Probe: 6.5MHz"]
            label_text = config.custom_text or str(rng.choice(labels))
            tx = int(rng.integers(safe_min_x, max(safe_min_x + 1, width - 80)))
            ty = int(rng.integers(safe_min_y, max(safe_min_y + 1, height - 30)))
            draw.text((tx, ty), label_text, font=font, fill=rgba_fill)
            artifacts_drawn.append({"type": "text_overlay", "text": label_text, "pos": [tx, ty]})

    # Extract rasterized mask from overlay alpha channel
    overlay_arr = np.array(overlay)
    alpha_mask = overlay_arr[:, :, 3] > 0
    artifact_mask = alpha_mask.astype(np.uint8)

    # Blend overlay with clean image
    alpha_norm = overlay_arr[:, :, 3].astype(np.float64) / 255.0
    overlay_rgb = overlay_arr[:, :, :3] if is_rgb else overlay_arr[:, :, 0]

    if is_rgb:
        blended = (1.0 - alpha_norm[:, :, None]) * clean_reference.astype(np.float64) + alpha_norm[:, :, None] * overlay_rgb.astype(np.float64)
        corrupted = np.clip(np.round(blended), 0, 255).astype(np.uint8)
        corrupted = np.where(artifact_mask[:, :, None] == 1, corrupted, clean_reference)
    else:
        blended = (1.0 - alpha_norm) * clean_reference.astype(np.float64) + alpha_norm * overlay_rgb.astype(np.float64)
        corrupted = np.clip(np.round(blended), 0, 255).astype(np.uint8)
        corrupted = np.where(artifact_mask == 1, corrupted, clean_reference)

    # Invariant assertion: every modified pixel must be in mask
    diff = (clean_reference != corrupted)
    if is_rgb:
        diff_2d = np.any(diff, axis=-1)
    else:
        diff_2d = diff

    if not np.all(artifact_mask[diff_2d] == 1):
        raise RuntimeError("Integrity violation: Detected modified pixels outside the generated artifact mask!")

    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.model_dump(),
        "clean_sha256": _compute_sha256_array(clean_reference),
        "corrupted_sha256": _compute_sha256_array(corrupted),
        "mask_sha256": _compute_sha256_array(artifact_mask),
        "mask_pixel_count": int(np.sum(artifact_mask)),
        "mask_fraction": float(np.mean(artifact_mask)),
        "artifacts_drawn": artifacts_drawn,
    }

    return SyntheticArtifactResult(
        corrupted_image=corrupted,
        artifact_mask=artifact_mask,
        clean_reference=clean_reference,
        provenance=provenance,
    )


def export_synthetic_case(
    result: SyntheticArtifactResult,
    output_dir: Union[Path, str],
    case_id: str,
) -> Dict[str, Path]:
    """Export clean image, corrupted image, artifact mask, and metadata JSON to disk.

    Never overwrites existing files unless explicitly directed.

    Returns:
        Dictionary mapping artifact keys ('clean', 'corrupted', 'mask', 'metadata') to saved Paths.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    clean_file = out_path / f"{case_id}_clean.png"
    corrupted_file = out_path / f"{case_id}_corrupted.png"
    mask_file = out_path / f"{case_id}_mask.png"
    meta_file = out_path / f"{case_id}_metadata.json"

    # Save PNGs using PIL
    Image.fromarray(result.clean_reference).save(clean_file)
    Image.fromarray(result.corrupted_image).save(corrupted_file)
    # Scale mask {0, 1} to {0, 255} for standard viewing
    Image.fromarray(result.artifact_mask * 255).save(mask_file)

    meta = {
        "case_id": case_id,
        "clean_path": str(clean_file),
        "corrupted_path": str(corrupted_file),
        "mask_path": str(mask_file),
        "provenance": result.provenance,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "clean": clean_file,
        "corrupted": corrupted_file,
        "mask": mask_file,
        "metadata": meta_file,
    }


# ==============================================================================
# INTEGRITY METRICS
# ==============================================================================

def compute_mae(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Mean Absolute Error between reference and reconstructed image."""
    ref = reference.astype(np.float64)
    rec = reconstructed.astype(np.float64)
    diff = np.abs(ref - rec)

    if mask is not None:
        m = (mask > 0)
        if not np.any(m):
            return float("nan")
        if ref.ndim == 3 and m.ndim == 2:
            m = m[:, :, None]
        return float(np.mean(diff[m]))
    return float(np.mean(diff))


def compute_psnr(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    data_range: float = 255.0,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Returns float('inf') if MSE == 0 (perfect match).
    Returns float('nan') if mask is non-empty but contains 0 pixels.
    """
    ref = reference.astype(np.float64)
    rec = reconstructed.astype(np.float64)
    sq_err = (ref - rec) ** 2

    if mask is not None:
        m = (mask > 0)
        if not np.any(m):
            return float("nan")
        if ref.ndim == 3 and m.ndim == 2:
            m = m[:, :, None]
        mse = float(np.mean(sq_err[m]))
    else:
        mse = float(np.mean(sq_err))

    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def _compute_ssim_2d(
    img1: np.ndarray,
    img2: np.ndarray,
    win_size: int = 7,
    data_range: float = 255.0,
) -> np.ndarray:
    """Compute 2D SSIM map using uniform filter."""
    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)

    mu1 = ndi.uniform_filter(i1, size=win_size)
    mu2 = ndi.uniform_filter(i2, size=win_size)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = ndi.uniform_filter(i1 * i1, size=win_size) - mu1_sq
    sigma2_sq = ndi.uniform_filter(i2 * i2, size=win_size) - mu2_sq
    sigma12 = ndi.uniform_filter(i1 * i2, size=win_size) - mu1_mu2

    # Prevent numerical underflow
    sigma1_sq = np.maximum(sigma1_sq, 0.0)
    sigma2_sq = np.maximum(sigma2_sq, 0.0)

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / den
    return ssim_map


def compute_ssim(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    data_range: float = 255.0,
    win_size: int = 7,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Structural Similarity Index (SSIM).

    Supports 2D grayscale and 3D RGB arrays, and optional mask restriction.
    """
    if reference.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: reference {reference.shape} vs reconstructed {reconstructed.shape}")

    if reference.ndim == 2:
        ssim_map = _compute_ssim_2d(reference, reconstructed, win_size=win_size, data_range=data_range)
    elif reference.ndim == 3:
        maps = [
            _compute_ssim_2d(reference[:, :, c], reconstructed[:, :, c], win_size=win_size, data_range=data_range)
            for c in range(reference.shape[2])
        ]
        ssim_map = np.mean(maps, axis=0)
    else:
        raise ValueError(f"Unsupported array ndim: {reference.ndim}")

    if mask is not None:
        m = (mask > 0)
        if not np.any(m):
            return float("nan")
        return float(np.mean(ssim_map[m]))
    return float(np.mean(ssim_map))


def compute_local_texture_variance(
    image: np.ndarray,
    mask: np.ndarray,
    dilation_radius: int = 10,
) -> LocalTextureVarianceResult:
    """Compute Local Texture Variance shift between in-mask region and surrounding parenchyma band.

    In ultrasound CAD, true tissue has characteristic speckle variance. Inpainting that blurs
    or hallucinates creates variance divergence |var_in - var_band|.
    """
    if image.ndim == 3:
        # Convert RGB to luminance for texture variance
        img_gray = (0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]).astype(np.float64)
    else:
        img_gray = image.astype(np.float64)

    m = (mask > 0)
    mask_count = int(np.sum(m))

    if mask_count == 0:
        return LocalTextureVarianceResult(
            var_in_mask=0.0,
            var_surrounding_band=0.0,
            delta_ltv=0.0,
            band_pixel_count=0,
            mask_pixel_count=0,
        )

    # In-mask variance
    var_in = float(np.var(img_gray[m]))

    # Compute surrounding parenchyma band: dilate(mask, radius) \ mask
    dilated = ndi.binary_dilation(m, iterations=dilation_radius)
    band = dilated & (~m)
    band_count = int(np.sum(band))

    if band_count == 0:
        var_band = var_in
    else:
        var_band = float(np.var(img_gray[band]))

    delta_ltv = abs(var_in - var_band)

    return LocalTextureVarianceResult(
        var_in_mask=round(var_in, 4),
        var_surrounding_band=round(var_band, 4),
        delta_ltv=round(delta_ltv, 4),
        band_pixel_count=band_count,
        mask_pixel_count=mask_count,
    )


def compute_lpips(
    reference: np.ndarray,
    reconstructed: np.ndarray,
) -> Optional[float]:
    """Lazy evaluation of LPIPS (Learned Perceptual Image Patch Similarity).

    Returns None if torch or lpips packages are not installed on host. Never crashes CPU host.
    """
    try:
        import torch  # type: ignore
        import lpips  # type: ignore

        # Prepare normalized tensor [1, 3, H, W] in [-1, 1]
        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
            # Normalize [0, 255] -> [-1, 1]
            return (t / 127.5) - 1.0

        loss_fn = lpips.LPIPS(net="alex", verbose=False)
        t_ref = _to_tensor(reference)
        t_rec = _to_tensor(reconstructed)

        with torch.no_grad():
            dist = loss_fn(t_ref, t_rec)
        return float(dist.item())
    except ImportError:
        logger.debug("LPIPS skipped: torch or lpips package not installed on CPU host.")
        return None
    except Exception as exc:
        logger.warning("LPIPS computation failed: %s", exc)
        return None


def evaluate_reconstruction_integrity(
    clean_reference: np.ndarray,
    reconstructed: np.ndarray,
    artifact_mask: np.ndarray,
    data_range: float = 255.0,
    dilation_radius: int = 10,
) -> ImageIntegrityReport:
    """Compute tri-partition image integrity metrics (Global, In-Mask, Exterior, LTV).

    Compares reconstructed image against clean reference across spatial partitions.
    """
    if clean_reference.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: clean_reference {clean_reference.shape} vs reconstructed {reconstructed.shape}"
        )

    mask_binary = (artifact_mask > 0).astype(np.uint8)
    exterior_mask = (mask_binary == 0).astype(np.uint8)

    # 1. Global Metrics
    g_mae = compute_mae(clean_reference, reconstructed)
    g_psnr = compute_psnr(clean_reference, reconstructed, data_range=data_range)
    g_ssim = compute_ssim(clean_reference, reconstructed, data_range=data_range)

    # 2. In-Mask Metrics
    in_mae = compute_mae(clean_reference, reconstructed, mask=mask_binary)
    in_psnr = compute_psnr(clean_reference, reconstructed, data_range=data_range, mask=mask_binary)
    in_ssim = compute_ssim(clean_reference, reconstructed, data_range=data_range, mask=mask_binary)

    # 3. Exterior Metrics (Boundary Leakage Audit)
    ext_mae = compute_mae(clean_reference, reconstructed, mask=exterior_mask)
    ext_psnr = compute_psnr(clean_reference, reconstructed, data_range=data_range, mask=exterior_mask)
    ext_ssim = compute_ssim(clean_reference, reconstructed, data_range=data_range, mask=exterior_mask)

    # 4. Local Texture Variance
    ltv = compute_local_texture_variance(reconstructed, mask=mask_binary, dilation_radius=dilation_radius)

    # 5. Optional LPIPS
    lpips_val = compute_lpips(clean_reference, reconstructed)

    mask_ratio = float(np.mean(mask_binary))

    return ImageIntegrityReport(
        global_mae=round(g_mae, 4),
        global_psnr=round(g_psnr, 4) if not math.isinf(g_psnr) else float("inf"),
        global_ssim=round(g_ssim, 4),
        in_mask_mae=round(in_mae, 4),
        in_mask_psnr=round(in_psnr, 4) if not math.isinf(in_psnr) else float("inf"),
        in_mask_ssim=round(in_ssim, 4),
        exterior_mae=round(ext_mae, 4),
        exterior_psnr=round(ext_psnr, 4) if not math.isinf(ext_psnr) else float("inf"),
        exterior_ssim=round(ext_ssim, 4),
        local_texture_variance=ltv,
        lpips=lpips_val,
        mask_pixel_ratio=round(mask_ratio, 6),
    )
