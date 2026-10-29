"""Typed configuration loader and schema validation for OvaryVision."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import yaml


class ProjectMetadata(BaseModel):
    name: str = "OvaryVision"
    version: str = "0.1.0"
    description: str = "Data-Centric Benchmark of Generative Inpainting vs Non-Generative Masking"


class PathsConfig(BaseModel):
    project_root: Path = Field(default_factory=lambda: Path(os.getenv("OVARYVISION_PROJECT_ROOT", ".")))
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    raw_dir: Path = Field(default_factory=lambda: Path("data/raw"))
    interim_dir: Path = Field(default_factory=lambda: Path("data/interim"))
    processed_dir: Path = Field(default_factory=lambda: Path("data/processed"))
    manifests_dir: Path = Field(default_factory=lambda: Path("data/manifests"))
    outputs_dir: Path = Field(default_factory=lambda: Path("outputs"))
    logs_dir: Path = Field(default_factory=lambda: Path("logs"))

    def resolve_paths(self, base_dir: Path) -> "PathsConfig":
        """Resolve all relative paths against the project root base directory."""
        return PathsConfig(
            project_root=base_dir,
            data_dir=(base_dir / self.data_dir).resolve(),
            raw_dir=(base_dir / self.raw_dir).resolve(),
            interim_dir=(base_dir / self.interim_dir).resolve(),
            processed_dir=(base_dir / self.processed_dir).resolve(),
            manifests_dir=(base_dir / self.manifests_dir).resolve(),
            outputs_dir=(base_dir / self.outputs_dir).resolve(),
            logs_dir=(base_dir / self.logs_dir).resolve(),
        )


class PreprocessingConfig(BaseModel):
    image_size: List[int] = Field(default_factory=lambda: [256, 256])
    channels: int = 3
    normalization: str = "imagenet"
    arms: List[str] = Field(default_factory=lambda: ["arm_a_raw", "arm_b_inpainted", "arm_c_masked"])
    neutral_fill_value: int = 0
    mask_dilation_radius: int = 2
    inpainting_engine: str = "mggan"

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, v: List[int]) -> List[int]:
        if len(v) != 2 or any(dim <= 0 for dim in v):
            raise ValueError(f"image_size must be a pair of positive integers [H, W], got {v}")
        return v

    @field_validator("neutral_fill_value")
    @classmethod
    def validate_fill_value(cls, v: int) -> int:
        if not (0 <= v <= 255):
            raise ValueError(f"neutral_fill_value must be in [0, 255], got {v}")
        return v


class ModelingConfig(BaseModel):
    available_backbones: List[str] = Field(default_factory=lambda: ["convnext_tiny", "efficientnet_b0"])
    default_backbone: str = "convnext_tiny"
    pretrained: bool = True
    num_classes: int = 2
    positive_class_index: int = 1


class LossConfig(BaseModel):
    name: str = "class_weighted_focal_loss"
    gamma: float = 2.0
    alpha_balanced: bool = True


class OptimizerConfig(BaseModel):
    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


class SchedulerConfig(BaseModel):
    name: str = "cosine_annealing"
    warmup_epochs: int = 5
    min_lr: float = 1e-6


class TrainingConfig(BaseModel):
    batch_size: int = 32
    num_workers: int = 2
    max_epochs: int = 50
    early_stopping_patience: int = 10
    monitor_metric: str = "val_macro_f1"
    mixed_precision: bool = True
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)


class CalibrationConfig(BaseModel):
    method: str = "temperature_scaling"
    num_bins: int = 15
    split_fit: str = "val"


class EvaluationConfig(BaseModel):
    seeds: List[int] = Field(default_factory=lambda: [42, 1337, 2026])
    non_inferiority_margin: float = -0.02
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95


class LoggingConfig(BaseModel):
    level: str = "INFO"
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5


class AppConfig(BaseModel):
    """Canonical application configuration model."""
    project: ProjectMetadata = Field(default_factory=ProjectMetadata)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Optional[Path | str] = None) -> AppConfig:
    """Load and validate an AppConfig from a YAML file.

    If config_path is None, loads from `configs/default.yaml` relative to current directory.
    Environment variable OVARYVISION_PROJECT_ROOT can override the project root path.
    """
    if config_path is None:
        # Check standard default locations
        candidates = [
            Path("configs/default.yaml"),
            Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml",
        ]
        for c in candidates:
            if c.is_file():
                config_path = c
                break

    data: Dict[str, Any] = {}
    if config_path:
        cfg_file = Path(config_path)
        if not cfg_file.is_file():
            raise FileNotFoundError(f"Configuration file not found: {cfg_file}")
        with open(cfg_file, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
            if isinstance(raw_data, dict):
                data = raw_data

    # Environment variable override for project root
    env_root = os.getenv("OVARYVISION_PROJECT_ROOT")
    if env_root and "paths" in data:
        data["paths"]["project_root"] = env_root

    config = AppConfig(**data)

    # Resolve paths against base project root
    base_root = config.paths.project_root.resolve()
    config.paths = config.paths.resolve_paths(base_root)
    return config

