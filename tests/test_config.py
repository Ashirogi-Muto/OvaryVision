"""Unit tests for OvaryVision configuration loader and schemas."""

from pathlib import Path
import pytest
from pydantic import ValidationError
import yaml

from ovaryvision.config import (
    AppConfig,
    load_config,
    PreprocessingConfig,
    LossConfig,
    PathsConfig,
)


def test_default_config_loading():
    """Verify default configuration loads properly and passes validation."""
    cfg = load_config()
    assert cfg.project.name == "OvaryVision"
    assert cfg.modeling.default_backbone in ["convnext_small", "efficientnet_b3", "convnext_tiny", "efficientnet_b0"]
    assert cfg.modeling.default_backbone == "convnext_small"
    assert "arm_c_masked" in cfg.preprocessing.arms
    assert cfg.training.loss.gamma == 2.0
    assert cfg.paths.project_root.is_absolute()


def test_invalid_yaml_file(tmp_path: Path):
    """Verify loading non-existent or invalid configuration raises appropriate error."""
    non_existent = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(non_existent)

    invalid_yaml = tmp_path / "corrupt.yaml"
    invalid_yaml.write_text("preprocessing:\n  image_size: [-1, 256]\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(invalid_yaml)


def test_preprocessing_validation():
    """Verify PreprocessingConfig validates image size and neutral fill values."""
    # Valid config
    p = PreprocessingConfig(image_size=[224, 224], neutral_fill_value=0)
    assert p.image_size == [224, 224]

    # Invalid image size dimensions
    with pytest.raises(ValidationError):
        PreprocessingConfig(image_size=[256])

    with pytest.raises(ValidationError):
        PreprocessingConfig(image_size=[0, 256])

    # Invalid fill value (outside 0-255)
    with pytest.raises(ValidationError):
        PreprocessingConfig(neutral_fill_value=300)


def test_path_resolution(tmp_path: Path):
    """Verify relative paths resolve properly against project_root."""
    p = PathsConfig(project_root=tmp_path, data_dir=Path("data"))
    resolved = p.resolve_paths(tmp_path)
    assert resolved.data_dir == (tmp_path / "data").resolve()
    assert resolved.project_root == tmp_path

