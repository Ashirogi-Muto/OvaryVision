"""Unit tests for environment diagnostics and hardware inspection."""

import platform
from ovaryvision.environment import (
    get_environment_diagnostics,
    _get_disk_info,
    _get_memory_info,
    _detect_platform,
    _check_packages,
)


def test_environment_diagnostics():
    """Verify that get_environment_diagnostics returns a valid, populated model."""
    diag = get_environment_diagnostics()
    assert diag.python_version == platform.python_version()
    assert diag.hardware.cpu_count_logical >= 1
    assert diag.hardware.disk_total_gb > 0.0
    assert diag.hardware.memory_total_gb > 0.0
    assert diag.platform_detected in ["oci", "colab", "kaggle", "generic_linux"]
    assert isinstance(diag.cuda.gpu_available, bool)
    assert len(diag.training_recommendation) > 0


def test_disk_info():
    """Verify disk info returns non-negative values."""
    total, free = _get_disk_info(".")
    assert total > 0.0
    assert free >= 0.0
    assert total >= free


def test_package_checks():
    """Verify core packages like pydantic and numpy are recognized."""
    pkgs = _check_packages()
    assert pkgs["pydantic"] is not None
    assert pkgs["numpy"] is not None
    assert pkgs["pyyaml"] is not None

