"""Reproducibility and execution provenance tracking for OvaryVision."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ProvenanceRecord(BaseModel):
    """Immutable execution provenance snapshot for an experiment run."""
    run_id: str
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit_hash: str = "no_git_repo"
    git_is_dirty: bool = False
    config_hash: str
    python_version: str = Field(default_factory=platform.python_version)
    platform: str = Field(default_factory=platform.platform)
    cpu_model: str = "Unknown"
    cuda_available: bool = False
    seed: int
    arm: Optional[str] = None
    backbone: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def compute_config_hash(config_obj: Dict[str, Any] | str | Path) -> str:
    """Compute a deterministic SHA-256 hash of a configuration dictionary or file."""
    if isinstance(config_obj, (str, Path)):
        p = Path(config_obj)
        if p.is_file():
            content = p.read_bytes()
            return hashlib.sha256(content).hexdigest()
        else:
            content = str(config_obj).encode("utf-8")
            return hashlib.sha256(content).hexdigest()
    elif isinstance(config_obj, dict):
        # Canonical JSON string with sorted keys
        canonical_json = json.dumps(config_obj, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(canonical_json).hexdigest()
    else:
        return hashlib.sha256(str(config_obj).encode("utf-8")).hexdigest()


def _get_git_info() -> tuple[str, bool]:
    """Safely retrieve git commit hash and dirty status without throwing."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0:
            commit = res.stdout.strip()
            dirty_res = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            is_dirty = bool(dirty_res.stdout.strip())
            return commit, is_dirty
    except Exception:
        pass
    return "no_git_repo", False


def create_provenance_record(
    run_id: str,
    config_dict: Dict[str, Any] | str,
    seed: int,
    arm: Optional[str] = None,
    backbone: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceRecord:
    """Create a standardized provenance record."""
    git_hash, is_dirty = _get_git_info()
    cfg_hash = compute_config_hash(config_dict)

    # Check CUDA availability if torch is present
    cuda_avail = False
    try:
        import torch  # type: ignore
        cuda_avail = torch.cuda.is_available()
    except ImportError:
        pass

    return ProvenanceRecord(
        run_id=run_id,
        git_commit_hash=git_hash,
        git_is_dirty=is_dirty,
        config_hash=cfg_hash,
        seed=seed,
        arm=arm,
        backbone=backbone,
        cuda_available=cuda_avail,
        metadata=metadata or {},
    )


def save_provenance_record(record: ProvenanceRecord, output_path: Path | str) -> None:
    """Save provenance record to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record.model_dump(), f, indent=2)

