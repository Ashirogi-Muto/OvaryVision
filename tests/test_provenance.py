"""Unit tests for execution provenance and configuration hashing."""

import json
from pathlib import Path
from ovaryvision.provenance import (
    compute_config_hash,
    create_provenance_record,
    save_provenance_record,
)


def test_compute_config_hash():
    """Verify deterministic hashing of dictionary configurations."""
    cfg1 = {"model": "convnext", "lr": 0.001, "seed": 42}
    cfg2 = {"seed": 42, "lr": 0.001, "model": "convnext"}  # Reordered keys
    cfg3 = {"model": "convnext", "lr": 0.0001, "seed": 42}

    h1 = compute_config_hash(cfg1)
    h2 = compute_config_hash(cfg2)
    h3 = compute_config_hash(cfg3)

    assert len(h1) == 64
    assert h1 == h2  # Order-independent
    assert h1 != h3  # Different value yields different hash


def test_provenance_record_creation(tmp_path: Path):
    """Verify provenance record creation and serialization."""
    rec = create_provenance_record(
        run_id="TEST_RUN_001",
        config_dict={"arm": "arm_c_masked"},
        seed=42,
        arm="arm_c_masked",
        backbone="convnext_tiny",
    )
    assert rec.run_id == "TEST_RUN_001"
    assert rec.arm == "arm_c_masked"
    assert rec.seed == 42
    assert len(rec.config_hash) == 64

    out_file = tmp_path / "provenance.json"
    save_provenance_record(rec, out_file)
    assert out_file.is_file()

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["run_id"] == "TEST_RUN_001"

