"""Unit tests for the OvaryVision CLI subcommands."""

import json
from pathlib import Path
from click.testing import CliRunner
import numpy as np
from PIL import Image

from ovaryvision.cli import main


def test_cli_check_env():
    """Verify check-env command runs and outputs diagnostic information."""
    runner = CliRunner()
    res = runner.invoke(main, ["check-env"])
    assert res.exit_code == 0
    assert "OVARYVISION ENVIRONMENT DIAGNOSTICS" in res.output
    assert "CPU Cores" in res.output


def test_cli_check_env_json():
    """Verify check-env --json-out outputs valid parseable JSON."""
    runner = CliRunner()
    res = runner.invoke(main, ["check-env", "--json-out"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert "platform_detected" in data
    assert "hardware" in data


def test_cli_validate_config():
    """Verify validate-config succeeds on default.yaml."""
    runner = CliRunner()
    res = runner.invoke(main, ["validate-config", "configs/default.yaml"])
    assert res.exit_code == 0
    assert "is VALID" in res.output


def test_cli_init_dirs(tmp_path: Path):
    """Verify init-dirs creates expected directories."""
    runner = CliRunner()
    # Test running init-dirs
    res = runner.invoke(main, ["init-dirs"])
    assert res.exit_code == 0
    assert "successfully initialized" in res.output


def test_cli_generate_template(tmp_path: Path):
    """Verify generate-manifest-template outputs a valid CSV."""
    runner = CliRunner()
    out_file = tmp_path / "template.csv"
    res = runner.invoke(main, ["generate-manifest-template", "--output", str(out_file)])
    assert res.exit_code == 0
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert "image_id" in content
    assert "artifact_mask_path" in content


def test_cli_audit_data_missing_dir(tmp_path: Path):
    """Verify audit-data fails gracefully when directory does not exist."""
    runner = CliRunner()
    missing_dir = tmp_path / "missing_dir"
    res = runner.invoke(main, ["audit-data", "--data-dir", str(missing_dir)])
    assert res.exit_code != 0
    assert "does not exist" in res.output


def test_cli_audit_data_success(tmp_path: Path):
    """Verify audit-data succeeds on directory with valid image."""
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    img_path = data_dir / "sample.png"
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(img_path)

    out_dir = tmp_path / "audit_out"

    runner = CliRunner()
    res = runner.invoke(
        main,
        ["audit-data", "--data-dir", str(data_dir), "--output-dir", str(out_dir)],
    )
    assert res.exit_code == 0
    assert "DATASET AUDIT REPORT" in res.output
    assert "Valid Images Found    : 1" in res.output
    assert (out_dir / "data_audit_report.json").is_file()


def test_cli_ingest_guide():
    """Verify ingest-guide command displays authoritative sources and safety rules."""
    runner = CliRunner()
    res = runner.invoke(main, ["ingest-guide"])
    assert res.exit_code == 0
    assert "OVARYVISION DATASET INGESTION GUIDELINES" in res.output
    assert "MMOTU" in res.output
    assert "Zenodo" in res.output
    assert "OvaTUS-V1" in res.output


def test_cli_generate_synthetic_benchmark(tmp_path: Path):
    """Verify generate-synthetic-benchmark generates cases and computes metrics."""
    out_dir = tmp_path / "synth_cases"
    metrics_file = tmp_path / "metrics.json"

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "generate-synthetic-benchmark",
            "--output-dir", str(out_dir),
            "--num-cases", "2",
            "--seed", "100",
            "--metrics-out", str(metrics_file),
        ],
    )
    assert res.exit_code == 0
    assert "CONTROLLED SYNTHETIC-ARTIFACT BENCHMARK SUITE" in res.output
    assert (out_dir / "SYNTH_CASE_0001_clean.png").is_file()
    assert (out_dir / "SYNTH_CASE_0001_corrupted.png").is_file()
    assert (out_dir / "SYNTH_CASE_0001_mask.png").is_file()
    assert metrics_file.is_file()


def test_cli_generate_splits(tmp_path: Path):
    """Verify generate-splits command creates split manifests across seeds."""
    from ovaryvision.manifests import ImageRecord, save_image_manifest

    manifest_file = tmp_path / "images.csv"
    recs = [
        ImageRecord(
            image_id=f"IMG_{i:04d}",
            dataset_id="MMOTU_2D",
            relative_image_path=f"data/raw/mmotu/img_{i}.png",
            sha256_checksum="c" * 64,
            width=256,
            height=256,
            channels=1,
            file_size_bytes=1024,
            diagnostic_label=i % 2,
            pseudo_patient_cluster=f"CLUSTER_{i % 5:02d}",
        )
        for i in range(20)
    ]
    save_image_manifest(recs, manifest_file)

    split_dir = tmp_path / "splits"
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "generate-splits",
            "--manifest", str(manifest_file),
            "--output-dir", str(split_dir),
            "--seeds", "42,1337",
        ],
    )
    assert res.exit_code == 0
    assert "GROUP-STRATIFIED SPLIT ENGINE" in res.output
    assert (split_dir / "mmotu_split_seed42.csv").is_file()
    assert (split_dir / "mmotu_split_seed1337.csv").is_file()


def test_cli_run_preprocessing(tmp_path: Path):
    """Verify run-preprocessing processes manifest records."""
    from ovaryvision.manifests import ImageRecord, save_image_manifest

    img_dir = tmp_path / "raw"
    img_dir.mkdir()
    img_file = img_dir / "test.png"
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(img_file)

    manifest_file = tmp_path / "manifest.csv"
    recs = [
        ImageRecord(
            image_id="test",
            dataset_id="MMOTU_2D",
            relative_image_path=str(img_file),
            sha256_checksum="d" * 64,
            width=64,
            height=64,
            channels=1,
            file_size_bytes=100,
            diagnostic_label=0,
        )
    ]
    save_image_manifest(recs, manifest_file)

    out_base = tmp_path / "processed"
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "run-preprocessing",
            "--manifest", str(manifest_file),
            "--output-base-dir", str(out_base),
            "--arm", "arm_a_raw",
        ],
    )
    assert res.exit_code == 0
    assert (out_base / "arm_a_raw" / "test.png").is_file()


def test_cli_evaluate_calibration(tmp_path: Path):
    """Verify evaluate-calibration fits temperature scaling and exports report JSON."""
    import csv

    val_file = tmp_path / "val_logits.csv"
    test_file = tmp_path / "test_logits.csv"
    report_file = tmp_path / "calib_report.json"

    with open(val_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "binary_logit", "true_label"])
        for i in range(50):
            writer.writerow([f"V_{i}", 1.5 if i % 2 == 1 else -1.5, i % 2])

    with open(test_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "binary_logit", "true_label"])
        for i in range(50):
            writer.writerow([f"T_{i}", 1.5 if i % 2 == 1 else -1.5, i % 2])

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "evaluate-calibration",
            "--val-logits-file", str(val_file),
            "--test-logits-file", str(test_file),
            "--output", str(report_file),
        ],
    )
    assert res.exit_code == 0
    assert "Calibration report saved" in res.output
    assert report_file.is_file()


def test_cli_evaluate_statistics(tmp_path: Path):
    """Verify evaluate-statistics executes paired testing and outputs report."""
    import csv

    c_file = tmp_path / "preds_c.csv"
    b_file = tmp_path / "preds_b.csv"
    stat_report = tmp_path / "stat_report.json"

    with open(c_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "calibrated_prob", "true_label"])
        for i in range(60):
            writer.writerow([f"IMG_{i}", 0.8 if i % 2 == 1 else 0.2, i % 2])

    with open(b_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "calibrated_prob", "true_label"])
        for i in range(60):
            writer.writerow([f"IMG_{i}", 0.75 if i % 2 == 1 else 0.25, i % 2])

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "evaluate-statistics",
            "--arm-c-preds", str(c_file),
            "--arm-b-preds", str(b_file),
            "--output", str(stat_report),
            "--num-bootstrap", "100",
        ],
    )
    assert res.exit_code == 0
    assert "Statistical hypothesis testing report saved" in res.output
    assert stat_report.is_file()


