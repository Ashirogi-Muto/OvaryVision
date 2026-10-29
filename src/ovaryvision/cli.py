"""Command-line interface (CLI) for OvaryVision research project."""

import json
from pathlib import Path
import sys
import click

from ovaryvision import __version__
from ovaryvision.config import load_config
from ovaryvision.data_audit import audit_image_dataset
from ovaryvision.environment import get_environment_diagnostics
from ovaryvision.logging_utils import get_logger
from ovaryvision.manifests import generate_manifest_template_csv

from ovaryvision.synthetic_artifacts import (
    SyntheticArtifactConfig,
    inject_synthetic_artifacts,
    export_synthetic_case,
    evaluate_reconstruction_integrity,
)
from ovaryvision.splits import (
    SplitRatio,
    compute_split_summary,
    generate_and_save_splits,
)
from ovaryvision.manifests import load_image_manifest
from ovaryvision.preprocessing import (
    HarmonicDiffusionInpainter,
    process_single_image,
)
from ovaryvision.calibration import (
    compute_calibration_bins,
    compute_brier_score,
    evaluate_probability_calibration,
)
from ovaryvision.statistics import (
    generate_comprehensive_statistical_report,
)

logger = get_logger("ovaryvision.cli")


@click.group()
@click.version_option(version=__version__, prog_name="ovaryvision")
def main() -> None:
    """OvaryVision: Preprocessing Benchmark for Ovarian Ultrasound CAD."""
    pass


@main.command(name="check-env")
@click.option("--json-out", is_flag=True, help="Print machine-readable JSON output.")
def check_env_cmd(json_out: bool) -> None:
    """Inspect host compute environment, hardware, memory, and CUDA availability."""
    diag = get_environment_diagnostics()

    if json_out:
        click.echo(json.dumps(diag.model_dump(), indent=2))
        return

    click.secho("\n=======================================================", fg="cyan", bold=True)
    click.secho("         OVARYVISION ENVIRONMENT DIAGNOSTICS          ", fg="cyan", bold=True)
    click.secho("=======================================================", fg="cyan", bold=True)
    click.echo(f"  Platform Detected  : {diag.platform_detected.upper()}")
    click.echo(f"  OS / Architecture  : {diag.hardware.os_name} {diag.hardware.os_release} ({diag.hardware.architecture})")
    click.echo(f"  Python Version     : {diag.python_version} ({diag.python_executable})")
    click.echo(f"  CPU Cores / Model  : {diag.hardware.cpu_count_logical} vCPUs ({diag.hardware.cpu_model})")
    click.echo(f"  System RAM         : {diag.hardware.memory_total_gb:.1f} GB Total ({diag.hardware.memory_available_gb:.1f} GB Available)")
    click.echo(f"  Partition Disk     : {diag.hardware.disk_total_gb:.1f} GB Total ({diag.hardware.disk_free_gb:.1f} GB Free)")
    
    click.secho("\n--- CUDA / GPU Status ---", fg="yellow", bold=True)
    if diag.cuda.gpu_available:
        click.secho(f"  Status             : AVAILABLE ({diag.cuda.gpu_count} GPU(s))", fg="green", bold=True)
        for i, dev in enumerate(diag.cuda.device_names):
            click.echo(f"  Device {i}           : {dev}")
        if diag.cuda.cuda_version:
            click.echo(f"  CUDA Version       : {diag.cuda.cuda_version}")
    else:
        click.secho("  Status             : NOT AVAILABLE (CPU-Only Host)", fg="yellow", bold=True)
        click.echo(f"  Note               : {diag.cuda.note}")

    click.secho("\n--- Key Dependencies ---", fg="yellow", bold=True)
    for pkg, ver in diag.package_versions.items():
        if ver:
            click.echo(f"  {pkg:<18}: {ver}")
        else:
            click.secho(f"  {pkg:<18}: NOT INSTALLED", fg="red")

    click.secho("\n--- Compute Recommendation ---", fg="cyan", bold=True)
    click.echo(f"  {diag.training_recommendation}\n")


@main.command(name="init-dirs")
@click.option("--config-path", "-c", type=click.Path(exists=True), help="Path to config YAML.")
def init_dirs_cmd(config_path: str | None) -> None:
    """Initialize all canonical project directories and placeholders."""
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        click.secho(f"Error loading configuration: {exc}", fg="red", err=True)
        sys.exit(1)

    root = cfg.paths.project_root
    dirs_to_create = [
        cfg.paths.raw_dir,
        cfg.paths.raw_dir / "mmotu",
        cfg.paths.raw_dir / "ovatus_v1",
        cfg.paths.interim_dir,
        cfg.paths.interim_dir / "artifact_masks",
        cfg.paths.interim_dir / "near_duplicate_clusters",
        cfg.paths.processed_dir,
        cfg.paths.processed_dir / "arm_a_raw",
        cfg.paths.processed_dir / "arm_b_inpainted",
        cfg.paths.processed_dir / "arm_c_masked",
        cfg.paths.manifests_dir,
        cfg.paths.outputs_dir,
        cfg.paths.outputs_dir / "checkpoints",
        cfg.paths.outputs_dir / "figures",
        cfg.paths.outputs_dir / "metrics",
        cfg.paths.outputs_dir / "logs",
        cfg.paths.logs_dir,
    ]

    click.echo(f"Initializing project directories under: {root}")
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        click.echo(f"  [✓] {d}")

    click.secho("\nAll directories successfully initialized.", fg="green", bold=True)


@main.command(name="validate-config")
@click.argument("config_path", type=click.Path(exists=True))
def validate_config_cmd(config_path: str) -> None:
    """Validate a YAML configuration file against Pydantic schema."""
    try:
        cfg = load_config(config_path)
        click.secho(f"\nConfiguration file '{config_path}' is VALID.", fg="green", bold=True)
        click.echo(f"  Project Name        : {cfg.project.name} v{cfg.project.version}")
        click.echo(f"  Default Backbone    : {cfg.modeling.default_backbone}")
        click.echo(f"  Preprocessing Arms  : {', '.join(cfg.preprocessing.arms)}")
        click.echo(f"  Batch Size / Epochs : {cfg.training.batch_size} / {cfg.training.max_epochs}")
        click.echo(f"  Seeds               : {cfg.evaluation.seeds}\n")
    except Exception as exc:
        click.secho(f"\nConfiguration validation FAILED: {exc}", fg="red", bold=True, err=True)
        sys.exit(1)


@main.command(name="audit-data")
@click.option("--data-dir", "-d", required=True, type=click.Path(), help="Path to directory containing images.")
@click.option("--output-dir", "-o", type=click.Path(), default="outputs/metrics", help="Directory to save report & inventory.")
@click.option("--dataset-id", default="MMOTU_2D", help="Dataset identifier (MMOTU_2D, OvaTUS_V1).")
@click.option("--hamming-threshold", default=4, type=int, help="dHash threshold for near-duplicates.")
def audit_data_cmd(data_dir: str, output_dir: str, dataset_id: str, hamming_threshold: int) -> None:
    """Scan and audit an image directory, detecting duplicates and verifying integrity."""
    target_path = Path(data_dir)
    if not target_path.exists():
        click.secho(
            f"Error: Target data directory does not exist: {data_dir}\n"
            "Action: Please download the dataset and place files in the target directory before auditing.",
            fg="red",
            bold=True,
            err=True,
        )
        sys.exit(1)

    try:
        summary, records = audit_image_dataset(
            target_dir=target_path,
            output_dir=output_dir,
            dataset_id=dataset_id,
            hamming_threshold=hamming_threshold,
        )

        click.secho("\n=======================================================", fg="cyan", bold=True)
        click.secho(f"         DATASET AUDIT REPORT: {dataset_id}           ", fg="cyan", bold=True)
        click.secho("=======================================================", fg="cyan", bold=True)
        click.echo(f"  Target Directory      : {summary.target_directory}")
        click.echo(f"  Total Files Scanned   : {summary.total_files_scanned}")
        click.echo(f"  Valid Images Found    : {summary.valid_images_found}")
        click.echo(f"  Corrupt/Unreadable    : {summary.corrupt_or_unreadable_files}")
        click.echo(f"  Unique SHA-256 Hashes : {summary.unique_sha256_count}")
        click.echo(f"  Exact Duplicate Pairs : {summary.exact_duplicate_clusters_count}")
        click.echo(f"  Near Duplicate Pairs  : {summary.near_duplicate_clusters_count}")
        click.echo(f"  Patient IDs Present   : {'YES' if summary.patient_ids_available else 'NO (Frame Leakage Risk)'}")
        click.echo(f"  Audit Status          : {summary.status}")

        if output_dir:
            click.secho(f"\nAudit report JSON and inventory CSV saved to: {output_dir}", fg="green")
        click.echo()
    except Exception as exc:
        click.secho(f"Data audit failed with error: {exc}", fg="red", bold=True, err=True)
        sys.exit(1)


@main.command(name="generate-manifest-template")
@click.option("--output", "-o", default="data/manifests/template.csv", help="Output path for template CSV.")
def generate_template_cmd(output: str) -> None:
    """Generate a blank image manifest template CSV."""
    try:
        generate_manifest_template_csv(output)
        click.secho(f"Manifest template generated at: {output}", fg="green", bold=True)
    except Exception as exc:
        click.secho(f"Failed to generate template: {exc}", fg="red", bold=True, err=True)
        sys.exit(1)


@main.command(name="generate-synthetic-benchmark")
@click.option("--output-dir", "-o", default="data/interim/synthetic_benchmark", help="Directory to export cases.")
@click.option("--num-cases", "-n", default=5, type=int, help="Number of benchmark cases to generate.")
@click.option("--seed", "-s", default=42, type=int, help="Random seed for reproducibility.")
@click.option("--metrics-out", "-m", default="outputs/metrics/synthetic_benchmark_metrics.json", help="Path for JSON metrics output.")
def generate_synthetic_cmd(output_dir: str, num_cases: int, seed: int, metrics_out: str) -> None:
    """Generate controlled synthetic caliper benchmark cases and compute integrity metrics."""
    import numpy as np

    click.secho("\n=======================================================", fg="cyan", bold=True)
    click.secho("    CONTROLLED SYNTHETIC-ARTIFACT BENCHMARK SUITE     ", fg="cyan", bold=True)
    click.secho("=======================================================", fg="cyan", bold=True)
    click.echo(f"  Cases to Generate : {num_cases}")
    click.echo(f"  Random Seed       : {seed}")
    click.echo(f"  Output Directory  : {output_dir}")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    metrics_p = Path(metrics_out)
    metrics_p.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    case_reports = []

    for i in range(1, num_cases + 1):
        case_id = f"SYNTH_CASE_{i:04d}"
        case_seed = seed + i * 17

        # Generate a simulated 2D ultrasound patch with acoustic speckle and a central lesion
        bg = rng.rayleigh(scale=35.0, size=(256, 256))
        yy, xx = np.mgrid[:256, :256]
        lesion_mask = ((xx - 128) ** 2 + (yy - 128) ** 2) < 45 ** 2
        bg[lesion_mask] = rng.rayleigh(scale=10.0, size=lesion_mask.sum())
        clean_patch = np.clip(bg, 0, 255).astype(np.uint8)

        cfg = SyntheticArtifactConfig(seed=case_seed, caliper_size=14, line_thickness=1)
        res = inject_synthetic_artifacts(clean_patch, cfg)
        saved = export_synthetic_case(res, out_p, case_id)

        # Baseline: Arm C simulated zero-masking on the synthetic artifact
        masked_patch = np.where(res.artifact_mask == 1, 0, res.corrupted_image)
        report = evaluate_reconstruction_integrity(res.clean_reference, masked_patch, res.artifact_mask)

        case_reports.append({
            "case_id": case_id,
            "export_paths": {k: str(v) for k, v in saved.items()},
            "arm_c_zero_masking_metrics": report.model_dump(),
        })
        click.echo(f"  [✓] Generated {case_id} (Mask ratio: {report.mask_pixel_ratio*100:.2f}%, Exterior PSNR: {report.exterior_psnr} dB)")

    with open(metrics_p, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_name": "Controlled Synthetic Caliper Injection",
            "timestamp_utc": res.provenance["timestamp_utc"],
            "total_cases": num_cases,
            "cases": case_reports,
        }, f, indent=2)

    click.secho(f"\nBenchmark dataset saved to: {out_p}", fg="green", bold=True)
    click.secho(f"Evaluation metrics JSON saved to: {metrics_p}", fg="green")
    click.secho("NOTE: Synthetic test cases are software benchmarks and do not replace clinical trials.\n", fg="yellow")


@main.command(name="ingest-guide")
def ingest_guide_cmd() -> None:
    """Display authoritative dataset acquisition guidelines and safety protocols."""
    click.secho("\n=======================================================", fg="cyan", bold=True)
    click.secho("        OVARYVISION DATASET INGESTION GUIDELINES      ", fg="cyan", bold=True)
    click.secho("=======================================================", fg="cyan", bold=True)
    click.echo("  1. MMOTU (Internal Benchmark):")
    click.echo("     - Authoritative Zenodo Record: https://zenodo.org/records/21128657")
    click.echo("     - Figshare Record            : https://figshare.com/articles/dataset/_zip/25058690")
    click.echo("     - Target Directory           : data/raw/mmotu/images/")
    click.echo("     - Tumor Masks Directory      : data/raw/mmotu/tumor_masks/")
    click.echo("     - Ingestion Helper Script    : python3 scripts/ingest_mmotu.py --guide")
    click.echo("\n  2. OvaTUS-V1 (External Benchmark):")
    click.echo("     - Source Institution         : NHOG Hanoi / SigM Lab HUST")
    click.echo("     - Access Policy              : Restricted / Clinical Research Agreement")
    click.echo("     - Target Schema              : 6 IOTA Morphological Categories")
    click.echo("     - Binary CAD Status          : BLOCKED pending surgical histopathology records")
    click.echo("\n  Safety Rule: No automated background downloading permitted without explicit user instruction.\n")


@main.command(name="generate-splits")
@click.option("--manifest", "-m", required=True, help="Path to input image manifest CSV.")
@click.option("--output-dir", "-o", default="data/manifests", help="Directory where split files are written.")
@click.option("--seeds", "-s", default="42,1337,2026", help="Comma-separated random seeds.")
@click.option("--train-ratio", default=0.70, type=float, help="Proportion of training cohort.")
@click.option("--val-ratio", default=0.15, type=float, help="Proportion of validation cohort.")
@click.option("--test-ratio", default=0.15, type=float, help="Proportion of internal test cohort.")
def generate_splits_cmd(
    manifest: str,
    output_dir: str,
    seeds: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Generate leak-resistant group-stratified train/val/test splits."""
    try:
        ratio = SplitRatio(train=train_ratio, val=val_ratio, test=test_ratio)
        seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
        records = load_image_manifest(manifest)

        click.secho("\n=======================================================", fg="cyan", bold=True)
        click.secho("       OVARYVISION GROUP-STRATIFIED SPLIT ENGINE       ", fg="cyan", bold=True)
        click.secho("=======================================================", fg="cyan", bold=True)
        click.echo(f"  Records Loaded    : {len(records)}")
        click.echo(f"  Target Proportions: Train={ratio.train:.2f}, Val={ratio.val:.2f}, Test={ratio.test:.2f}")
        click.echo(f"  Seeds Configured  : {seed_list}")

        results = generate_and_save_splits(records, output_dir=output_dir, seeds=seed_list, ratio=ratio)
        click.secho(f"\n[✓] Successfully generated and verified {len(results)} leak-free split manifests:", fg="green", bold=True)
        for seed, path in results.items():
            click.echo(f"  - Seed {seed:4d}: {path}")
        click.echo()
    except Exception as exc:
        click.secho(f"Failed to generate splits: {exc}", fg="red", bold=True, err=True)
        sys.exit(1)


@main.command(name="run-preprocessing")
@click.option("--manifest", "-m", required=True, help="Path to image manifest CSV/JSON.")
@click.option("--output-base-dir", "-o", default="data/processed", help="Base output directory.")
@click.option(
    "--arm",
    "-a",
    default="all",
    type=click.Choice(["all", "arm_a_raw", "arm_b_inpainted", "arm_c_masked"]),
    help="Target preprocessing arm or 'all'.",
)
@click.option("--dilation-radius", default=2, type=int, help="Mask dilation radius in pixels.")
@click.option("--fill-value", default=0, type=int, help="Arm C neutral fill value.")
@click.option("--limit", default=0, type=int, help="Optional record limit for smoke testing.")
def run_preprocessing_cmd(
    manifest: str,
    output_base_dir: str,
    arm: str,
    dilation_radius: int,
    fill_value: int,
    limit: int,
) -> None:
    """Preprocess ultrasound images across Arms A, B, and C."""
    try:
        from scripts.run_preprocessing import run_batch_preprocessing

        target_arms = ["arm_a_raw", "arm_b_inpainted", "arm_c_masked"] if arm == "all" else [arm]
        run_batch_preprocessing(
            manifest_path=Path(manifest),
            output_base_dir=Path(output_base_dir),
            arms=target_arms,
            dilation_radius=dilation_radius,
            fill_value=fill_value,
            limit=limit,
        )
        click.secho("\n[✓] Image preprocessing completed successfully.", fg="green", bold=True)
    except Exception as exc:
        click.secho(f"Preprocessing failed: {exc}", fg="red", bold=True, err=True)
        sys.exit(1)


@main.command(name="evaluate-calibration")
@click.option("--val-logits-file", required=True, help="CSV/JSON with validation logits and true labels.")
@click.option("--test-logits-file", required=True, help="CSV/JSON with test logits and true labels.")
@click.option("--num-bins", default=10, type=int, help="Number of calibration bins.")
@click.option("--output", "-o", default="outputs/metrics/calibration_report.json", help="Output JSON path.")
def evaluate_calibration_cmd(
    val_logits_file: str,
    test_logits_file: str,
    num_bins: int,
    output: str,
) -> None:
    """Fit validation Temperature Scaling and evaluate test calibration (ECE/MCE/Brier)."""
    import csv
    import numpy as np

    try:
        def load_logits(file_path: str) -> tuple[np.ndarray, np.ndarray]:
            logits, labels = [], []
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Look for binary_logit or raw_logit_1
                    z = float(row.get("binary_logit") or row.get("raw_logit_1") or row.get("logit", 0.0))
                    y = int(row.get("true_label") or row.get("diagnostic_label", 0))
                    logits.append(z)
                    labels.append(y)
            return np.array(logits), np.array(labels)

        val_z, val_y = load_logits(val_logits_file)
        test_z, test_y = load_logits(test_logits_file)

        report = evaluate_probability_calibration(
            val_logits=val_z,
            val_labels=val_y,
            test_logits=test_z,
            test_labels=test_y,
            num_bins=num_bins,
        )

        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        click.secho(f"\n[✓] Calibration report saved to: {out_p}", fg="green", bold=True)
        click.echo(f"  T* = {report.optimal_temperature:.4f} | ECE: {report.uncalibrated_ece:.4f} -> {report.calibrated_ece:.4f}")
    except Exception as exc:
        click.secho(f"Calibration evaluation failed: {exc}", fg="red", bold=True, err=True)
        sys.exit(1)


@main.command(name="evaluate-statistics")
@click.option("--arm-c-preds", required=True, help="CSV containing predictions for Arm C (Masked).")
@click.option("--arm-b-preds", required=True, help="CSV containing predictions for Arm B (Inpainted).")
@click.option("--primary-delta", default=-0.02, type=float, help="Primary non-inferiority margin delta.")
@click.option("--num-bootstrap", default=1000, type=int, help="Number of bootstrap resamples.")
@click.option("--output", "-o", default="outputs/metrics/statistical_hypothesis_report.json", help="Output JSON path.")
def evaluate_statistics_cmd(
    arm_c_preds: str,
    arm_b_preds: str,
    primary_delta: float,
    num_bootstrap: int,
    output: str,
) -> None:
    """Run paired DeLong test, non-inferiority margin test, bootstrap CIs, and McNemar's test."""
    import csv
    import numpy as np

    try:
        def load_preds(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
            probs, labels, ids = [], [], []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    p = float(row.get("calibrated_prob") or row.get("uncalibrated_prob") or row.get("prob", 0.5))
                    y = int(row.get("true_label") or row.get("diagnostic_label", 0))
                    img_id = row.get("image_id", "")
                    probs.append(p)
                    labels.append(y)
                    ids.append(img_id)
            return np.array(probs), np.array(labels), ids

        c_probs, c_labels, c_ids = load_preds(arm_c_preds)
        b_probs, b_labels, b_ids = load_preds(arm_b_preds)

        if len(c_probs) != len(b_probs):
            raise ValueError(f"Sample size mismatch: Arm C ({len(c_probs)}) vs Arm B ({len(b_probs)})")

        report = generate_comprehensive_statistical_report(
            scores_arm_c=c_probs,
            scores_arm_b=b_probs,
            labels=c_labels,
            primary_delta=primary_delta,
            num_bootstrap=num_bootstrap,
        )

        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        click.secho(f"\n[✓] Statistical hypothesis testing report saved to: {out_p}", fg="green", bold=True)
        click.echo(f"  Observed AUC Diff (C - B): {report.delong_test.auc_difference:+.4f}")
        click.echo(f"  Non-Inferiority Declared : {report.non_inferiority_primary.non_inferiority_declared} (p={report.non_inferiority_primary.p_value:.4f})")
    except Exception as exc:
        click.secho(f"Statistical evaluation failed: {exc}", fg="red", bold=True, err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

