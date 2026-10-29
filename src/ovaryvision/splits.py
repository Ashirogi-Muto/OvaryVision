"""Group-stratified train/val/test splitting with pseudo-patient cluster isolation and zero leakage."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from ovaryvision.logging_utils import get_logger
from ovaryvision.manifests import ImageRecord, SplitRecord, save_split_manifest, validate_manifest_leakage

logger = get_logger("splits")


class SplitRatio(BaseModel):
    """Proportion targets for internal train/val/test partitioning."""
    train: float = Field(0.70, ge=0.0, le=1.0)
    val: float = Field(0.15, ge=0.0, le=1.0)
    test: float = Field(0.15, ge=0.0, le=1.0)

    def model_post_init(self, __context: Any) -> None:
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0, atol=1e-4):
            raise ValueError(f"Split ratios must sum to 1.0, got train={self.train}, val={self.val}, test={self.test} (sum={total})")


class RoleClassCounts(BaseModel):
    """Count breakdown for a single split role."""
    total: int = 0
    benign: int = 0
    malignant: int = 0
    unlabeled: int = 0
    clusters: int = 0


class SplitSummary(BaseModel):
    """Summary metrics of a generated split partition."""
    seed: int
    train: RoleClassCounts = Field(default_factory=RoleClassCounts)
    val: RoleClassCounts = Field(default_factory=RoleClassCounts)
    test_internal: RoleClassCounts = Field(default_factory=RoleClassCounts)
    test_external: RoleClassCounts = Field(default_factory=RoleClassCounts)
    total_images: int = 0
    total_clusters: int = 0


def generate_group_stratified_split(
    records: List[ImageRecord],
    seed: int = 42,
    ratio: Optional[SplitRatio] = None,
    external_records: Optional[List[ImageRecord]] = None,
) -> List[SplitRecord]:
    """Generate a leak-resistant, group-stratified train/val/test split.

    Groups are defined by `pseudo_patient_cluster` (or `image_id` if cluster is absent/None).
    All images within the same cluster are strictly assigned to the SAME split role.
    Stratification is performed across diagnostic labels (0: Benign, 1: Malignant).

    Args:
        records: List of internal image records.
        seed: Random seed for deterministic group shuffling.
        ratio: Target split ratios (defaults to 70% train, 15% val, 15% test).
        external_records: Optional external cohort records (assigned to 'test_external').

    Returns:
        List of SplitRecord objects validated to be 100% leak-free.
    """
    if not records:
        raise ValueError("Cannot generate splits from an empty record list.")

    if ratio is None:
        ratio = SplitRatio(train=0.70, val=0.15, test=0.15)

    # 1. Group records by pseudo_patient_cluster (or image_id fallback)
    cluster_records: Dict[str, List[ImageRecord]] = defaultdict(list)
    for r in records:
        group_key = r.pseudo_patient_cluster if r.pseudo_patient_cluster else f"INDIVIDUAL_{r.image_id}"
        cluster_records[group_key].append(r)

    # 2. Determine majority diagnostic label and size for each cluster
    clusters_by_label: Dict[int, List[str]] = defaultdict(list)
    for cluster_id, recs in cluster_records.items():
        # Count labels in cluster
        label_counts: Dict[int, int] = defaultdict(int)
        for rec in recs:
            label_counts[rec.diagnostic_label] += 1
        
        # Determine dominant label
        majority_label = max(label_counts.keys(), key=lambda k: label_counts[k])
        clusters_by_label[majority_label].append(cluster_id)

    # 3. Deterministic partitioning per stratum
    rng = np.random.default_rng(seed)
    cluster_assignments: Dict[str, str] = {}

    target_roles = ["train", "val", "test_internal"]
    target_proportions = [ratio.train, ratio.val, ratio.test]

    for label, cluster_list in clusters_by_label.items():
        # Deterministically shuffle clusters for this label
        shuffled_clusters = list(cluster_list)
        # Sort first for cross-platform platform-independent ordering before shuffle
        shuffled_clusters.sort()
        rng.shuffle(shuffled_clusters)

        # Track image counts per role for this stratum
        total_stratum_images = sum(len(cluster_records[c]) for c in shuffled_clusters)
        target_counts = {
            role: total_stratum_images * prop
            for role, prop in zip(target_roles, target_proportions)
        }
        current_counts = {role: 0 for role in target_roles}

        # Greedy allocation to the role with the largest remaining deficit
        for c in shuffled_clusters:
            c_size = len(cluster_records[c])
            # Calculate deficit for each role: (target - current)
            deficits = {role: target_counts[role] - current_counts[role] for role in target_roles}
            # Pick the role with the maximum deficit (or train if tie)
            chosen_role = max(target_roles, key=lambda r: (deficits[r], -target_roles.index(r)))
            cluster_assignments[c] = chosen_role
            current_counts[chosen_role] += c_size

    # 4. Create SplitRecord list
    split_records: List[SplitRecord] = []
    for r in records:
        group_key = r.pseudo_patient_cluster if r.pseudo_patient_cluster else f"INDIVIDUAL_{r.image_id}"
        role = cluster_assignments[group_key]
        split_records.append(
            SplitRecord(
                image_id=r.image_id,
                split_role=role,
                split_seed=seed,
                pseudo_patient_cluster=r.pseudo_patient_cluster,
                diagnostic_label=r.diagnostic_label,
            )
        )

    # 5. Process external records if provided
    if external_records:
        for r in external_records:
            split_records.append(
                SplitRecord(
                    image_id=r.image_id,
                    split_role="test_external",
                    split_seed=seed,
                    pseudo_patient_cluster=r.pseudo_patient_cluster,
                    diagnostic_label=r.diagnostic_label,
                )
            )

    # 6. Strict two-tier leakage validation
    validate_manifest_leakage(split_records)
    logger.info(
        f"Generated split for seed {seed}: {len(split_records)} records across "
        f"{len(cluster_records)} clusters. Leakage validation passed."
    )

    return split_records


def compute_split_summary(split_records: List[SplitRecord]) -> SplitSummary:
    """Compute detailed summary counts for a split assignment."""
    if not split_records:
        raise ValueError("Cannot summarize empty split records.")

    seed = split_records[0].split_seed
    summary = SplitSummary(seed=seed, total_images=len(split_records))

    seen_clusters: Dict[str, set] = defaultdict(set)
    for r in split_records:
        role = r.split_role
        target_counts: RoleClassCounts = getattr(summary, role)
        target_counts.total += 1
        if r.diagnostic_label == 0:
            target_counts.benign += 1
        elif r.diagnostic_label == 1:
            target_counts.malignant += 1
        else:
            target_counts.unlabeled += 1

        cluster = r.pseudo_patient_cluster or f"INDIVIDUAL_{r.image_id}"
        seen_clusters[role].add(cluster)

    for role, cluster_set in seen_clusters.items():
        getattr(summary, role).clusters = len(cluster_set)

    all_clusters = set()
    for s in seen_clusters.values():
        all_clusters.update(s)
    summary.total_clusters = len(all_clusters)

    return summary


def generate_and_save_splits(
    records: List[ImageRecord],
    output_dir: Path | str,
    seeds: Optional[List[int]] = None,
    ratio: Optional[SplitRatio] = None,
    external_records: Optional[List[ImageRecord]] = None,
) -> Dict[int, Path]:
    """Generate and save split manifest CSVs across multiple seeds.

    Args:
        records: Image records to partition.
        output_dir: Directory where split manifests will be written.
        seeds: List of integer seeds (defaults to [42, 1337, 2026]).
        ratio: Target proportions.
        external_records: Optional external validation cohort records.

    Returns:
        Mapping of seed to output CSV path.
    """
    if seeds is None:
        seeds = [42, 1337, 2026]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: Dict[int, Path] = {}
    for seed in seeds:
        split_records = generate_group_stratified_split(
            records=records,
            seed=seed,
            ratio=ratio,
            external_records=external_records,
        )
        csv_file = out_path / f"mmotu_split_seed{seed}.csv"
        save_split_manifest(split_records, csv_file)
        results[seed] = csv_file
        logger.info(f"Saved split manifest for seed {seed} to {csv_file}")

    return results

