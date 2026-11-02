"""Generate group-stratified internal train/validation split for MMOTU 2D segmentation.

Guarantees:
1. The 469 official test images (`val.txt`) remain completely untouched as `held_out_test`.
2. The 1,000 official training images (`train.txt`) are split into exactly 850 `internal_train` and 150 `internal_val`.
3. Whole dHash pseudo-patient clusters are preserved; zero cluster straddles internal_train and internal_val.
4. Stratification across morphological classes is enforced.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_segmentation_split(
    master_manifest_path: Path = Path("data/manifests/mmotu_master_manifest.csv"),
    train_txt_path: Path = Path("data/raw/mmotu/labels/train.txt"),
    val_txt_path: Path = Path("data/raw/mmotu/labels/val.txt"),
    output_path: Path = Path("data/manifests/mmotu_2d_segmentation_split.csv"),
    seed: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(master_manifest_path)
    df["image_id_str"] = df["image_id"].astype(str)

    with open(train_txt_path, "r", encoding="utf-8") as f:
        official_train_ids = set(line.strip().split()[0].replace(".JPG", "") for line in f if line.strip())

    with open(val_txt_path, "r", encoding="utf-8") as f:
        official_test_ids = set(line.strip().split()[0].replace(".JPG", "") for line in f if line.strip())

    assert len(official_train_ids) == 1000, f"Expected 1000 train IDs, got {len(official_train_ids)}"
    assert len(official_test_ids) == 469, f"Expected 469 test IDs, got {len(official_test_ids)}"
    assert len(official_train_ids.intersection(official_test_ids)) == 0, "Train and Val official splits intersect!"

    # 1. Assign held_out_test
    df["segmentation_split"] = "unknown"
    df.loc[df["image_id_str"].isin(official_test_ids), "segmentation_split"] = "held_out_test"

    # 2. Partition official train into internal_train (850) and internal_val (150)
    train_mask = df["image_id_str"].isin(official_train_ids)
    train_df = df[train_mask].copy()

    train_df["effective_cluster"] = train_df["pseudo_patient_cluster"].fillna("SINGLETON_" + train_df["image_id_str"])

    cluster_meta = train_df.groupby("effective_cluster").agg(
        n_images=("image_id", "count"),
        n_normal=("fine_grained_class", lambda s: (s == "NormalOvary").sum()),
        n_malignant=("diagnostic_label", lambda s: (s == 1).sum()),
    ).reset_index()

    cluster_meta["is_normal"] = cluster_meta["n_normal"] > 0
    cluster_meta["is_malignant"] = cluster_meta["n_malignant"] > 0

    def get_strat(row):
        if row["is_malignant"]:
            return "malignant"
        if row["is_normal"]:
            return "normal"
        return "benign_tumour"

    cluster_meta["strat"] = cluster_meta.apply(get_strat, axis=1)

    val_clusters = set()
    targets = {"normal": 27, "malignant": 6, "benign_tumour": 117}
    val_counts = {"normal": 0, "malignant": 0, "benign_tumour": 0}

    for strat, target_img in targets.items():
        sub = cluster_meta[cluster_meta["strat"] == strat].copy()
        shuffled = sub.sample(frac=1.0, random_state=seed)
        cur = 0
        for _, row in shuffled.iterrows():
            if cur + row["n_images"] <= target_img:
                val_clusters.add(row["effective_cluster"])
                cur += row["n_images"]
        val_counts[strat] = cur

    deficit = 150 - sum(val_counts.values())
    if deficit > 0:
        remaining = cluster_meta[~cluster_meta["effective_cluster"].isin(val_clusters) & (cluster_meta["n_images"] == 1)]
        fillers = remaining.sample(n=deficit, random_state=seed)
        for _, row in fillers.iterrows():
            val_clusters.add(row["effective_cluster"])
            val_counts[row["strat"]] += 1

    train_df["internal_role"] = train_df["effective_cluster"].apply(
        lambda c: "internal_val" if c in val_clusters else "internal_train"
    )

    # Validate split counts
    role_counts = train_df["internal_role"].value_counts().to_dict()
    assert role_counts.get("internal_train", 0) == 850, f"Expected 850 internal_train, got {role_counts.get('internal_train')}"
    assert role_counts.get("internal_val", 0) == 150, f"Expected 150 internal_val, got {role_counts.get('internal_val')}"

    # Validate zero cluster leakage
    c_train = set(train_df[train_df["internal_role"] == "internal_train"]["effective_cluster"])
    c_val = set(train_df[train_df["internal_role"] == "internal_val"]["effective_cluster"])
    leakage = c_train.intersection(c_val)
    assert len(leakage) == 0, f"Cluster leakage detected between internal_train and internal_val: {leakage}"

    # Map back to main dataframe
    internal_role_map = dict(zip(train_df["image_id_str"], train_df["internal_role"]))
    df.loc[train_mask, "segmentation_split"] = df.loc[train_mask, "image_id_str"].map(internal_role_map)

    # Save manifest
    output_cols = [
        "image_id",
        "relative_image_path",
        "tumor_mask_path",
        "diagnostic_label",
        "fine_grained_class",
        "pseudo_patient_cluster",
        "segmentation_split",
        "sha256_checksum",
    ]
    df[output_cols].to_csv(output_path, index=False)
    print(f"Saved segmentation split manifest to {output_path}")
    print("\n=== Split Distribution ===")
    print(df["segmentation_split"].value_counts())
    print("\n=== Class Distribution by Split Role ===")
    print(pd.crosstab(df["fine_grained_class"], df["segmentation_split"], margins=True))

    return df


if __name__ == "__main__":
    generate_segmentation_split()

