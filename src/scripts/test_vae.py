"""
Test VAE classification accuracy on labeled test data.

Loads test.csv, normalizes the data using stored dataset statistics,
encodes via VAE, predicts clusters via KMeans, and computes accuracy
against ground truth cluster labels.

Usage:
    python -m src.scripts.test_vae
"""

import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.analysis.dataset_normalization import (
    load_normalized_data,
    normalize_tire_life,
    normalize_sample,
    one_hot_encode_compound,
    create_padding_mask,
    PADDING_VALUE,
    COMPOUND_CATEGORIES,
)
from src.scripts.evaluate import load_model, load_kmeans_centroids, EvaluateConfig


# =============================================================================
# CONFIGURATION
# =============================================================================
TEST_CSV_PATH = "data/dataset/test.csv"
DATASET_NPZ_PATH = "data/dataset/normalized_dataset_2024_2025.npz"
WEIGHTS_PATH = "src/models/weights/VAE_32z_weights.pth"
CENTROIDS_PATH = "src/models/weights/kmeans_centroids.npy"
NUM_CLUSTERS = 4
LATENT_DIM = 32


# =============================================================================
# FUNCTIONS
# =============================================================================
def load_and_prepare_test_data(
    csv_path: str, 
    npz_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load test CSV, clean columns, normalize, and return data + labels + metadata.
    
    Returns:
        (normalized_data, masks, true_labels, metadata_df) as numpy arrays + DataFrame
    """
    # 1. Load CSV
    df = pd.read_csv(csv_path, sep=",", encoding="utf-8", decimal=".")
    print(f"Loaded {len(df)} samples from {csv_path}")
    
    # 2. Extract ground truth labels (last column)
    true_labels = df["cluster"].values
    df = df.drop("cluster", axis=1)
    
    # 3. Save metadata before dropping
    metadata_cols = ["GrandPrix", "Session", "Driver", "Lap", "CornerID", "Stint"]
    metadata_df = df[[c for c in metadata_cols if c in df.columns]].copy()
    df = df.drop(columns=[c for c in metadata_cols if c in df.columns])
    
    # 4. Drop extra signal columns (x, y, z, distance, time)
    extra_prefixes = ["x_", "y_", "z_", "distance_", "time_"]
    cols_to_drop = [c for c in df.columns if any(c.startswith(p) for p in extra_prefixes)]
    df = df.drop(columns=cols_to_drop)
    print(f"After dropping extra columns: {df.shape[1]} features remaining")
    
    # 5. Normalize TireLife per compound
    df["TireLife"] = df.apply(
        lambda row: normalize_tire_life(row["TireLife"], row["Compound"]), axis=1
    )
    
    # 6. One-hot encode Compound
    df = one_hot_encode_compound(df, COMPOUND_CATEGORIES)
    
    # 7. Z-score normalize using stored mean/std
    _, _, mean, std, columns = load_normalized_data(npz_path)
    
    # Verify column alignment
    df_cols = list(df.columns)
    npz_cols = list(columns)
    assert df_cols == npz_cols, (
        f"Column mismatch!\n"
        f"  CSV columns ({len(df_cols)}): {df_cols[:5]}...{df_cols[-5:]}\n"
        f"  NPZ columns ({len(npz_cols)}): {npz_cols[:5]}...{npz_cols[-5:]}"
    )
    
    # Normalize each sample
    normalized_data = []
    masks = []
    for _, row in df.iterrows():
        raw = row.values.astype(np.float64)
        norm, mask = normalize_sample(raw, mean, std, PADDING_VALUE)
        normalized_data.append(norm)
        masks.append(mask)
    
    return np.array(normalized_data), np.array(masks), true_labels, metadata_df


def compute_accuracy(predicted: np.ndarray, true_labels: np.ndarray, num_clusters: int):
    """Compute and print accuracy metrics."""
    total = len(true_labels)
    correct = np.sum(predicted == true_labels)
    accuracy = correct / total * 100
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    # Per-cluster accuracy
    print(f"\n{'Cluster':<10} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 40)
    for c in range(num_clusters):
        mask = true_labels == c
        cluster_total = mask.sum()
        if cluster_total > 0:
            cluster_correct = (predicted[mask] == c).sum()
            cluster_acc = cluster_correct / cluster_total * 100
            print(f"{c:<10} {cluster_correct:<10} {cluster_total:<10} {cluster_acc:.1f}%")
        else:
            print(f"{c:<10} {'—':<10} {0:<10} {'N/A':<10}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>8}", end="")
    for c in range(num_clusters):
        print(f"{'P'+str(c):>8}", end="")
    print()
    for true_c in range(num_clusters):
        print(f"{'T'+str(true_c):>8}", end="")
        for pred_c in range(num_clusters):
            count = ((true_labels == true_c) & (predicted == pred_c)).sum()
            print(f"{count:>8}", end="")
        print()
    
    print("=" * 60)
    return accuracy


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("VAE Test Accuracy")
    print("=" * 60)
    
    # 1. Load and normalize test data
    print("\n[1/4] Loading and normalizing test data...")
    normalized_data, masks, true_labels, metadata = load_and_prepare_test_data(
        TEST_CSV_PATH, DATASET_NPZ_PATH
    )
    print(f"Normalized data shape: {normalized_data.shape}")
    
    # 2. Load model
    print("\n[2/4] Loading VAE model...")
    config = EvaluateConfig(
        weights_path=WEIGHTS_PATH,
        latent_dim=LATENT_DIM,
        use_vae=True,
    )
    model = load_model(normalized_data.shape[1], config)
    
    # 3. Load KMeans centroids
    print("\n[3/4] Loading KMeans centroids...")
    kmeans = load_kmeans_centroids(CENTROIDS_PATH, NUM_CLUSTERS)
    
    # 4. Predict clusters
    print("\n[4/4] Predicting clusters...")
    with torch.no_grad():
        data_tensor = torch.tensor(normalized_data, dtype=torch.float32)
        latent = model.get_latent(data_tensor).cpu().numpy()
    
    predicted = kmeans.predict(latent)
    
    # 5. Per-row results
    print("\n" + "=" * 80)
    print("PER-SAMPLE PREDICTIONS")
    print("=" * 80)
    print(f"{'#':<4} {'GrandPrix':<28} {'Driver':<7} {'Corner':<8} {'True':<6} {'Pred':<6} {'Match'}")
    print("-" * 80)
    for i in range(len(true_labels)):
        gp = metadata.iloc[i]["GrandPrix"] if "GrandPrix" in metadata.columns else "?"
        driver = metadata.iloc[i]["Driver"] if "Driver" in metadata.columns else "?"
        corner = metadata.iloc[i]["CornerID"] if "CornerID" in metadata.columns else "?"
        match = "✓" if predicted[i] == true_labels[i] else "✗"
        print(f"{i+1:<4} {gp:<28} {driver:<7} {corner:<8} {true_labels[i]:<6} {predicted[i]:<6} {match}")
    print("=" * 80)
    
    # 6. Recap
    compute_accuracy(predicted, true_labels, NUM_CLUSTERS)


if __name__ == "__main__":
    main()
