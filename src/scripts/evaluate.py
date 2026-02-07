import torch
import numpy as np
import os
import re

from src.analysis.CurveDetector import CurveDetector
from src.analysis.Curve import Curve
from sklearn.cluster import KMeans
from src.models.auto_encoder import AutoEncoder
from src.models.VAE import VAE

# ==================== CONFIGURATION ====================
# Paths
TELEMETRY_PATH = "data/2025-main/Australian Grand Prix/Qualifying/ALB/4_tel.json"
CORNERS_PATH = "data/2025-main/Australian Grand Prix/Race/corners.json"
ENCODER_WEIGHTS_PATH = "src/models/weights/VAE_32z_weights.pth"
DATASET_PATH = "data/dataset/normalized_dataset_2024_2025_WITH_WET.npz"


# Model configuration
LATENT_DIM = 32
NUM_CLUSTERS = 4
USE_VAE = True  # Set to True to use VAE, False for standard AutoEncoder

# Feature configuration (same as dataset_normalization.py)
PADDING_VALUE = -1000.0
MAX_SAMPLES_PER_CURVE = 50  # Number of samples per feature
TARGET_COMPOUNDS = ['HARD', 'INTERMEDIATE', 'WET', 'MEDIUM', 'SOFT']


def load_pretrained_model(input_dim: int, latent_dim: int, weights_path: str, use_vae: bool = USE_VAE):
    """
    Load a pretrained model (AutoEncoder or VAE).
    
    Args:
        input_dim: Dimension of input features
        latent_dim: Dimension of latent space
        weights_path: Path to encoder weights
        use_vae: If True, load VAE; otherwise load AutoEncoder
        
    Returns:
        Loaded model in eval mode
    """
    if use_vae:
        model = VAE(input_dim, latent_dim=latent_dim)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        model = AutoEncoder(input_dim, latent_dim=latent_dim)
        model.encoder.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    print(f"Model loaded from {weights_path}")
    return model


def load_normalization_stats(dataset_path: str) -> tuple:
    """
    Load normalization statistics (mean, std) from the dataset file.
    
    Args:
        dataset_path: Path to the normalized dataset .npz file
        
    Returns:
        Tuple of (mean, std, columns, input_dim)
    """
    dataset = np.load(dataset_path, allow_pickle=True)
    mean = dataset["mean"]
    std = dataset["std"]
    columns = dataset["columns"]
    input_dim = dataset["data"].shape[1]
    
    print(f"Loaded normalization stats: {len(mean)} features")
    return mean, std, columns, input_dim


def detect_curves(telemetry_path: str, corners_path: str) -> list:
    """
    Detect curves from telemetry data.
    
    Args:
        telemetry_path: Path to telemetry JSON file
        corners_path: Path to corners JSON file
        
    Returns:
        List of Curve objects
    """
    detector = CurveDetector(telemetry_path, corners_path)
    curves = detector.calcolo_curve()
    print(f"Detected {len(curves)} curves")
    return curves


def pad_or_truncate(arr, target_length: int, padding_value: float = PADDING_VALUE) -> list:
    """
    Pad or truncate an array to a target length.
    """
    # Convert numpy arrays to list
    if hasattr(arr, 'tolist'):
        arr = arr.tolist()
    elif not isinstance(arr, list):
        arr = list(arr)
    
    if len(arr) >= target_length:
        return arr[:target_length]
    else:
        return arr + [padding_value] * (target_length - len(arr))


def curve_to_feature_vector(curve: Curve, columns: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple:
    """
    Convert a Curve object to a normalized feature vector matching the dataset format.
    
    Layout expected (based on from_norm_data):
      0: life
      1:51 speed (50 colonne)
      51:101 rpm (50 colonne)
      101:151 throttle (50 colonne)
      151:201 brake (50 colonne)
      201:251 acc_x (50 colonne)
      251:301 acc_y (50 colonne)
      301:351 acc_z (50 colonne)
      351: Compound_HARD
      352: Compound_INTERMEDIATE
      353: Compound_WET
      354: Compound_MEDIUM
      355: Compound_SOFT
    
    Args:
        curve: Curve object from CurveDetector
        columns: Column names from the dataset
        mean: Mean values for normalization
        std: Std values for normalization
        
    Returns:
        Tuple of (feature_vector, mask) as numpy arrays
    """
    n_samples = MAX_SAMPLES_PER_CURVE
    
    # Prepare raw features with padding
    life = curve.life if hasattr(curve, 'life') else 0
    speed = pad_or_truncate(curve.speed, n_samples)
    rpm = pad_or_truncate(curve.rpm, n_samples)
    throttle = pad_or_truncate(curve.throttle, n_samples)
    brake = pad_or_truncate(curve.brake, n_samples)
    acc_x = pad_or_truncate(curve.acc_x, n_samples)
    acc_y = pad_or_truncate(curve.acc_y, n_samples)
    acc_z = pad_or_truncate(curve.acc_z, n_samples)
    
    # Compound one-hot encoding
    compound_hard = 1.0 if curve.compound == "HARD" else 0.0
    compound_inter = 1.0 if curve.compound == "INTERMEDIATE" else 0.0
    compound_wet = 1.0 if curve.compound == "WET" else 0.0
    compound_medium = 1.0 if curve.compound == "MEDIUM" else 0.0
    compound_soft = 1.0 if curve.compound == "SOFT" else 0.0
    
    # Build raw feature vector
    raw_features = [life] + speed + rpm + throttle + brake + acc_x + acc_y + acc_z + \
                   [compound_hard, compound_inter, compound_wet, compound_medium, compound_soft]
    raw_features = np.array(raw_features, dtype=np.float64)
    
    # Build mask (1 for valid data, 0 for padding)
    mask = np.ones(len(raw_features), dtype=np.float64)
    mask[0] = 1  # life is always valid
    
    for i, val in enumerate(raw_features):
        if val == PADDING_VALUE:
            mask[i] = 0
    
    # Normalize using mean and std (Z-score normalization)
    # Replace padding with NaN for normalization, then back to 0
    normalized = np.copy(raw_features)
    for i in range(len(normalized)):
        if raw_features[i] != PADDING_VALUE and std[i] != 0:
            normalized[i] = (raw_features[i] - mean[i]) / std[i]
        elif raw_features[i] == PADDING_VALUE:
            normalized[i] = 0.0  # Padding becomes 0 after normalization
    
    return normalized, mask


def get_cluster_centroids_from_dataset(dataset_path: str, model: AutoEncoder, n_clusters: int = NUM_CLUSTERS) -> KMeans:
    """
    Train KMeans on the full dataset latent space to get cluster centroids.
    This is needed to assign new samples to existing clusters.
    
    Args:
        dataset_path: Path to the normalized dataset
        model: Trained AutoEncoder
        n_clusters: Number of clusters
        
    Returns:
        Fitted KMeans model
    """
    print("Loading dataset for clustering reference...")
    dataset = np.load(dataset_path, allow_pickle=True)
    data = dataset["data"]
    
    # Encode all data
    print(f"Encoding {len(data)} samples for clustering...")
    with torch.no_grad():
        model.eval()
        data_tensor = torch.tensor(data, dtype=torch.float32).to(next(model.parameters()).device)
        # Use get_latent for VAE (returns mu), encode for AutoEncoder
        if isinstance(model, VAE):
            latent_space = model.get_latent(data_tensor).cpu().numpy()
        else:
            latent_space = model.encode(data_tensor).cpu().numpy()
    
    # Fit KMeans
    print(f"Fitting KMeans with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(latent_space)
    
    return kmeans


def evaluate_curves(curves: list, model: AutoEncoder, kmeans: KMeans, 
                   columns: np.ndarray, mean: np.ndarray, std: np.ndarray) -> list:
    """
    Evaluate a list of curves and assign them to clusters.
    
    Args:
        curves: List of Curve objects
        model: Trained AutoEncoder
        kmeans: Fitted KMeans model
        columns: Column names
        mean: Mean for normalization
        std: Std for normalization
        
    Returns:
        List of (curve, cluster_id, latent_vector) tuples
    """
    results = []
    
    with torch.no_grad():
        model.eval()
        
        for i, curve in enumerate(curves):
            # Convert curve to feature vector
            feature_vector, mask = curve_to_feature_vector(curve, columns, mean, std)
            
            # Encode
            input_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            input_tensor = input_tensor.to(next(model.parameters()).device)
            
            # Use get_latent for VAE (returns mu), encode for AutoEncoder
            if isinstance(model, VAE):
                latent_vector = model.get_latent(input_tensor)
            else:
                latent_vector = model.encode(input_tensor)
            latent_np = latent_vector.squeeze(0).cpu().numpy()
            
            # Predict cluster
            cluster_id = kmeans.predict(latent_np.reshape(1, -1))[0]
            
            results.append({
                'curve': curve,
                'cluster_id': cluster_id,
                'latent_vector': latent_np,
                'corner_id': curve.corner_id if hasattr(curve, 'corner_id') else i
            })
    
    return results


def print_results(results: list):
    """
    Print evaluation results in a formatted way.
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    cluster_names = {
        0: "Cluster 0",
        1: "Cluster 1", 
        2: "Cluster 2",
        3: "Cluster 3"
    }
    
    cluster_counts = {0: 0, 1: 0, 2: 0, 3:0}
    
    for result in results:
        cluster_id = result['cluster_id']
        corner_id = result['corner_id']
        compound = result['curve'].compound
        
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        print(f"Corner {corner_id:2d} | Compound: {compound:12s} | Cluster: {cluster_id}")
    
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    
    total = len(results)
    for cluster_id, count in sorted(cluster_counts.items()):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"Cluster {cluster_id}: {count:3d} curves ({percentage:5.1f}%)")
    
    # Determine dominant driving style
    dominant_cluster = max(cluster_counts, key=cluster_counts.get)
    print(f"\nDominant style for this lap: Cluster {dominant_cluster}")
    
    print("=" * 60)


def main():
    """
    Main execution function.
    """
    print("=" * 60)
    print("Telemetry Evaluation - Driving Style Cluster Assignment")
    print("=" * 60)
    
    # Step 1: Load normalization statistics
    print("\n[1/5] Loading normalization statistics...")
    mean, std, columns, input_dim = load_normalization_stats(DATASET_PATH)
    
    # Step 2: Load pretrained model
    print("\n[2/5] Loading pretrained autoencoder...")
    model = load_pretrained_model(input_dim, LATENT_DIM, ENCODER_WEIGHTS_PATH)
    
    # Step 3: Detect curves from telemetry
    print("\n[3/5] Detecting curves from telemetry...")
    curves = detect_curves(TELEMETRY_PATH, CORNERS_PATH)
    
    if len(curves) == 0:
        print("ERROR: No curves detected!")
        return
    
    # Step 4: Get cluster centroids from dataset
    print("\n[4/5] Computing cluster centroids from dataset...")
    kmeans = get_cluster_centroids_from_dataset(DATASET_PATH, model)
    
    # Step 5: Evaluate curves
    print("\n[5/5] Evaluating curves...")
    results = evaluate_curves(curves, model, kmeans, columns, mean, std)
    
    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
