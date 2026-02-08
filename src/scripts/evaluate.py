import torch
import numpy as np
import os
from dataclasses import dataclass
from sklearn.cluster import KMeans

from src.analysis.dataset_normalization import (
    load_normalized_data,
    normalize_telemetry_json
)
from src.models.auto_encoder import AutoEncoder
from src.models.VAE import VAE
from src.models.dataset_loader import download_dataset_from_hf, download_raw_telemetry_from_hf


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class EvaluateConfig:
    """Configuration for telemetry evaluation."""
    
    # ----- Paths -----
    telemetry_path: str = "data/2025-main/Italian Grand Prix/Qualifying/LEC/1_tel.json"
    corners_path: str = "data/2025-main/Italian Grand Prix/Race/corners.json"
    weights_path: str = "src/models/weights/VAE_32z_weights.pth"
    dataset_path: str = "data/dataset/normalized_dataset_2024_2025.npz"
    centroids_path: str = "src/models/weights/kmeans_centroids.npy"
    
    # ----- Model -----
    use_vae: bool = True
    latent_dim: int = 32
    
    # ----- Hugging Face -----
    download_from_hf: bool = False  # True = download normalized dataset from HF
    download_raw_from_hf: bool = False  # True = download raw telemetry from HF
    raw_data_subfolder: str = "2025-main"  # Subfolder to download (2024-main or 2025-main)
    
    # ----- Clustering -----
    num_clusters: int = 4
    random_state: int = 0
    load_centroids: bool = True  # True = load centroids from file, False = fit KMeans


CONFIG = EvaluateConfig()


# =============================================================================
# FUNCTIONS
# =============================================================================
def load_model(input_dim: int, config: EvaluateConfig):
    """Load pretrained model."""
    if config.use_vae:
        model = VAE(input_dim, latent_dim=config.latent_dim)
        model.load_state_dict(torch.load(config.weights_path, map_location="cpu"))
    else:
        model = AutoEncoder(input_dim, latent_dim=config.latent_dim)
        model.encoder.load_state_dict(torch.load(config.weights_path, map_location="cpu"))
    
    model.eval()
    print(f"Model loaded from {config.weights_path}")
    return model


def load_kmeans_centroids(load_path: str, num_clusters: int, random_state: int = 0) -> KMeans:
    """
    Load KMeans centroids from a .npy file.
    
    Args:
        load_path: Path to the saved centroids
        num_clusters: Number of clusters
        random_state: Random state for KMeans
        
    Returns:
        KMeans object with pre-loaded centroids
    """
    centroids = np.load(load_path)
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=1)
    kmeans.cluster_centers_ = centroids
    kmeans._n_threads = 1
    print(f"KMeans centroids loaded from {load_path}")
    return kmeans


def fit_kmeans(data: np.ndarray, model, config: EvaluateConfig) -> KMeans:
    """Fit KMeans on dataset latent space."""
    print(f"Encoding {len(data)} samples...")
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32)
        if isinstance(model, VAE):
            latent_space = model.get_latent(data_tensor).cpu().numpy()
        else:
            latent_space = model.encode(data_tensor).cpu().numpy()
    
    print(f"Fitting KMeans ({config.num_clusters} clusters)...")
    kmeans = KMeans(n_clusters=config.num_clusters, random_state=config.random_state)
    kmeans.fit(latent_space)
    return kmeans


def predict_clusters(normalized_curves: list, model, kmeans: KMeans) -> list:
    """Predict cluster for each normalized curve."""
    results = []
    
    with torch.no_grad():
        for item in normalized_curves:
            curve = item['curve']
            feature_vector = item['normalized']
            
            # Encode
            input_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            if isinstance(model, VAE):
                latent = model.get_latent(input_tensor).squeeze(0).numpy()
            else:
                latent = model.encode(input_tensor).squeeze(0).numpy()
            
            # Predict
            cluster_id = kmeans.predict(latent.reshape(1, -1))[0]
            
            results.append({
                'curve': curve,
                'cluster_id': cluster_id,
                'corner_id': curve.corner_id
            })
    
    return results


def print_results(results: list, num_clusters: int):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    cluster_counts = {i: 0 for i in range(num_clusters)}
    
    for r in results:
        cluster_counts[r['cluster_id']] += 1
        print(f"Corner {r['corner_id']:2d} | {r['curve'].compound:12s} | Cluster {r['cluster_id']}")
    
    print("\n" + "-" * 60)
    total = len(results)
    for cid, count in sorted(cluster_counts.items()):
        pct = (count / total * 100) if total > 0 else 0
        print(f"Cluster {cid}: {count:3d} curves ({pct:5.1f}%)")
    
    dominant = max(cluster_counts, key=cluster_counts.get)
    print(f"\nDominant style: Cluster {dominant}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================
def main(config: EvaluateConfig = CONFIG):
    print("=" * 60)
    print("Telemetry Evaluation")
    print("=" * 60)
    
    # 0. Download raw telemetry if configured
    if config.download_raw_from_hf:
        print("\n[0/5] Downloading raw telemetry from Hugging Face...")
        download_raw_telemetry_from_hf(subfolder=config.raw_data_subfolder)
    
    # 1. Load dataset stats (download from HF if configured)
    print("\n[1/5] Loading dataset...")
    dataset_path = config.dataset_path
    if config.download_from_hf:
        print("Downloading dataset from Hugging Face...")
        dataset_path = download_dataset_from_hf(filename=config.dataset_path)
    data, _, mean, std, _ = load_normalized_data(dataset_path)
    
    # 2. Load model
    print("\n[2/5] Loading model...")
    model = load_model(data.shape[1], config)
    
    # 3. Normalize telemetry (uses normalize_telemetry_json!)
    print("\n[3/5] Processing telemetry...")
    normalized_curves = normalize_telemetry_json(
        config.telemetry_path,
        config.corners_path,
        config.dataset_path
    )
    print(f"Detected {len(normalized_curves)} curves")
    
    if len(normalized_curves) == 0:
        print("ERROR: No curves detected!")
        return
    
    # 4. Get KMeans (load or fit)
    print("\n[4/5] Clustering...")
    if config.load_centroids and os.path.exists(config.centroids_path):
        kmeans = load_kmeans_centroids(config.centroids_path, config.num_clusters, config.random_state)
    else:
        kmeans = fit_kmeans(data, model, config)
    
    results = predict_clusters(normalized_curves, model, kmeans)
    
    print("\n[5/5] Results:")
    print_results(results, config.num_clusters)


if __name__ == "__main__":
    main()
