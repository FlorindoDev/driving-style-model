import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from dataclasses import dataclass

from src.analysis.Curve import Curve
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.models.auto_encoder import AutoEncoder
from src.models.VAE import VAE
from src.models.dataset_loader import load_and_split_dataset, load_dataset


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class TrainConfig:
    """Configuration for training and analysis."""
    
    # ----- Paths -----
    dataset_path: str = "data/dataset/normalized_dataset_2024_2025_WITH_WET.npz"
    load_weights_path: str = "src/models/weights/VAE_32z_weights.pth"
    save_weights_path: str = "src/models/weights/VAE_32z_weights.pth"
    
    # ----- Model -----
    use_vae: bool = True              # True = VAE, False = AutoEncoder
    latent_dim: int = 32
    
    # ----- Training -----
    train_model: bool = False         # True = train new model, False = load weights
    save_weights: bool = True
    learning_rate: float = 0.001
    weight_decay: float = 3e-4
    num_epochs: int = 50
    batch_size: int = 512
    train_ratio: float = 0.8
    
    # ----- Clustering -----
    num_samples: int = 1127865        # Number of samples to encode
    num_clusters: int = 4
    random_state: int = 0
    
    # ----- Visualization -----
    show_latent_space_2d: bool = False
    show_latent_space_3d: bool = False
    show_clusters_2d: bool = False
    show_clusters_3d: bool = False
    show_cluster_stats: bool = True


# =============================================================================
# DEFAULT CONFIG - MODIFY THIS
# =============================================================================
CONFIG = TrainConfig(
    # Paths
    dataset_path="data/dataset/normalized_dataset_2024_2025_WITH_WET.npz",
    load_weights_path="src/models/weights/VAE_32z_weights.pth",
    save_weights_path="src/models/weights/VAE_32z_weights.pth",
    
    # Model
    use_vae=True,
    latent_dim=32,
    
    # Training
    train_model=False,
    save_weights=True,
    learning_rate=0.001,
    weight_decay=3e-4,
    num_epochs=50,
    batch_size=512,
    train_ratio=0.8,
    
    # Clustering
    num_samples=1127865,
    num_clusters=4,
    random_state=0,
    
    # Visualization
    show_latent_space_2d=False,
    show_latent_space_3d=False,
    show_clusters_2d=False,
    show_clusters_3d=False,
    show_cluster_stats=True,
)


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================
def load_model(input_dim: int, config: TrainConfig):
    """
    Initialize the model and load pre-trained weights.
    
    Args:
        input_dim: Dimension of input features
        config: Training configuration
        
    Returns:
        Loaded model (VAE or AutoEncoder)
    """
    if config.use_vae:
        model = VAE(input_dim, latent_dim=config.latent_dim)
        model.load_state_dict(torch.load(config.load_weights_path, map_location="cpu"))
    else:
        model = AutoEncoder(input_dim, latent_dim=config.latent_dim)
        model.encoder.load_state_dict(torch.load(config.load_weights_path, map_location="cpu"))
    
    model.eval()
    print(f"Model loaded from {config.load_weights_path}")
    
    return model


def train_model(
    model: AutoEncoder,
    train_data: np.ndarray,
    train_mask: np.ndarray,
    val_data: np.ndarray,
    val_mask: np.ndarray,
    config: TrainConfig
) -> AutoEncoder:
    """
    Train the AutoEncoder model with validation for early stopping.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    print(f"Starting training for {config.num_epochs} epochs...")
    print(f"Learning rate: {config.learning_rate}, Weight decay: {config.weight_decay}")
    
    model.train_model(
        optimizer=optimizer,
        epochs=config.num_epochs,
        train_data=train_data,
        mask=train_mask,
        val_data=val_data,
        val_mask=val_mask,
        batch_size=config.batch_size
    )
    
    print("Training completed!")
    
    return model


def save_model_weights(model, save_path: str):
    """Save the model weights to a file."""
    if isinstance(model, VAE):
        torch.save(model.state_dict(), save_path)
    else:
        torch.save(model.encoder.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


def encode_data(
    model: AutoEncoder, 
    data: np.ndarray, 
    mask: list, 
    mean: list, 
    std: list, 
    num_samples: int
) -> tuple[list, np.ndarray]:
    """Encode data samples into the latent space."""
    curves = []
    latent_vectors = []
    
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(data[:num_samples]):
            sample_tensor = torch.tensor(
                np.atleast_2d(sample), dtype=torch.float32
            ).to(next(model.parameters()).device)
            
            if isinstance(model, VAE):
                latent_vector = model.get_latent(sample_tensor)
            else:
                latent_vector = model.encode(sample_tensor)
            latent_np = latent_vector.squeeze(0).cpu().numpy()
            
            curva = Curve.from_norm_data(sample, mask[i], mean, std, latent_np)
            curves.append(curva)
            latent_vectors.append(latent_np)
    
    latent_space = np.array(latent_vectors)
    print(f"Encoded {num_samples} samples into latent space: {latent_space.shape}")
    
    return curves, latent_space


def perform_clustering(
    latent_space: np.ndarray, 
    curves: list, 
    config: TrainConfig
) -> np.ndarray:
    """Perform K-Means clustering on the latent space."""
    kmeans = KMeans(n_clusters=config.num_clusters, random_state=config.random_state)
    clusters = kmeans.fit_predict(latent_space)

    for i, num_cluster in enumerate(clusters):
        curves[i].num_cluster = num_cluster

    print(f"Clustering completed: {config.num_clusters} clusters identified")
    
    return clusters


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_latent_space_2d(latent_space: np.ndarray, title: str = "Latent Space (PCA 2D)"):
    """Visualize latent space in 2D using PCA."""
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_space)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_latent_space_3d(latent_space: np.ndarray, title: str = "Latent Space (PCA 3D)"):
    """Visualize latent space in 3D using PCA."""
    pca = PCA(n_components=3)
    latent_3d = pca.fit_transform(latent_space)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2], alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    
    plt.show()


def plot_clusters_2d(latent_space: np.ndarray, clusters: np.ndarray):
    """Visualize clusters in 2D latent space."""
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_space)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clusters in Latent Space (PCA 2D)")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.show()


def plot_clusters_3d(latent_space: np.ndarray, clusters: np.ndarray):
    """Visualize clusters in 3D latent space."""
    pca = PCA(n_components=3)
    latent_3d = pca.fit_transform(latent_space)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    scatter = ax.scatter(
        latent_3d[:, 0],
        latent_3d[:, 1],
        latent_3d[:, 2],
        c=clusters,
        cmap='viridis',
        alpha=0.6
    )
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Clusters in Latent Space (PCA 3D)")
    
    plt.colorbar(scatter, label="Cluster")
    plt.show()


def print_cluster_stats(curves: list, num_clusters: int):
    """Print statistics for each cluster."""
    print("\n[7/7] Cluster Statistics...")
    
    for cluster_id in range(num_clusters):
        scores = [curve.pushing_score() for curve in curves if curve.num_cluster == cluster_id]
        scores = np.asarray(scores)
        
        print(f"Cluster {cluster_id}:")
        print(f"\t\tSize  : {scores.shape[0]}")
        if len(scores) > 0:
            print(f"\t\tMean  : {scores.mean():.4f}")
            print(f"\t\tStd   : {scores.std():.4f}")


# =============================================================================
# MAIN
# =============================================================================
def main(config: TrainConfig = CONFIG):
    """Main execution function for latent space analysis."""
    print("=" * 60)
    print("Latent Space Analysis - Driving Style Model")
    print("=" * 60)
    
    # Load and split dataset
    print("\n[1/7] Loading and splitting dataset...")
    dataset = load_and_split_dataset(config.dataset_path, train_ratio=config.train_ratio)
    
    # Per encoding/clustering usiamo tutti i dati originali
    full_data, full_mask, mean, std = load_dataset(config.dataset_path)
    
    # Initialize or load model
    if config.train_model:
        model_type = "VAE" if config.use_vae else "AutoEncoder"
        print(f"\n[2/7] Initializing and training {model_type} model...")
        
        if config.use_vae:
            model = VAE(dataset.train_data.shape[1], latent_dim=config.latent_dim)
        else:
            model = AutoEncoder(dataset.train_data.shape[1], latent_dim=config.latent_dim)
        
        model = train_model(
            model=model,
            train_data=dataset.train_data,
            train_mask=dataset.train_mask,
            val_data=dataset.val_data,
            val_mask=dataset.val_mask,
            config=config
        )
        
        if config.save_weights:
            save_model_weights(model, config.save_weights_path)
        
        model.eval()
    else:
        print("\n[2/7] Loading pre-trained model...")
        model = load_model(full_data.shape[1], config)
    
    # Encode data into latent space
    print("\n[3/7] Encoding data into latent space...")
    curves, latent_space = encode_data(
        model, full_data, full_mask, mean, std, config.num_samples
    )
    
    # Visualize latent space without clustering
    if config.show_latent_space_2d or config.show_latent_space_3d:
        print("\n[4/7] Visualizing latent space...")
        if config.show_latent_space_2d:
            plot_latent_space_2d(latent_space)
        if config.show_latent_space_3d:
            plot_latent_space_3d(latent_space)
    
    # Perform clustering
    print("\n[5/7] Performing K-Means clustering...")
    clusters = perform_clustering(latent_space, curves, config)
    
    # Visualize clusters
    if config.show_clusters_2d or config.show_clusters_3d:
        print("\n[6/7] Visualizing clusters...")
        if config.show_clusters_2d:
            plot_clusters_2d(latent_space, clusters)
        if config.show_clusters_3d:
            plot_clusters_3d(latent_space, clusters)
    
    # Print cluster statistics
    if config.show_cluster_stats:
        print_cluster_stats(curves, config.num_clusters)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()