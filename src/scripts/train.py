"""
Neural Model Latent Space Analysis

This script loads a trained autoencoder model, encodes driving data into a latent space,
performs dimensionality reduction using PCA, and visualizes clusters in both 2D and 3D.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from src.analysis.Curve import Curve
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.models.auto_encoder import AutoEncoder
from src.models.VAE import VAE
from src.models.dataset_loader import load_and_split_dataset, load_dataset

# Configuration constants
DATASET_PATH = "data\\dataset\\normalized_dataset_2024_2025_WITH_WET.npz"
ENCODER_WEIGHTS_PATH = "src\\models\\weights\\encoder5.pth"
SAVE_ENCODER_PATH = "src\\models\\weights\\encoder5.pth"  # Path for saving new trained weights

LATENT_DIM = 32
NUM_SAMPLES = 1127865 
NUM_CLUSTERS = 4
RANDOM_STATE = 0

# Training configuration
TRAIN_MODEL = False  # Set to True to train the model instead of loading weights
SAVE_WEIGHTS = True  # Set to True to save weights after training
USE_VAE = True  # Set to True to use VAE, False for standard AutoEncoder
LEARNING_RATE = 0.001
WEIGHT_DECAY = 3e-4
NUM_EPOCHS = 50
BATCH_SIZE = 512
TRAIN_RATIO = 0.8  # Percentuale dati per training (resto per validation)


def load_model(input_dim: int, latent_dim: int, weights_path: str, use_vae: bool = False):
    """
    Initialize the model and load pre-trained weights.
    
    Args:
        input_dim: Dimension of input features
        latent_dim: Dimension of latent space
        weights_path: Path to the saved weights
        use_vae: If True, load a VAE model; if False, load an AutoEncoder
        
    Returns:
        Loaded model (VAE or AutoEncoder)
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


def train_model(
    model: AutoEncoder,
    train_data: np.ndarray,
    train_mask: np.ndarray,
    val_data: np.ndarray,
    val_mask: np.ndarray,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int
) -> AutoEncoder:
    """
    Train the AutoEncoder model with validation for early stopping.
    
    Args:
        model: AutoEncoder model to train
        train_data: Training data
        train_mask: Mask for training data
        val_data: Validation data
        val_mask: Mask for validation data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        
    Returns:
        Trained AutoEncoder model
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    
    model.train_model(
        optimizer=optimizer,
        epochs=epochs,
        train_data=train_data,
        mask=train_mask,
        val_data=val_data,
        val_mask=val_mask,
        batch_size=batch_size
    )
    
    print("Training completed!")
    
    return model


def save_model_weights(model, save_path: str):
    """
    Save the model weights to a file.
    
    Args:
        model: AutoEncoder or VAE model
        save_path: Path where to save the weights
    """
    if isinstance(model, VAE):
        # Save entire VAE state (includes encoder, fc_mu, fc_logvar, decoder)
        torch.save(model.state_dict(), save_path)
    else:
        # Save only encoder for AutoEncoder
        torch.save(model.encoder.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


def encode_data(model: AutoEncoder, data: np.ndarray, mask: list, mean : list, std : list, num_samples: int) -> tuple[list, np.ndarray]:
    """
    Encode data samples into the latent space.
    
    Args:
        model: Trained AutoEncoder model
        data: Input data array
        mask: Mask array for padding values
        num_samples: Number of samples to encode
        
    Returns:
        Tuple of (curves list, latent_space array)
    """
    curves = []
    latent_vectors = []
    
    with torch.no_grad():
        model.eval()  # Ensure model is in eval mode
        for i, sample in enumerate(data[:num_samples]):
            # Convert to tensor, add batch dimension and move to device
            sample_tensor = torch.tensor(np.atleast_2d(sample), dtype=torch.float32).to(next(model.parameters()).device)
            
            # Use get_latent for VAE (returns mu), encode for AutoEncoder
            if isinstance(model, VAE):
                latent_vector = model.get_latent(sample_tensor)
            else:
                latent_vector = model.encode(sample_tensor)
            latent_np = latent_vector.squeeze(0).cpu().numpy() # Squeeze to remove batch dim for storage
            
            # Create Curve object
            curva = Curve.from_norm_data(sample, mask[i], mean, std, latent_np)
            curves.append(curva)
            latent_vectors.append(latent_np)
    
    # Convert latent vectors to numpy array
    latent_space = np.array(latent_vectors)
    print(f"Encoded {num_samples} samples into latent space: {latent_space.shape}")
    
    return curves, latent_space


def perform_clustering(latent_space: np.ndarray, curves: Curve, n_clusters: int, random_state: int) -> np.ndarray:
    """
    Perform K-Means clustering on the latent space.
    
    Args:
        latent_space: Latent space representations
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        
    Returns:
        Cluster labels for each sample
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(latent_space)

    for i,num_cluster in enumerate(clusters):
        curves[i].num_cluster = num_cluster

    print(f"Clustering completed: {n_clusters} clusters identified")
    
    return clusters


def plot_latent_space_2d(latent_space: np.ndarray, title: str = "Latent Space (PCA 2D)"):
    """
    Visualize latent space in 2D using PCA.
    
    Args:
        latent_space: Latent space representations
        title: Plot title
    """
    # Reduce to 2 dimensions
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_space)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_latent_space_3d(latent_space: np.ndarray, title: str = "Latent Space (PCA 3D)"):
    """
    Visualize latent space in 3D using PCA.
    
    Args:
        latent_space: Latent space representations
        title: Plot title
    """
    # Reduce to 3 dimensions
    pca = PCA(n_components=3)
    latent_3d = pca.fit_transform(latent_space)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2], alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    
    plt.show()


def plot_clusters_2d(latent_space: np.ndarray, clusters: np.ndarray):
    """
    Visualize clusters in 2D latent space.
    
    Args:
        latent_space: Latent space representations
        clusters: Cluster labels for each sample
    """
    # Reduce to 2 dimensions
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_space)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clusters in Latent Space (PCA 2D)")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.show()


def plot_clusters_3d(latent_space: np.ndarray, clusters: np.ndarray):
    """
    Visualize clusters in 3D latent space.
    
    Args:
        latent_space: Latent space representations
        clusters: Cluster labels for each sample
    """
    # Reduce to 3 dimensions
    pca = PCA(n_components=3)
    latent_3d = pca.fit_transform(latent_space)
    
    # Create 3D plot
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


def main():
    """
    Main execution function for latent space analysis.
    """
    print("=" * 60)
    print("Latent Space Analysis - Driving Style Model")
    print("=" * 60)
    
    # Load and split dataset
    print("\n[1/7] Loading and splitting dataset...")
    dataset = load_and_split_dataset(DATASET_PATH, train_ratio=TRAIN_RATIO)
    
    # Per encoding/clustering usiamo tutti i dati originali
    full_data, full_mask, mean, std = load_dataset(DATASET_PATH)
    
    # Initialize or load model
    if TRAIN_MODEL:
        model_type = "VAE" if USE_VAE else "AutoEncoder"
        print(f"\n[2/7] Initializing and training {model_type} model...")
        
        if USE_VAE:
            model = VAE(dataset.train_data.shape[1], latent_dim=LATENT_DIM)
        else:
            model = AutoEncoder(dataset.train_data.shape[1], latent_dim=LATENT_DIM)
        
        model = train_model(
            model=model,
            train_data=dataset.train_data,
            train_mask=dataset.train_mask,
            val_data=dataset.val_data,
            val_mask=dataset.val_mask,
            epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            batch_size = BATCH_SIZE
        )
        
        # Save weights if requested
        if SAVE_WEIGHTS:
            save_model_weights(model, SAVE_ENCODER_PATH)
        
        model.eval() # Ensure evaluation mode
    else:
        print("\n[2/7] Loading pre-trained model...")
        model = load_model(full_data.shape[1], LATENT_DIM, ENCODER_WEIGHTS_PATH, use_vae=USE_VAE)
    
    # Encode data into latent space
    print("\n[3/7] Encoding data into latent space...")
    curves, latent_space = encode_data(model, full_data, full_mask, mean, std, NUM_SAMPLES)
    

    # # Visualize latent space without clustering
    # print("\n[4/7] Visualizing latent space...")
    # plot_latent_space_2d(latent_space)
    # plot_latent_space_3d(latent_space)
    
    # Perform clustering
    print("\n[5/7] Performing K-Means clustering...")
    clusters = perform_clustering(latent_space, curves, NUM_CLUSTERS, RANDOM_STATE)
    
    # Visualize clusters
    print("\n[6/7] Visualizing clusters...")
    plot_clusters_2d(latent_space, clusters)
    plot_clusters_3d(latent_space, clusters)

    # print("\n[7/7] Visualizing Curves...")
    # for i in range(0 , len(curves)):
    #     if curves[i].num_cluster == 1:
    #         print(f"Clutser:{curves[i].num_cluster}")
    #         curves[i].plot_all()

    
    print("\n[7/7] Media Cluster...")
    cluster_0 = [curve.pushing_score() for curve in curves if curve.num_cluster == 0]
    cluster_1 = [curve.pushing_score() for curve in curves if curve.num_cluster == 1]
    cluster_2 = [curve.pushing_score() for curve in curves if curve.num_cluster == 2]
    cluster_3 = [curve.pushing_score() for curve in curves if curve.num_cluster == 3]
    
    cluster_0 = np.asarray(cluster_0)
    cluster_1 = np.asarray(cluster_1)
    cluster_2 = np.asarray(cluster_2)
    cluster_3 = np.asarray(cluster_3)
    print("Cluster 0:")
    print(f"\t\tGrandezza : {cluster_0.shape}")
    print(f"\t\tMedia : {cluster_0.mean()}")
    print(f"\t\tStd : {cluster_0.std()}")

    print("Cluster 1:")
    print(f"\t\tGrandezza : {cluster_1.shape}")
    print(f"\t\tMedia : {cluster_1.mean()}")
    print(f"\t\tStd : {cluster_1.std()}")

    print("Cluster 2:")
    print(f"\t\tGrandezza : {cluster_2.shape}")
    print(f"\t\tMedia : {cluster_2.mean()}")
    print(f"\t\tStd : {cluster_2.std()}")

    print("Cluster 3:")
    print(f"\t\tGrandezza : {cluster_3.shape}")
    print(f"\t\tMedia : {cluster_3.mean()}")
    print(f"\t\tStd : {cluster_3.std()}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()