"""
Dataset Loader per Neural Model

Gestisce il caricamento del dataset normalizzato e lo split train/validation.
"""

import os
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# HUGGING FACE DOWNLOAD
# =============================================================================
HF_REPO_ID = "FlorindoDev/f1_corner_telemetry_2024_2025"
HF_DEFAULT_FILENAME = "data/dataset/normalized_dataset_2024_2025.npz"


def download_raw_telemetry_from_hf(
    subfolder: str = "2025-main",
    repo_id: str = HF_REPO_ID,
    local_dir: str = "data",
    force_download: bool = False
) -> str:
    """
    Scarica i dati grezzi della telemetria F1 da Hugging Face.
    
    Args:
        subfolder: Cartella da scaricare (es. "2025-main", "2024-main")
        repo_id: ID del repository HF
        local_dir: Directory locale dove salvare i file
        force_download: Se True, riscarica anche se esiste già
        
    Returns:
        Path locale della cartella scaricata
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub non installato. Installa con: pip install huggingface_hub"
        )
    
    local_path = os.path.join(local_dir, subfolder)
    
    # Se esiste già e non forziamo, ritorna il path
    if os.path.exists(local_path) and not force_download:
        print(f"Raw telemetry già presente: {local_path}")
        return local_path
    
    print(f"Scaricando raw telemetry da Hugging Face...")
    print(f"  Repo: {repo_id}")
    print(f"  Subfolder: {subfolder}")
    
    # Scarica l'intera cartella usando snapshot_download con allow_patterns
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=f"{subfolder}/**",
        force_download=force_download
    )
    
    result_path = os.path.join(downloaded_path, subfolder)
    print(f"Raw telemetry scaricata: {result_path}")
    return result_path


def download_dataset_from_hf(
    filename: str = HF_DEFAULT_FILENAME,
    repo_id: str = HF_REPO_ID,
    local_dir: str = ".",
    force_download: bool = False
) -> str:
    """
    Scarica il dataset normalizzato da Hugging Face.
    
    Args:
        filename: Path del file nel repo HF (es. "data/dataset/normalized_dataset_2024_2025.npz")
        repo_id: ID del repository HF
        local_dir: Directory locale dove salvare il file
        force_download: Se True, riscarica anche se esiste già
        
    Returns:
        Path locale del file scaricato
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub non installato. Installa con: pip install huggingface_hub"
        )
    
    local_path = os.path.join(local_dir, filename)
    
    # Se esiste già e non forziamo, ritorna il path
    if os.path.exists(local_path) and not force_download:
        print(f"Dataset già presente: {local_path}")
        return local_path
    
    print(f"Scaricando dataset da Hugging Face...")
    print(f"  Repo: {repo_id}")
    print(f"  File: {filename}")
    
    # Scarica il file
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
        force_download=force_download
    )
    
    print(f"Dataset scaricato: {downloaded_path}")
    return downloaded_path


@dataclass
class DatasetSplit:
    """Container per i dati di training e validation."""
    train_data: np.ndarray
    train_mask: np.ndarray
    val_data: Optional[np.ndarray] = None
    val_mask: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carica il dataset normalizzato da un file NPZ.
    
    Args:
        path: Path al file .npz contenente il dataset
        
    Returns:
        Tuple di (data, mask, mean, std)
    """
    dataset = np.load(path, allow_pickle=True)
    data = dataset["data"]
    mask = dataset["mask"]
    mean = dataset["mean"]
    std = dataset["std"]
    
    print(f"Dataset caricato: {path}")
    print(f"  Data shape: {data.shape}")
    print(f"  Mask shape: {mask.shape}")
    
    return data, mask, mean, std


def split_dataset(
    data: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    random_state: int = 42
) -> DatasetSplit:
    """
    Divide il dataset in training e validation.
    
    Args:
        data: Array dei dati
        mask: Array delle mask
        mean: Media per denormalizzazione (opzionale)
        std: Std per denormalizzazione (opzionale)
        train_ratio: Percentuale dei dati per training (0.0 - 1.0)
        shuffle: Se True, mescola i dati prima dello split
        random_state: Seed per riproducibilità
        
    Returns:
        DatasetSplit contenente train e validation data
    """
    n_samples = len(data)
    n_train = int(n_samples * train_ratio)
    
    # Crea indici
    indices = np.arange(n_samples)
    
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
    
    # Split
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_data = data[train_indices]
    train_mask = mask[train_indices]
    
    val_data = data[val_indices] if len(val_indices) > 0 else None
    val_mask = mask[val_indices] if len(val_indices) > 0 else None
    
    print(f"Dataset split (shuffle={shuffle}):")
    print(f"  Training: {len(train_data)} samples ({train_ratio*100:.0f}%)")
    if val_data is not None:
        print(f"  Validation: {len(val_data)} samples ({(1-train_ratio)*100:.0f}%)")
    
    return DatasetSplit(
        train_data=train_data,
        train_mask=train_mask,
        val_data=val_data,
        val_mask=val_mask,
        mean=mean,
        std=std
    )


def load_and_split_dataset(
    path: str,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    random_state: int = 42
) -> DatasetSplit:
    """
    Carica il dataset e lo divide in training e validation.
    
    Args:
        path: Path al file .npz
        train_ratio: Percentuale per training (0.0 - 1.0)
        shuffle: Se True, mescola prima dello split
        random_state: Seed per riproducibilità
        
    Returns:
        DatasetSplit con train/validation data
    """
    data, mask, mean, std = load_dataset(path)
    return split_dataset(data, mask, mean, std, train_ratio, shuffle, random_state)
