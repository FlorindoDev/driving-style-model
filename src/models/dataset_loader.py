"""
Dataset Loader per Neural Model

Gestisce il caricamento del dataset normalizzato e lo split train/validation.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


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
