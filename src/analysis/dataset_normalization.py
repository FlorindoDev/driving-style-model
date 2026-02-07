import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class NormalizationConfig:
    """Configuration for normalization operations."""
    
    # ----- Constants -----
    padding_value: float = -1000.0
    max_samples_per_curve: int = 50
    compound_categories: tuple = ('HARD', 'INTERMEDIATE', 'WET', 'MEDIUM', 'SOFT')
    
    # ----- Dataset Paths (for script mode) -----
    input_csv_path: str = "data/dataset/dataset_curves.csv"
    output_dir: str = "data/dataset"
    output_filename: str = "normalized_dataset.npz"


# Default config instance
DEFAULT_CONFIG = NormalizationConfig()

# Export constants for backward compatibility
PADDING_VALUE = DEFAULT_CONFIG.padding_value
COMPOUND_CATEGORIES = list(DEFAULT_CONFIG.compound_categories)
MAX_SAMPLES_PER_CURVE = DEFAULT_CONFIG.max_samples_per_curve


# =============================================================================
# SINGLE CURVE NORMALIZATION (for inference)
# =============================================================================
def pad_or_truncate(
    arr, 
    target_length: int, 
    padding_value: float = PADDING_VALUE
) -> list:
    """
    Pad or truncate an array to a target length.
    
    Args:
        arr: Input array or list
        target_length: Desired length
        padding_value: Value to use for padding
        
    Returns:
        List with exactly target_length elements
    """
    if hasattr(arr, 'tolist'):
        arr = arr.tolist()
    elif not isinstance(arr, list):
        arr = list(arr)
    
    if len(arr) >= target_length:
        return arr[:target_length]
    return arr + [padding_value] * (target_length - len(arr))


def curve_to_raw_features(
    curve,
    config: NormalizationConfig = None
) -> np.ndarray:
    """
    Convert a Curve object to a raw (non-normalized) feature vector.
    
    Layout:
      0: life
      1:51 speed (50 columns)
      51:101 rpm (50 columns)
      101:151 throttle (50 columns)
      151:201 brake (50 columns)
      201:251 acc_x (50 columns)
      251:301 acc_y (50 columns)
      301:351 acc_z (50 columns)
      351-355: Compound one-hot (HARD, INTERMEDIATE, WET, MEDIUM, SOFT)
    
    Args:
        curve: Curve object from CurveDetector
        config: Normalization configuration
        
    Returns:
        Raw feature vector as numpy array
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    max_samples = config.max_samples_per_curve
    padding = config.padding_value
    categories = config.compound_categories
    
    # Prepare raw features with padding
    life = curve.life if hasattr(curve, 'life') else 0
    speed = pad_or_truncate(curve.speed, max_samples, padding)
    rpm = pad_or_truncate(curve.rpm, max_samples, padding)
    throttle = pad_or_truncate(curve.throttle, max_samples, padding)
    brake = pad_or_truncate(curve.brake, max_samples, padding)
    acc_x = pad_or_truncate(curve.acc_x, max_samples, padding)
    acc_y = pad_or_truncate(curve.acc_y, max_samples, padding)
    acc_z = pad_or_truncate(curve.acc_z, max_samples, padding)
    
    # Compound one-hot encoding
    compound_one_hot = [
        1.0 if curve.compound == cat else 0.0 
        for cat in categories
    ]
    
    # Build raw feature vector
    raw_features = (
        [life] + speed + rpm + throttle + brake + 
        acc_x + acc_y + acc_z + compound_one_hot
    )
    
    return np.array(raw_features, dtype=np.float64)


def normalize_sample(
    raw_features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    padding_value: float = PADDING_VALUE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize a single sample (feature vector) using pre-computed mean and std.
    
    Args:
        raw_features: Raw feature vector as numpy array
        mean: Mean values for normalization
        std: Std values for normalization  
        padding_value: Value used for padding
        
    Returns:
        Tuple of (normalized_features, mask)
    """
    # Build mask (1 = valid, 0 = padding)
    mask = np.array([
        0.0 if val == padding_value else 1.0 
        for val in raw_features
    ], dtype=np.float64)
    
    # Z-score normalization
    normalized = np.zeros_like(raw_features, dtype=np.float64)
    for i in range(len(raw_features)):
        if raw_features[i] != padding_value and std[i] != 0:
            normalized[i] = (raw_features[i] - mean[i]) / std[i]
        elif raw_features[i] == padding_value:
            normalized[i] = 0.0  # Padding becomes 0
    
    return normalized, mask


def normalize_curve(
    curve,
    mean: np.ndarray,
    std: np.ndarray,
    config: NormalizationConfig = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a Curve object to a normalized feature vector.
    
    This is the main function to use for normalizing a single curve
    during inference (e.g., in evaluate.py).
    
    Args:
        curve: Curve object from CurveDetector
        mean: Mean values for normalization (from dataset)
        std: Std values for normalization (from dataset)
        config: Normalization configuration
        
    Returns:
        Tuple of (normalized_features, mask) as numpy arrays
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Get raw features
    raw_features = curve_to_raw_features(curve, config)
    
    # Normalize using normalize_sample
    normalized, mask = normalize_sample(
        raw_features, mean, std, config.padding_value
    )
    
    return normalized, mask


def normalize_telemetry_json(
    telemetry_path: str,
    corners_path: str,
    dataset_path: str,
    config: NormalizationConfig = None
) -> list:
    """
    Load telemetry JSON, detect curves, and normalize them.
    
    This is the HIGH-LEVEL function that does everything in one call:
    1. Loads normalization stats from dataset
    2. Detects curves from telemetry JSON
    3. Normalizes each curve
    
    Args:
        telemetry_path: Path to telemetry JSON file (e.g., "4_tel.json")
        corners_path: Path to corners JSON file (e.g., "corners.json")
        dataset_path: Path to normalized dataset .npz file (for mean/std)
        config: Normalization configuration
        
    Returns:
        List of dicts containing:
        - 'curve': Original Curve object
        - 'normalized': Normalized feature vector (np.ndarray)
        - 'mask': Padding mask (np.ndarray)
    
    """
    from src.analysis.CurveDetector import CurveDetector
    
    if config is None:
        config = DEFAULT_CONFIG
    
    # Load normalization stats
    data, mask, mean, std, columns = load_normalized_data(dataset_path)
    
    # Detect curves
    detector = CurveDetector(telemetry_path, corners_path)
    curves = detector.calcolo_curve()
    
    # Normalize each curve
    results = []
    for curve in curves:
        normalized, curve_mask = normalize_curve(curve, mean, std, config)
        results.append({
            'curve': curve,
            'normalized': normalized,
            'mask': curve_mask
        })
    
    return results


# =============================================================================
# DATASET NORMALIZATION (for training data preparation)
# =============================================================================
def one_hot_encode_compound(
    df: pd.DataFrame, 
    categories: List[str] = None
) -> pd.DataFrame:
    """
    Perform one-hot encoding on the 'Compound' column.
    
    Args:
        df: Input DataFrame
        categories: List of compound categories to encode
        
    Returns:
        DataFrame with 'Compound' column replaced by one-hot encoded columns
    """
    if categories is None:
        categories = COMPOUND_CATEGORIES
        
    if 'Compound' not in df.columns:
        return df
    
    df = df.copy()
    compound_dummies = pd.get_dummies(df['Compound'], prefix='Compound', dtype=float)
    
    # Add missing columns and enforce order
    for cat in categories:
        col = f"Compound_{cat}"
        if col not in compound_dummies.columns:
            compound_dummies[col] = 0.0
    
    compound_dummies = compound_dummies[[f"Compound_{cat}" for cat in categories]]

    df = pd.concat([df, compound_dummies], axis=1)
    df = df.drop('Compound', axis=1)
    
    return df


def create_padding_mask(
    df: pd.DataFrame, 
    padding_value: float = PADDING_VALUE
) -> pd.DataFrame:
    """
    Create a mask where 1 = valid data, 0 = padding.
    """
    return (df != padding_value).astype(float)


def compute_grouped_stats(
    df: pd.DataFrame,
    skip_prefixes: List[str] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute mean and std for each column, grouping columns with same prefix.
    
    Columns ending with _<number> are grouped together and share the same
    mean/std computed from all values in the group.
    """
    skip_prefixes = skip_prefixes or ["Compound"]
    
    mean_dict: Dict[str, float] = {}
    std_dict: Dict[str, float] = {}
    
    grouped_cols: Dict[str, List[str]] = {}
    single_cols: List[str] = []
    
    # Identify column groups (columns ending with _<number>)
    for col in df.columns:
        match = re.match(r"^(.*)_\d+$", col)
        if match:
            prefix = match.group(1)
            if prefix not in grouped_cols:
                grouped_cols[prefix] = []
            grouped_cols[prefix].append(col)
        else:
            single_cols.append(col)
    
    # Process groups - compute global mean/std across all columns in group
    for prefix, cols in grouped_cols.items():
        group_data = df[cols].values.flatten()
        g_mean = np.nanmean(group_data)
        g_std = np.nanstd(group_data)
        
        if g_std == 0 or np.isnan(g_std):
            g_std = 1.0
            
        for col in cols:
            mean_dict[col] = g_mean
            std_dict[col] = g_std
    
    # Process single columns
    for col in single_cols:
        # Skip normalization for specified prefixes
        if any(skip in col for skip in skip_prefixes):
            mean_dict[col] = 0.0
            std_dict[col] = 1.0
            continue
        
        val_mean = df[col].mean()
        val_std = df[col].std()
        
        if pd.isna(val_std) or val_std == 0:
            val_std = 1.0
        if pd.isna(val_mean):
            val_mean = 0.0
            
        mean_dict[col] = val_mean
        std_dict[col] = val_std
    
    mean = pd.Series(mean_dict)[df.columns]
    std = pd.Series(std_dict)[df.columns]
    
    return mean, std


def normalize_dataframe(
    df: pd.DataFrame,
    config: NormalizationConfig = None,
    apply_one_hot: bool = True,
    skip_prefixes: List[str] = None,
    return_stats: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """
    Normalize a DataFrame using z-score normalization.
    
    This is the main function to use for normalizing an entire dataset
    (e.g., for training data preparation).
    
    Performs:
    1. One-hot encoding of 'Compound' column (if present and apply_one_hot=True)
    2. Creation of padding mask
    3. Grouped z-score normalization
    
    Args:
        df: Input DataFrame to normalize
        config: Normalization configuration
        apply_one_hot: Whether to apply one-hot encoding to 'Compound' column
        skip_prefixes: Column prefixes to skip during normalization
        return_stats: Whether to return mean and std
        
    Returns:
        Tuple of:
        - Normalized DataFrame (padding replaced with 0.0)
        - Mask DataFrame (1=valid, 0=padding)
        - Mean Series (if return_stats=True, else None)
        - Std Series (if return_stats=True, else None)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    df = df.copy()
    categories = list(config.compound_categories)
    padding_value = config.padding_value
    
    # Apply one-hot encoding if needed
    if apply_one_hot:
        df = one_hot_encode_compound(df, categories)
    
    # Create mask before replacing padding
    mask = create_padding_mask(df, padding_value)
    
    # Replace padding with NaN for stats calculation
    df_for_stats = df.replace(padding_value, np.nan)
    
    # Compute grouped statistics
    mean, std = compute_grouped_stats(df_for_stats, skip_prefixes)
    
    # Apply z-score normalization
    df_normalized = (df_for_stats - mean) / std
    df_normalized = df_normalized.fillna(0.0)
    
    if return_stats:
        return df_normalized, mask, mean, std
    else:
        return df_normalized, mask, None, None


def denormalize_dataframe(
    df: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
    mask: Optional[pd.DataFrame] = None,
    padding_value: float = PADDING_VALUE
) -> pd.DataFrame:
    """
    Denormalize a DataFrame using stored mean and std.
    """
    df_denorm = (df * std) + mean
    
    if mask is not None:
        df_denorm = df_denorm.where(mask == 1, padding_value)
    
    return df_denorm


# =============================================================================
# I/O FUNCTIONS
# =============================================================================
def save_normalized_data(
    df_normalized: pd.DataFrame,
    mask: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
    output_path: str
) -> None:
    """Save normalized data to .npz file."""
    np.savez(
        output_path,
        data=df_normalized.values,
        mask=mask.values,
        mean=mean.values,
        std=std.values,
        columns=df_normalized.columns.values
    )
    print(f"Saved normalized data to {output_path}")


def load_normalized_data(
    input_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load normalized data from .npz file.
    
    Returns:
        Tuple of (data, mask, mean, std, columns)
    """
    loaded = np.load(input_path, allow_pickle=True)
    return (
        loaded['data'],
        loaded['mask'],
        loaded['mean'],
        loaded['std'],
        loaded['columns']
    )


# =============================================================================
# SCRIPT MODE: Run as standalone script
# =============================================================================
if __name__ == "__main__":
    # Configuration
    config = NormalizationConfig(
        input_csv_path="data/dataset/dataset_curves.csv",
        output_dir="data/dataset",
        output_filename="normalized_dataset.npz"
    )
    
    # Load dataset
    print(f"Loading CSV from {config.input_csv_path}...")
    df = pd.read_csv(
        config.input_csv_path,
        sep=",",
        encoding="utf-8",
        decimal="."
    )
    
    # Remove unnecessary columns
    df = df.drop(df.columns[:5], axis=1)
    df = df.drop(df.columns[2], axis=1)
    
    # Remove X, Y, Z columns
    df = df.drop(df.columns[352:502], axis=1)
    
    # Remove time and distance columns
    df = df.drop(df.columns[352:], axis=1)
    
    # Normalize
    print("Normalizing dataset...")
    df_normalized, mask, mean, std = normalize_dataframe(df, config)
    
    print(f"Normalized Data Shape: {df_normalized.shape}")
    print(f"Mask Shape: {mask.shape}")
    
    # Save
    output_path = f"{config.output_dir}/{config.output_filename}"
    save_normalized_data(df_normalized, mask, mean, std, output_path)
