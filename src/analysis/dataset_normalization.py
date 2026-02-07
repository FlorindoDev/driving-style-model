import re
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict


# Default configuration
PADDING_VALUE = -1000.0
COMPOUND_CATEGORIES = ['HARD', 'INTERMEDIATE','WET', 'MEDIUM', 'SOFT']
PATH_CSV = "data/dataset/dataset_curves.csv"
PATH_SAVE = "data/dataset"
NAME_SAVE_FILE = "normalized_dataset.npz"


def one_hot_encode_compound( df: pd.DataFrame, categories: List[str] = COMPOUND_CATEGORIES) -> pd.DataFrame:
    """
    Perform one-hot encoding on the 'Compound' column.
    
    Args:
        df: Input DataFrame
        categories: List of compound categories to encode
        
    Returns:
        DataFrame with 'Compound' column replaced by one-hot encoded columns
    """
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


def create_padding_mask(df: pd.DataFrame, padding_value: float = PADDING_VALUE) -> pd.DataFrame:
    """
    Create a mask where 1 = valid data, 0 = padding.
    
    Args:
        df: Input DataFrame
        padding_value: Value used for padding
        
    Returns:
        DataFrame with same shape, containing 1s and 0s
    """
    return (df != padding_value).astype(float)


def compute_grouped_stats(df: pd.DataFrame,skip_prefixes: List[str] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Compute mean and std for each column, grouping columns with same prefix.
    
    Columns ending with _<number> are grouped together and share the same
    mean/std computed from all values in the group.
    
    Args:
        df: DataFrame with NaN for padding values
        skip_prefixes: List of column prefixes to skip (won't normalize)
        
    Returns:
        Tuple of (mean Series, std Series) aligned with df columns
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
        padding_value: float = PADDING_VALUE,
        apply_one_hot: bool = True,
        compound_categories: List[str] = COMPOUND_CATEGORIES,
        skip_prefixes: List[str] = None,
        return_stats: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """
    Normalize a DataFrame using z-score normalization.
    
    Performs:
    1. One-hot encoding of 'Compound' column (if present and apply_one_hot=True)
    2. Creation of padding mask
    3. Grouped z-score normalization
    
    Args:
        df: Input DataFrame to normalize
        padding_value: Value used for padding (will be masked)
        apply_one_hot: Whether to apply one-hot encoding to 'Compound' column
        compound_categories: Categories for one-hot encoding
        skip_prefixes: Column prefixes to skip during normalization
        return_stats: Whether to return mean and std
        
    Returns:
        Tuple of:
        - Normalized DataFrame (padding replaced with 0.0)
        - Mask DataFrame (1=valid, 0=padding)
        - Mean Series (if return_stats=True, else None)
        - Std Series (if return_stats=True, else None)
    """
    df = df.copy()
    
    # Apply one-hot encoding if needed
    if apply_one_hot:
        df = one_hot_encode_compound(df, compound_categories)
    
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
    mask: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Denormalize a DataFrame using stored mean and std.
    
    Args:
        df: Normalized DataFrame
        mean: Mean values used for normalization
        std: Std values used for normalization
        mask: Optional mask to restore padding values
        
    Returns:
        Denormalized DataFrame
    """
    df_denorm = (df * std) + mean
    
    if mask is not None:
        df_denorm = df_denorm.where(mask == 1, PADDING_VALUE)
    
    return df_denorm


def save_normalized_data(
        df_normalized: pd.DataFrame,
        mask: pd.DataFrame,
        mean: pd.Series,
        std: pd.Series,
        output_path: str
    ) -> None:
    """
    Save normalized data to .npz file.
    
    Args:
        df_normalized: Normalized DataFrame
        mask: Padding mask DataFrame
        mean: Mean values Series
        std: Std values Series
        output_path: Path to save the .npz file
    """
    np.savez(
        output_path,
        data=df_normalized.values,
        mask=mask.values,
        mean=mean.values,
        std=std.values,
        columns=df_normalized.columns.values
    )
    print(f"Saved normalized data to {output_path}")


def load_normalized_data(input_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load normalized data from .npz file.
    
    Args:
        input_path: Path to the .npz file
        
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


# ============================================================================
# SCRIPT MODE: Run as standalone script
# ============================================================================
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv(
        PATH_CSV,
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
    df_normalized, mask, mean, std = normalize_dataframe(df)
    
    print("Normalized Data Shape:", df_normalized.shape)
    print("Mask Shape:", mask.shape)
    
    # Save
    save_normalized_data(
        df_normalized, mask, mean, std,
        output_path=f"{PATH_SAVE}/{NAME_SAVE_FILE}"
    )
