
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


df = pd.read_csv(
    "../../data/dataset/dataset_curves.csv",   
    sep=",",
    encoding="utf-8",
    decimal="."
)


# Rimozione delle colonne non necessarie
df = df.drop(df.columns[:5], axis=1)

df = df.drop(df.columns[2], axis=1)

# Rimozione delle colonne X, Y, Z
df = df.drop(df.columns[352:502], axis=1)

# Rimozione time e distanza 
df = df.drop(df.columns[352:], axis=1)


# --- ONE-HOT ENCODING ---
if 'Compound' in df.columns:
    print("Found 'Compound' column, performing One-Hot Encoding...")

    compound_dummies = pd.get_dummies(df['Compound'], prefix='Compound', dtype=float)

    df = pd.concat([df, compound_dummies], axis=1)
    df = df.drop('Compound', axis=1)
else:
    print("WARNING: 'Compound' column not found in dataframe columns:", df.columns)



# --- MASKING & NORMALIZATION ---
PADDING_VALUE = -1000.0

# Create mask (1 for valid data, 0 for padding)
mask = (df != PADDING_VALUE).astype(float)


df_for_stats = df.replace(PADDING_VALUE, np.nan)

# --- GROUPED STATS CALCULATION ---
import re

# Dictionary to hold the final mean and std for each column
mean_dict = {}
std_dict = {}

# Identify column groups
# We look for columns ending in _<number>
grouped_cols = {}
single_cols = []

for col in df.columns:
    match = re.match(r"^(.*)_\d+$", col)
    if match:
        prefix = match.group(1)
        if prefix not in grouped_cols:
            grouped_cols[prefix] = []
        grouped_cols[prefix].append(col)
    else:
        single_cols.append(col)

# 1. Process Groups
for prefix, cols in grouped_cols.items():
    print(f"Processing group '{prefix}' with {len(cols)} columns...")
    # Extract all data for this group
    group_data = df_for_stats[cols].values.flatten()
    
    # Calculate global mean and std ignoring NaNs
    g_mean = np.nanmean(group_data)
    g_std = np.nanstd(group_data)
    
    # Safety for std
    if g_std == 0:
        g_std = 1.0
        
    # Assign to all columns in this group
    for col in cols:
        mean_dict[col] = g_mean
        std_dict[col] = g_std

# 2. Process Single Columns
for col in single_cols:
    if "Compound" in col: 
        mean_dict[col] = 0.0
        std_dict[col] = 1.0
        continue  # Skip one-hot encoded columns

    col_data = df_for_stats[col]
    val_mean = col_data.mean()
    val_std = col_data.std()
    
    if pd.isna(val_std) or val_std == 0:
        val_std = 1.0
        
    mean_dict[col] = val_mean
    std_dict[col] = val_std

# Convert dicts to Series to align with dataframe
mean = pd.Series(mean_dict)
std = pd.Series(std_dict)

# Reorder mean/std to match df columns order
mean = mean[df.columns]
std = std[df.columns]

# Normalize (Z-Score)
# (X - mu) / sigma
df_normalized = (df_for_stats - mean) / std


df_normalized = df_normalized.fillna(0.0)


# Output shapes
print("Normalized Data Shape:", df_normalized.shape)
print("Mask Shape:", mask.shape)


# Save to .npz
output_filename = "../../data/dataset/normalized_dataset2.npz"
np.savez(
    output_filename, 
    data=df_normalized.values, 
    mask=mask.values, 
    mean=mean.values, 
    std=std.values,
    columns=df_normalized.columns.values
)
print(f"Saved normalized data to {output_filename}")

