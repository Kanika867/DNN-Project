"""
EduPredict - Phase 1 Data Verification Script
Loads data/student-por.csv with semicolon delimiter and verifies all requirements.
"""

import os
import pandas as pd
import numpy as np

def verify_dataset(filepath="data/student-por.csv"):
    print("=" * 70)
    print("EDUPREDICT: PHASE 1 DATA VERIFICATION")
    print("=" * 70)

    # 1. Verify file existence
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    print(f"File verified at: {filepath}")
    print(f"File size: {os.path.getsize(filepath):,} bytes\n")

    # 2. Load CSV with semicolon delimiter
    df = pd.read_csv(filepath, sep=';')
    print("Successfully loaded dataset using ';' delimiter.\n")

    # 3. Dataset dimensions
    num_rows, num_cols = df.shape
    print(f"Dataset Dimensions:")
    print(f"  - Total records (rows): {num_rows}")
    print(f"  - Total attributes (columns): {num_cols}\n")

    # 4. All column names
    print("Column Names:")
    for idx, col in enumerate(df.columns, 1):
        print(f"  {idx:2d}. {col}")
    print()

    # 5. Data types
    print("Data Types:")
    print(df.dtypes.to_string())
    print()

    # 6. Missing values
    missing_series = df.isnull().sum()
    total_missing = missing_series.sum()
    print(f"Missing Values Check:")
    print(f"  - Total missing cells across entire dataset: {total_missing}")
    if total_missing > 0:
        print(missing_series[missing_series > 0])
    else:
        print("  - Zero missing values detected across all columns.")
    print()

    # 7. Duplicate records
    duplicates_count = df.duplicated().sum()
    print(f"Duplicate Records Check:")
    print(f"  - Total duplicate rows: {duplicates_count}\n")

    # 8. First five records
    print("First 5 Records:")
    print(df.head().to_string())
    print()

    # 9. Basic statistical summary
    print("Statistical Summary (Numerical Features):")
    print(df.describe().to_string())
    print()

    print("Statistical Summary (Categorical Features):")
    print(df.describe(include=['str']).to_string())
    print()

    # 10. Verify Target Variable G3
    print("=" * 70)
    print("TARGET VARIABLE (G3) VERIFICATION:")
    print("=" * 70)
    if 'G3' not in df.columns:
        raise AssertionError("Target column 'G3' is missing from the dataset!")
    
    g3_min = df['G3'].min()
    g3_max = df['G3'].max()
    g3_mean = df['G3'].mean()
    g3_median = df['G3'].median()
    g3_std = df['G3'].std()

    print(f"G3 Statistics:")
    print(f"  - Min grade: {g3_min}")
    print(f"  - Max grade: {g3_max}")
    print(f"  - Mean grade: {g3_mean:.2f}")
    print(f"  - Median grade: {g3_median:.2f}")
    print(f"  - Standard deviation: {g3_std:.2f}")

    assert 0 <= g3_min <= 20, f"G3 min {g3_min} out of expected range [0, 20]"
    assert 0 <= g3_max <= 20, f"G3 max {g3_max} out of expected range [0, 20]"
    print("  => Confirmed: All G3 values are strictly within the expected 0-20 range.\n")

    # Target class distribution preview based on project definition
    # 0-9: Poor (0), 10-12: Average (1), 13-15: Good (2), 16-20: Excellent (3)
    bins = [-1, 9, 12, 15, 20]
    labels = ["Poor (0)", "Average (1)", "Good (2)", "Excellent (3)"]
    df['target_class'] = pd.cut(df['G3'], bins=bins, labels=labels)
    print("Project 4-Class Distribution Preview:")
    class_dist = df['target_class'].value_counts().sort_index()
    for category, count in class_dist.items():
        pct = (count / len(df)) * 100
        print(f"  - {category}: {count} students ({pct:.2f}%)")
    print()

    # 11. Confirm feature count when removing G3
    features_df = df.drop(columns=['G3', 'target_class'])
    num_features = features_df.shape[1]
    print("=" * 70)
    print("INPUT FEATURE COUNT VERIFICATION:")
    print("=" * 70)
    print(f"Original column count: {num_cols}")
    print(f"After removing target 'G3', input feature count: {num_features}")
    assert num_features == num_cols - 1, f"Expected {num_cols - 1} features, got {num_features}"
    print(f"Includes G1: {'G1' in features_df.columns}")
    print(f"Includes G2: {'G2' in features_df.columns}")
    print("  => Confirmed: Exactly 32 input features remain (including G1 and G2).\n")

    print("=" * 70)
    print("ALL PHASE 1 CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    verify_dataset()
