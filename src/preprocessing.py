"""
EduPredict - Data Preprocessing & Pipeline Module
Handles target creation, column transformation (StandardScaler + OneHotEncoder),
stratified train/val/test splitting, and saving model-ready artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
try:
    from sklearn.compose import ColumnTransformer
except (ImportError, Exception):
    from sklearn.compose._column_transformer import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

try:
    from sklearn.model_selection import train_test_split
except (ImportError, Exception):
    from sklearn.model_selection._split import train_test_split

# Grade boundaries and target classification categories
CLASS_MAPPING = {
    0: {"name": "Poor", "grade_range": "0-9", "min_grade": 0, "max_grade": 9},
    1: {"name": "Average", "grade_range": "10-12", "min_grade": 10, "max_grade": 12},
    2: {"name": "Good", "grade_range": "13-15", "min_grade": 13, "max_grade": 15},
    3: {"name": "Excellent", "grade_range": "16-20", "min_grade": 16, "max_grade": 20},
}

# Explicit definition of the 32 input features (15 numerical, 17 categorical)
NUMERICAL_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2"
]

CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

TARGET_COLUMN = "G3"


def load_dataset(filepath="data/student-por.csv"):
    """
    Load the Portuguese student dataset using semicolon delimiter.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath, sep=';')
    return df


def convert_grades_to_classes(g3_series):
    """
    Converts G3 grades (0-20) into 4 discrete classes:
      - 0: Poor (0-9)
      - 1: Average (10-12)
      - 2: Good (13-15)
      - 3: Excellent (16-20)
    """
    bins = [-1, 9, 12, 15, 20]
    classes = pd.cut(g3_series, bins=bins, labels=[0, 1, 2, 3])
    return classes.astype(int).to_numpy()


def build_preprocessor(numerical_cols=None, categorical_cols=None):
    """
    Constructs a ColumnTransformer with:
      - StandardScaler on numerical features
      - OneHotEncoder(handle_unknown='ignore', sparse_output=False) on categorical features
    """
    if numerical_cols is None:
        numerical_cols = NUMERICAL_FEATURES
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor


def split_and_preprocess_dataset(
    data_path="data/student-por.csv",
    output_dir="artifacts",
    random_state=42,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15
):
    """
    Loads dataset, splits into stratified train/val/test sets,
    fits preprocessor ONLY on train data, transforms all sets,
    and serializes all artifacts for local and Colab usage.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_dataset(data_path)

    # 1. Separate features and target
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = convert_grades_to_classes(df[TARGET_COLUMN])

    print("=" * 70)
    print("EDUPREDICT: PREPROCESSING & DATASET SPLIT")
    print("=" * 70)
    print(f"Total dataset shape: {X.shape[0]} rows, {X.shape[1]} input features")
    print("\nOverall Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        meta = CLASS_MAPPING[int(cls)]
        pct = (count / len(y)) * 100
        print(f"  Class {cls} ({meta['name']}, Grades {meta['grade_range']}): {count} ({pct:.2f}%)")

    # 2. Stratified train/temp split (70% train, 30% temp)
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=y
    )

    # 3. Stratified val/test split (50/50 of temp -> 15% val, 15% test of total)
    relative_test_ratio = test_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_ratio,
        random_state=random_state,
        stratify=y_temp
    )

    print("\nDataset Split Counts:")
    print(f"  - Train set: {len(X_train)} samples ({len(X_train)/len(df):.2%})")
    print(f"  - Val set:   {len(X_val)} samples ({len(X_val)/len(df):.2%})")
    print(f"  - Test set:  {len(X_test)} samples ({len(X_test)/len(df):.2%})")

    # Function to print class breakdown for a split
    def print_split_dist(name, y_split):
        u, c = np.unique(y_split, return_counts=True)
        dist_str = ", ".join([f"Class {k} ({CLASS_MAPPING[k]['name']}): {v} ({v/len(y_split):.1%})" for k, v in zip(u, c)])
        print(f"    {name}: {dist_str}")

    print("\nClass Distribution per Split:")
    print_split_dist("Train", y_train)
    print_split_dist("Val  ", y_val)
    print_split_dist("Test ", y_test)

    # 4. Build and fit preprocessor ONLY on training data
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # 5. Transform train, val, and test data
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Ensure dense float32 arrays
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
    if hasattr(X_val_proc, "toarray"):
        X_val_proc = X_val_proc.toarray()
    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()

    X_train_proc = np.asarray(X_train_proc, dtype=np.float32)
    X_val_proc = np.asarray(X_val_proc, dtype=np.float32)
    X_test_proc = np.asarray(X_test_proc, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    # 6. Extract transformed feature names
    feature_names = list(preprocessor.get_feature_names_out())
    print(f"\nProcessed Feature Count: {len(feature_names)} features (from 32 input features)")

    # 7. Check for NaN or Inf
    for name, arr in [("X_train", X_train_proc), ("X_val", X_val_proc), ("X_test", X_test_proc)]:
        if np.isnan(arr).any():
            raise ValueError(f"NaN values detected in {name}")
        if np.isinf(arr).any():
            raise ValueError(f"Infinite values detected in {name}")
    print("Zero NaN or Infinite values detected in any processed arrays.")

    # 8. Save artifacts
    artifacts = {
        "preprocessor.joblib": os.path.join(output_dir, "preprocessor.joblib"),
        "X_train.npy": os.path.join(output_dir, "X_train.npy"),
        "X_val.npy": os.path.join(output_dir, "X_val.npy"),
        "X_test.npy": os.path.join(output_dir, "X_test.npy"),
        "y_train.npy": os.path.join(output_dir, "y_train.npy"),
        "y_val.npy": os.path.join(output_dir, "y_val.npy"),
        "y_test.npy": os.path.join(output_dir, "y_test.npy"),
        "feature_names.json": os.path.join(output_dir, "feature_names.json"),
        "class_mapping.json": os.path.join(output_dir, "class_mapping.json"),
        "split_metadata.json": os.path.join(output_dir, "split_metadata.json"),
    }

    joblib.dump(preprocessor, artifacts["preprocessor.joblib"])
    np.save(artifacts["X_train.npy"], X_train_proc)
    np.save(artifacts["X_val.npy"], X_val_proc)
    np.save(artifacts["X_test.npy"], X_test_proc)
    np.save(artifacts["y_train.npy"], y_train)
    np.save(artifacts["y_val.npy"], y_val)
    np.save(artifacts["y_test.npy"], y_test)

    with open(artifacts["feature_names.json"], "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    with open(artifacts["class_mapping.json"], "w", encoding="utf-8") as f:
        json.dump(CLASS_MAPPING, f, indent=2)

    def count_dict(y_arr):
        u, c = np.unique(y_arr, return_counts=True)
        return {str(int(k)): int(v) for k, v in zip(u, c)}

    split_metadata = {
        "random_state": random_state,
        "dataset_path": data_path,
        "total_samples": len(df),
        "input_features_count": X.shape[1],
        "processed_features_count": len(feature_names),
        "target_column": TARGET_COLUMN,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "train_class_distribution": count_dict(y_train),
        "val_class_distribution": count_dict(y_val),
        "test_class_distribution": count_dict(y_test),
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_names": feature_names,
    }

    with open(artifacts["split_metadata.json"], "w", encoding="utf-8") as f:
        json.dump(split_metadata, f, indent=2)

    print("\nSaved Artifacts in 'artifacts/' directory:")
    for fname, path in artifacts.items():
        size = os.path.getsize(path)
        print(f"  - {fname:<22} ({size:>8,} bytes)")

    return {
        "preprocessor": preprocessor,
        "X_train": X_train_proc,
        "X_val": X_val_proc,
        "X_test": X_test_proc,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_names,
        "metadata": split_metadata
    }


def verify_saved_artifacts(output_dir="artifacts"):
    """
    Reloads all saved artifacts and verifies integrity, shapes, and sample transformation.
    """
    print("\n" + "=" * 70)
    print("VERIFYING SAVED ARTIFACTS")
    print("=" * 70)

    # 1. Reload joblib preprocessor
    prep_path = os.path.join(output_dir, "preprocessor.joblib")
    preprocessor = joblib.load(prep_path)
    print(f"Successfully loaded {prep_path}")

    # 2. Reload numpy arrays
    X_train = np.load(os.path.join(output_dir, "X_train.npy"))
    X_val = np.load(os.path.join(output_dir, "X_val.npy"))
    X_test = np.load(os.path.join(output_dir, "X_test.npy"))
    y_train = np.load(os.path.join(output_dir, "y_train.npy"))
    y_val = np.load(os.path.join(output_dir, "y_val.npy"))
    y_test = np.load(os.path.join(output_dir, "y_test.npy"))

    print(f"  - X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"  - X_val shape:   {X_val.shape}, dtype: {X_val.dtype}")
    print(f"  - X_test shape:  {X_test.shape}, dtype: {X_test.dtype}")
    print(f"  - y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"  - y_val shape:   {y_val.shape}, dtype: {y_val.dtype}")
    print(f"  - y_test shape:  {y_test.shape}, dtype: {y_test.dtype}")

    # Assertions
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature column count mismatch!"
    assert len(X_train) == len(y_train), "X_train and y_train row count mismatch!"
    assert len(X_val) == len(y_val), "X_val and y_val row count mismatch!"
    assert len(X_test) == len(y_test), "X_test and y_test row count mismatch!"
    assert not np.isnan(X_train).any(), "NaN found in X_train!"
    assert not np.isinf(X_train).any(), "Inf found in X_train!"

    # 3. Reload JSON metadata
    with open(os.path.join(output_dir, "feature_names.json"), "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    with open(os.path.join(output_dir, "class_mapping.json"), "r", encoding="utf-8") as f:
        class_mapping = json.load(f)
    with open(os.path.join(output_dir, "split_metadata.json"), "r", encoding="utf-8") as f:
        split_metadata = json.load(f)

    assert len(feature_names) == X_train.shape[1], "feature_names count does not match X_train columns!"
    print(f"  - Feature names count: {len(feature_names)}")
    print(f"  - Class mapping classes: {list(class_mapping.keys())}")

    # 4. Test sample input transformation
    df_raw = load_dataset()
    sample_df = df_raw.drop(columns=[TARGET_COLUMN]).iloc[:3]
    sample_transformed = preprocessor.transform(sample_df)
    if hasattr(sample_transformed, "toarray"):
        sample_transformed = sample_transformed.toarray()
    assert sample_transformed.shape == (3, X_train.shape[1]), "Sample transformation shape mismatch!"
    print(f"  - Sample transformation test passed! Transformed shape: {sample_transformed.shape}")

    print("=" * 70)
    print("ALL ARTIFACT CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    split_and_preprocess_dataset()
    verify_saved_artifacts()
