"""
EduPredict - Random Forest Local Training Script
Trains a RandomForestClassifier using Phase 2 preprocessed arrays.
Evaluates on validation set, then assesses generalization performance on test set.
Generates metrics, feature importances, and diagnostic visualizations.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure optional unused estimators in sklearn.ensemble don't trigger Windows Application Control
from unittest.mock import MagicMock
if 'sklearn.ensemble._hist_gradient_boosting' not in sys.modules:
    sys.modules['sklearn.ensemble._hist_gradient_boosting'] = MagicMock()
if 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting' not in sys.modules:
    sys.modules['sklearn.ensemble._hist_gradient_boosting.gradient_boosting'] = MagicMock()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'


def train_and_evaluate_rf(
    artifacts_dir="artifacts",
    models_dir="models",
    assets_dir="assets",
    n_estimators=300,
    random_state=42
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    print("=" * 70)
    print("EDUPREDICT: RANDOM FOREST MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # 1. Load artifacts
    print("Loading preprocessed arrays and metadata from artifacts/...")
    X_train = np.load(os.path.join(artifacts_dir, "X_train.npy"))
    y_train = np.load(os.path.join(artifacts_dir, "y_train.npy"))
    X_val = np.load(os.path.join(artifacts_dir, "X_val.npy"))
    y_val = np.load(os.path.join(artifacts_dir, "y_val.npy"))
    X_test = np.load(os.path.join(artifacts_dir, "X_test.npy"))
    y_test = np.load(os.path.join(artifacts_dir, "y_test.npy"))

    with open(os.path.join(artifacts_dir, "feature_names.json"), "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    with open(os.path.join(artifacts_dir, "class_mapping.json"), "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    target_names = [class_mapping[str(i)]["name"] for i in range(len(class_mapping))]
    print(f"  - Train shape: {X_train.shape}, Targets: {y_train.shape}")
    print(f"  - Val shape:   {X_val.shape}, Targets: {y_val.shape}")
    print(f"  - Test shape:  {X_test.shape}, Targets: {y_test.shape}")
    print(f"  - Features:    {len(feature_names)} features")
    print(f"  - Classes:     {target_names}\n")

    # 2. Initialize and Train Random Forest Model
    print("Initializing RandomForestClassifier:")
    print(f"  n_estimators = {n_estimators}")
    print(f"  class_weight = 'balanced'")
    print(f"  random_state = {random_state}")
    print(f"  n_jobs = -1")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    print("\nFitting Random Forest on X_train...")
    rf.fit(X_train, y_train)
    print("Training complete.\n")

    # 3. Model-Selection Decision Evaluation on Validation Set
    val_preds = rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_macro_p, val_macro_r, val_macro_f1, _ = precision_recall_fscore_support(y_val, val_preds, average="macro", zero_division=0)
    val_weighted_f1 = precision_recall_fscore_support(y_val, val_preds, average="weighted", zero_division=0)[2]

    print("-" * 70)
    print("VALIDATION SET EVALUATION (Model Selection Decision Check):")
    print(f"  Validation Accuracy:    {val_acc:.4f} ({val_acc:.2%})")
    print(f"  Validation Macro F1:    {val_macro_f1:.4f}")
    print(f"  Validation Weighted F1: {val_weighted_f1:.4f}")
    print("-" * 70 + "\n")

    # 4. Final Evaluation on Untouched Test Set
    test_preds = rf.predict(X_test)
    test_proba = rf.predict_proba(X_test)

    test_acc = accuracy_score(y_test, test_preds)
    test_macro_p, test_macro_r, test_macro_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="macro", zero_division=0)
    test_weighted_p, test_weighted_r, test_weighted_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="weighted", zero_division=0)

    per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
        y_test, test_preds, average=None, zero_division=0
    )

    clf_report_str = classification_report(y_test, test_preds, target_names=target_names, zero_division=0)
    clf_report_dict = classification_report(y_test, test_preds, target_names=target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, test_preds)

    print("=" * 70)
    print("FINAL TEST SET EVALUATION METRICS:")
    print("=" * 70)
    print(f"  Accuracy:        {test_acc:.4f} ({test_acc:.2%})")
    print(f"  Macro Precision: {test_macro_p:.4f}")
    print(f"  Macro Recall:    {test_macro_r:.4f}")
    print(f"  Macro F1-Score:  {test_macro_f1:.4f}  <-- Primary metric (handles class imbalance)")
    print(f"  Weighted F1:     {test_weighted_f1:.4f}\n")

    print("Per-Class Metrics on Test Set:")
    for idx, name in enumerate(target_names):
        print(f"  Class {idx} ({name:<9}): Precision = {per_class_p[idx]:.4f}, Recall = {per_class_r[idx]:.4f}, F1 = {per_class_f1[idx]:.4f}, Support = {int(per_class_sup[idx])}")
    print()

    print("Classification Report:")
    print(clf_report_str)

    print("Confusion Matrix:")
    print(cm)
    print()

    # 5. Feature Importances
    importances = rf.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    fi_df["rank"] = fi_df.index + 1

    print("Top 10 Most Important Features:")
    for i, row in fi_df.head(10).iterrows():
        print(f"  {row['rank']:2d}. {row['feature']:<22}: {row['importance']:.4f}")
    print()

    # 6. Save Artifacts
    # A. Model
    model_save_path = os.path.join(models_dir, "random_forest.joblib")
    joblib.dump(rf, model_save_path)
    print(f"Saved model to: {model_save_path}")

    # B. Test Predictions
    preds_save_path = os.path.join(assets_dir, "rf_predictions.npy")
    np.save(preds_save_path, test_preds)
    print(f"Saved test predictions to: {preds_save_path}")

    # C. Metrics Summary JSON
    metrics_summary = {
        "model_name": "RandomForestClassifier",
        "parameters": {
            "n_estimators": n_estimators,
            "class_weight": "balanced",
            "random_state": random_state,
            "n_jobs": -1
        },
        "validation_metrics": {
            "accuracy": float(val_acc),
            "macro_precision": float(val_macro_p),
            "macro_recall": float(val_macro_r),
            "macro_f1": float(val_macro_f1),
            "weighted_f1": float(val_weighted_f1),
        },
        "test_metrics": {
            "accuracy": float(test_acc),
            "macro_precision": float(test_macro_p),
            "macro_recall": float(test_macro_r),
            "macro_f1": float(test_macro_f1),
            "weighted_precision": float(test_weighted_p),
            "weighted_recall": float(test_weighted_r),
            "weighted_f1": float(test_weighted_f1),
            "per_class": {
                name: {
                    "class_index": idx,
                    "precision": float(per_class_p[idx]),
                    "recall": float(per_class_r[idx]),
                    "f1_score": float(per_class_f1[idx]),
                    "support": int(per_class_sup[idx]),
                }
                for idx, name in enumerate(target_names)
            }
        },
        "confusion_matrix": cm.tolist()
    }
    metrics_save_path = os.path.join(assets_dir, "rf_metrics.json")
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics summary to: {metrics_save_path}")

    # D. Classification Report JSON
    clf_rep_save_path = os.path.join(assets_dir, "rf_classification_report.json")
    with open(clf_rep_save_path, "w", encoding="utf-8") as f:
        json.dump(clf_report_dict, f, indent=2)
    print(f"Saved classification report to: {clf_rep_save_path}")

    # E. Feature Importance CSV
    fi_csv_path = os.path.join(assets_dir, "feature_importance.csv")
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"Saved feature importance table to: {fi_csv_path}")

    # F. Confusion Matrix Plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names,
        cbar=True, linewidths=0.8, linecolor="#eeeeee"
    )
    plt.title("Random Forest: Test Confusion Matrix", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight='bold')
    plt.ylabel("True Label", fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_plot_path = os.path.join(assets_dir, "rf_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

    # G. Feature Importance Plot (Top 20)
    plt.figure(figsize=(10, 7))
    top_fi = fi_df.head(20).sort_values(by="importance", ascending=True)
    colors = ["#2980b9" if "G1" in f or "G2" in f else "#7f8c8d" for f in top_fi["feature"]]
    bars = plt.barh(top_fi["feature"], top_fi["importance"], color=colors, edgecolor="#333333", linewidth=0.6)
    plt.title("Random Forest: Top 20 Feature Importances", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Gini Importance Score", fontsize=11, fontweight='bold')
    plt.ylabel("Transformed Feature", fontsize=11)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va='center', fontsize=9, fontweight='bold')
    plt.xlim(0, max(top_fi["importance"]) * 1.15)
    plt.tight_layout()
    fi_plot_path = os.path.join(assets_dir, "rf_feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to: {fi_plot_path}")

    print("=" * 70)
    print("ALL RANDOM FOREST TRAINING AND EVALUATION TASKS COMPLETED!")
    print("=" * 70)

    return metrics_summary


if __name__ == "__main__":
    train_and_evaluate_rf()
