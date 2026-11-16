import json
import os
import numpy as np
import pandas as pd

print("=== 1. TEST SET PREDICTIONS & GROUND TRUTH ===")
y_test = np.load("artifacts/y_test.npy")
rf_preds = np.load("assets/rf_predictions.npy")
dnn_preds = np.load("assets/dnn_predictions.npy")
gb_preds = np.load("assets/improved_model_predictions.npy")

print(f"y_test shape: {y_test.shape}")
print(f"rf_preds shape: {rf_preds.shape}")
print(f"dnn_preds shape: {dnn_preds.shape}")
print(f"gb_preds shape: {gb_preds.shape}")

print("\n=== 2. PRINCIPAL MODELS METRICS ===")
rf_m = json.load(open("assets/rf_metrics.json"))
print(f"RF:  Acc={rf_m['accuracy']:.4f}, MacroF1={rf_m['macro_f1']:.4f}, MacroRec={rf_m['macro_recall']:.4f}, W_F1={rf_m['weighted_f1']:.4f}")

dnn_m = json.load(open("assets/dnn_metrics.json"))
print(f"DNN: Acc={dnn_m['accuracy']:.4f}, MacroF1={dnn_m['macro_f1']:.4f}, MacroRec={dnn_m['macro_recall']:.4f}, W_F1={dnn_m['weighted_f1']:.4f}")

gb_m = json.load(open("assets/improved_model_metrics.json"))["test_metrics_outer"]
print(f"GB:  Acc={gb_m['accuracy']:.4f}, MacroF1={gb_m['macro_f1']:.4f}, MacroRec={gb_m['macro_recall']:.4f}, W_F1={gb_m['weighted_f1']:.4f}")

print("\n=== 3. TRAINING HISTORY ===")
th = json.load(open("assets/training_history.json"))
print("History keys:", list(th.keys()))
print(f"Epochs: {len(th['loss'])}")
val_loss = th.get("val_loss", [])
best_epoch = int(np.argmin(val_loss)) + 1 if val_loss else None
print(f"Best Epoch by val_loss: {best_epoch} (val_loss = {min(val_loss):.4f})")

print("\n=== 4. CV MODEL COMPARISON RESULTS ===")
cv_df = pd.read_csv("experiments/final_audit/model_comparison_results.csv")
print(cv_df[["Model", "cv_macro_f1_mean", "cv_accuracy_mean", "cv_log_loss_mean"]].head(5))

print("\n=== 5. OVERSAMPLING RESULTS ===")
os_df = pd.read_csv("assets/oversampling_experiment_results.csv")
print(os_df[["Strategy", "CV_Macro_F1", "CV_Accuracy"]].head(5))

print("\n=== 6. FEATURE IMPORTANCE ===")
fi_df = pd.read_csv("assets/improved_model_feature_importance.csv")
print(fi_df.head(5))
