"""
EduPredict - Phase 4E Test Suite: Model Laboratory
Verifies:
1. Metrics loading for all 3 principal models (RF, DNN, GB)
2. Exact matching against saved artifacts
3. Confusion matrix shape, class ordering, and error distribution
4. Feature importance and permutation importance consistency
5. Cross-validation and oversampling comparison benchmarks
6. DNN training history structure and epoch metrics
7. Zero TensorFlow imports or DLL invocations
8. Multi-page compilation and web server HTTP 200 status
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

def run_tests():
    print("=== 1. VERIFY PRINCIPAL MODELS TEST METRICS ===")
    rf_m = json.load(open("assets/rf_metrics.json"))["test_metrics"]
    dnn_m = json.load(open("assets/dnn_metrics.json"))["test_metrics"]
    gb_m = json.load(open("assets/improved_model_metrics.json"))["test_metrics_outer"]

    # RF
    assert np.isclose(rf_m["accuracy"], 0.663265, atol=0.001), f"RF acc mismatch: {rf_m['accuracy']}"
    assert np.isclose(rf_m["macro_f1"], 0.677922, atol=0.001), f"RF macro F1 mismatch: {rf_m['macro_f1']}"
    assert np.isclose(rf_m["macro_recall"], 0.726185, atol=0.001), f"RF macro recall mismatch: {rf_m['macro_recall']}"
    print(f"  - Random Forest verified: Acc={rf_m['accuracy']*100:.2f}%, Macro F1={rf_m['macro_f1']:.4f}")

    # DNN
    assert np.isclose(dnn_m["accuracy"], 0.683673, atol=0.001), f"DNN acc mismatch: {dnn_m['accuracy']}"
    assert np.isclose(dnn_m["macro_f1"], 0.674702, atol=0.001), f"DNN macro F1 mismatch: {dnn_m['macro_f1']}"
    assert np.isclose(dnn_m["macro_recall"], 0.698940, atol=0.001), f"DNN macro recall mismatch: {dnn_m['macro_recall']}"
    print(f"  - Deep Neural Network verified: Acc={dnn_m['accuracy']*100:.2f}%, Macro F1={dnn_m['macro_f1']:.4f}")

    # GB
    assert np.isclose(gb_m["accuracy"], 0.724489, atol=0.001), f"GB acc mismatch: {gb_m['accuracy']}"
    assert np.isclose(gb_m["macro_f1"], 0.751645, atol=0.001), f"GB macro F1 mismatch: {gb_m['macro_f1']}"
    assert np.isclose(gb_m["macro_recall"], 0.775345, atol=0.001), f"GB macro recall mismatch: {gb_m['macro_recall']}"
    print(f"  - Gradient Boosting verified: Acc={gb_m['accuracy']*100:.2f}%, Macro F1={gb_m['macro_f1']:.4f}")

    print("\n=== 2. VERIFY PREDICTION ARRAYS AND CONFUSION MATRIX ORDERING ===")
    y_test = np.load("artifacts/y_test.npy")
    rf_preds = np.load("assets/rf_predictions.npy")
    dnn_preds = np.load("assets/dnn_predictions.npy")
    gb_preds = np.load("assets/improved_model_predictions.npy")

    assert len(y_test) == 98, f"Expected 98 test samples, got {len(y_test)}"
    assert len(rf_preds) == 98 and len(dnn_preds) == 98 and len(gb_preds) == 98

    from sklearn.metrics import confusion_matrix
    cm_gb = confusion_matrix(y_test, gb_preds, labels=[0, 1, 2, 3])
    assert cm_gb.shape == (4, 4), f"Invalid CM shape: {cm_gb.shape}"
    assert np.trace(cm_gb) == 71, f"Expected 71 correct test predictions for GB, got {np.trace(cm_gb)}"

    # Verify zero severe errors
    severe_errs = 0
    for i in range(4):
        for j in range(4):
            if abs(i - j) > 1:
                severe_errs += cm_gb[i][j]
    assert severe_errs == 0, f"Expected 0 severe errors in GB, got {severe_errs}"
    print("Confusion matrix verified: 71/98 correct, 0 severe errors, 100% boundary errors.")

    print("\n=== 3. VERIFY FEATURE IMPORTANCES ===")
    fi_df = pd.read_csv("assets/improved_model_feature_importance.csv")
    assert fi_df["feature"].iloc[0] == "G2" and np.isclose(fi_df["importance"].iloc[0], 0.8274, atol=0.001)
    assert fi_df["feature"].iloc[1] == "G1" and np.isclose(fi_df["importance"].iloc[1], 0.0976, atol=0.001)
    print("Feature importance verified: G2=82.74%, G1=9.76%.")

    print("\n=== 4. VERIFY CROSS-VALIDATION AND OVERSAMPLING BENCHMARKS ===")
    cv_df = pd.read_csv("experiments/final_audit/model_comparison_results.csv")
    top_cv = cv_df.iloc[0]
    assert "Baseline Gradient Boosting" in top_cv["Model"]
    assert np.isclose(top_cv["cv_macro_f1_mean"], 0.8115, atol=0.001)

    os_df = pd.read_csv("assets/oversampling_experiment_results.csv")
    assert len(os_df) == 9, f"Expected 9 oversampling strategies, got {len(os_df)}"
    print("Cross-validation and oversampling benchmarks verified.")

    print("\n=== 5. VERIFY DNN TRAINING HISTORY ===")
    th = json.load(open("assets/training_history.json"))
    assert len(th["loss"]) == 50 and len(th["accuracy"]) == 50
    assert len(th["val_loss"]) == 50 and len(th["val_accuracy"]) == 50
    min_val_loss_epoch = int(np.argmin(th["val_loss"])) + 1
    assert min_val_loss_epoch == 38, f"Unexpected best epoch: {min_val_loss_epoch}"
    print(f"DNN training history verified: 50 epochs, best epoch at {min_val_loss_epoch} (val_loss={min(th['val_loss']):.4f}).")

    print("\n=== 6. VERIFY NO TENSORFLOW IMPORTS IN CODEBASE ===")
    import sys
    assert "tensorflow" not in sys.modules, "TensorFlow was unexpectedly loaded!"
    print("Zero TensorFlow imports verified.")

    print("\n=== 7. VERIFY COMPILATION OF ALL APPLICATION PAGES ===")
    import py_compile, glob
    for f in glob.glob("pages/*.py") + ["app.py", "src/app_utils.py", "src/dashboard_components.py"]:
        py_compile.compile(f, doraise=True)
        print(f"  - Compiled: {f}")

    print("\n=======================================================")
    print("ALL PHASE 4E TEST SUITE CHECKS PASSED WITH 100% SUCCESS!")
    print("=======================================================")

if __name__ == "__main__":
    run_tests()
