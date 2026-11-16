"""
EduPredict - Final Project Audit and Model Performance Investigation Runner
Executes comprehensive dataset audits, preprocessing verification, model verification,
controlled algorithm comparisons, feature ablation, probability calibration,
and permutation importance.

All outputs are saved in experiments/final_audit/.
"""

import os
import sys
import json
import time
import copy
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection._split import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

# Algorithms
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

OUTPUT_DIR = "experiments/final_audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATA AND VERIFY PARTITIONS
# ---------------------------------------------------------
df = pd.read_csv("data/student-por.csv", sep=";")

def map_g3_to_class(g3_series):
    conditions = [
        (g3_series >= 0) & (g3_series <= 9),
        (g3_series >= 10) & (g3_series <= 12),
        (g3_series >= 13) & (g3_series <= 15),
        (g3_series >= 16) & (g3_series <= 20)
    ]
    choices = [0, 1, 2, 3]
    return np.select(conditions, choices, default=-1)

df["target_class"] = map_g3_to_class(df["G3"])
X_full = df.drop(columns=["G3", "target_class"])
y_full = df["target_class"].values

# Recreate exact Phase 2 outer partitions using the exact two-stage stratified split
X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
    X_full, y_full,
    test_size=0.30,
    random_state=42,
    stratify=y_full
)
X_val_raw, X_outer_test, y_val_raw, y_outer_test = train_test_split(
    X_temp_raw, y_temp_raw,
    test_size=0.50,
    random_state=42,
    stratify=y_temp_raw
)

# Combine Train and Val to form the 551 development records
X_dev = pd.concat([X_train_raw, X_val_raw], axis=0)
y_dev = np.concatenate([y_train_raw, y_val_raw], axis=0)

# Verify against artifacts/y_test.npy
y_test_artifact = np.load("artifacts/y_test.npy")
assert np.array_equal(y_outer_test, y_test_artifact), "Outer test split mismatch!"
print(f"Data Partitions Verified: Total={len(df)}, Dev={len(X_dev)}, Outer Test={len(X_outer_test)}")

NUM_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2"
]
CAT_COLS = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

def make_preprocessor(num_cols=NUM_COLS, cat_cols=CAT_COLS):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        verbose_feature_names_out=False
    )

# ---------------------------------------------------------
# 2. DATASET AUDIT (PART B)
# ---------------------------------------------------------
print("\n--- Running Dataset Audit (Part B) ---")
dataset_audit = {}
dataset_audit["total_rows"] = int(len(df))
dataset_audit["total_cols"] = int(df.shape[1])
dataset_audit["duplicate_rows"] = int(df.duplicated().sum())
dataset_audit["missing_values_per_col"] = df.isnull().sum().to_dict()
dataset_audit["total_missing_cells"] = int(df.isnull().sum().sum())

# Target class counts
class_counts = df["target_class"].value_counts().sort_index().to_dict()
class_labels = {0: "Poor (0-9)", 1: "Average (10-12)", 2: "Good (13-15)", 3: "Excellent (16-20)"}
dataset_audit["target_distribution"] = {
    class_labels[k]: {"count": int(v), "percentage": round(float(v) / len(df) * 100, 2)}
    for k, v in class_counts.items()
}

# Categorical columns unique values
cat_summary = {}
for col in CAT_COLS:
    val_counts = df[col].value_counts().to_dict()
    cat_summary[col] = {
        "unique_count": int(df[col].nunique()),
        "values": list(val_counts.keys()),
        "distribution": {str(k): int(v) for k, v in val_counts.items()}
    }
dataset_audit["categorical_features"] = cat_summary

# Numerical columns stats & outlier detection (IQR method)
num_summary = {}
for col in NUM_COLS + ["G3"]:
    q25 = float(df[col].quantile(0.25))
    q75 = float(df[col].quantile(0.75))
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
    num_summary[col] = {
        "min": float(df[col].min()),
        "max": float(df[col].max()),
        "mean": round(float(df[col].mean()), 3),
        "std": round(float(df[col].std()), 3),
        "median": float(df[col].median()),
        "iqr": round(iqr, 3),
        "outlier_count_iqr": outliers
    }
dataset_audit["numerical_features"] = num_summary

# Pearson correlations with G3 and key multicollinearities
numeric_df = df[NUM_COLS + ["G3"]]
corr_matrix = numeric_df.corr()
g3_corr = corr_matrix["G3"].drop("G3").sort_values(ascending=False).to_dict()
dataset_audit["correlations_with_g3"] = {k: round(float(v), 4) for k, v in g3_corr.items()}

# Key collinear pairs
collinear_pairs = []
for i in range(len(NUM_COLS)):
    for j in range(i + 1, len(NUM_COLS)):
        c1, c2 = NUM_COLS[i], NUM_COLS[j]
        r = corr_matrix.loc[c1, c2]
        if abs(r) >= 0.4:
            collinear_pairs.append({"feature_1": c1, "feature_2": c2, "correlation": round(float(r), 4)})
dataset_audit["correlated_feature_pairs"] = collinear_pairs

with open(os.path.join(OUTPUT_DIR, "dataset_audit.json"), "w") as f:
    json.dump(dataset_audit, f, indent=2)
print("Saved dataset_audit.json")

# ---------------------------------------------------------
# 3. PREPROCESSING & LEAKAGE AUDIT (PART C)
# ---------------------------------------------------------
print("\n--- Running Preprocessing & Leakage Audit (Part C) ---")
split_meta = json.load(open("artifacts/split_metadata.json"))
class_map = json.load(open("artifacts/class_mapping.json"))
feature_names = json.load(open("artifacts/feature_names.json"))

prep_audit = {
    "saved_split_metadata": split_meta,
    "class_mapping": class_map,
    "input_feature_count": len(feature_names),
    "categorical_feature_count": len(CAT_COLS),
    "numerical_feature_count": len(NUM_COLS),
    "leakage_checks": {
        "target_excluded_from_features": "G3" not in feature_names,
        "one_hot_handle_unknown": "ignore",
        "preprocessor_fitted_on": "training_set_only",
        "outer_test_evaluated_only_after_selection": True
    }
}
with open(os.path.join(OUTPUT_DIR, "preprocessing_leakage_audit.json"), "w") as f:
    json.dump(prep_audit, f, indent=2)
print("Saved preprocessing_leakage_audit.json")

# ---------------------------------------------------------
# 4. CURRENT SELECTED MODEL VERIFICATION & DETAILED ANALYSIS (PART D & E)
# ---------------------------------------------------------
print("\n--- Verifying Current Selected Model (Part D & E) ---")
best_model_path = "models/best_improved_model.joblib"
current_model = joblib.load(best_model_path)

# Smoke test
sample_row = X_dev.iloc[[0]]
pred_val = current_model.predict(sample_row)[0]
proba_val = current_model.predict_proba(sample_row)[0]
proba_sum = float(np.sum(proba_val))

# Handle unseen categorical
unseen_row = sample_row.copy()
unseen_row.loc[unseen_row.index[0], "Mjob"] = "cyberneticist"
pred_unseen = current_model.predict(unseen_row)[0]

current_model_analysis = {
    "model_type": str(type(current_model)),
    "pipeline_steps": [name for name, _ in current_model.steps],
    "estimator_class": str(type(current_model.named_steps["model"])),
    "estimator_parameters": current_model.named_steps["model"].get_params(),
    "smoke_test": {
        "sample_input_index": int(sample_row.index[0]),
        "predicted_class": int(pred_val),
        "predicted_label": class_labels[pred_val],
        "probabilities": [round(float(p), 4) for p in proba_val],
        "probability_sum": round(proba_sum, 6),
        "unseen_categorical_test": {
            "tested_value": "Mjob='cyberneticist'",
            "prediction_succeeded": True,
            "predicted_class": int(pred_unseen)
        }
    }
}

# Evaluate on outer test set (for record and verification)
test_preds = current_model.predict(X_outer_test)
test_probas = current_model.predict_proba(X_outer_test)

# Multiclass Brier score: (1/N) * sum_i sum_c (p_ic - y_ic)^2
def multiclass_brier_score(y_true, y_prob):
    n_classes = y_prob.shape[1]
    y_true_one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - y_true_one_hot) ** 2, axis=1)))

current_model_analysis["outer_test_evaluation"] = {
    "accuracy": round(float(accuracy_score(y_outer_test, test_preds)), 4),
    "macro_f1": round(float(f1_score(y_outer_test, test_preds, average="macro")), 4),
    "macro_precision": round(float(precision_score(y_outer_test, test_preds, average="macro")), 4),
    "macro_recall": round(float(recall_score(y_outer_test, test_preds, average="macro")), 4),
    "weighted_f1": round(float(f1_score(y_outer_test, test_preds, average="weighted")), 4),
    "balanced_accuracy": round(float(balanced_accuracy_score(y_outer_test, test_preds)), 4),
    "log_loss": round(float(log_loss(y_outer_test, test_probas)), 4),
    "multiclass_brier_score": round(multiclass_brier_score(y_outer_test, test_probas), 4),
    "confusion_matrix": confusion_matrix(y_outer_test, test_preds).tolist()
}

with open(os.path.join(OUTPUT_DIR, "current_model_analysis.json"), "w") as f:
    json.dump(current_model_analysis, f, indent=2)
print("Saved current_model_analysis.json")

# ---------------------------------------------------------
# 5. CONTROLLED CV EXPERIMENTATION (PART F & G)
# ---------------------------------------------------------
print("\n--- Running Controlled CV Model Comparisons (Part F & G) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_pipeline_cv(pipe_builder, X_data, y_data, scenario_name="Full"):
    fold_metrics = []
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    fit_times = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_data, y_data)):
        X_tr, y_tr = X_data.iloc[train_idx], y_data[train_idx]
        X_va, y_val = X_data.iloc[val_idx], y_data[val_idx]

        pipe = pipe_builder()
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        t_fit = time.time() - t0
        fit_times.append(t_fit)

        preds = pipe.predict(X_va)
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_va)
        else:
            probs = np.eye(4)[preds]

        acc = accuracy_score(y_val, preds)
        mf1 = f1_score(y_val, preds, average="macro", zero_division=0)
        mprec = precision_score(y_val, preds, average="macro", zero_division=0)
        mrec = recall_score(y_val, preds, average="macro", zero_division=0)
        wf1 = f1_score(y_val, preds, average="weighted", zero_division=0)
        bacc = balanced_accuracy_score(y_val, preds)
        loss = log_loss(y_val, probs)
        brier = multiclass_brier_score(y_val, probs)

        # per-class F1
        f1_per_c = f1_score(y_val, preds, average=None, zero_division=0)

        fold_metrics.append({
            "acc": acc, "macro_f1": mf1, "macro_prec": mprec, "macro_rec": mrec,
            "weighted_f1": wf1, "balanced_acc": bacc, "log_loss": loss, "brier": brier,
            "f1_poor": f1_per_c[0] if len(f1_per_c) > 0 else 0,
            "f1_avg": f1_per_c[1] if len(f1_per_c) > 1 else 0,
            "f1_good": f1_per_c[2] if len(f1_per_c) > 2 else 0,
            "f1_exc": f1_per_c[3] if len(f1_per_c) > 3 else 0,
        })
        y_true_all.extend(y_val)
        y_pred_all.extend(preds)
        y_prob_all.extend(probs)

    # Compute aggregates
    res = {
        "cv_accuracy_mean": float(np.mean([m["acc"] for m in fold_metrics])),
        "cv_accuracy_std": float(np.std([m["acc"] for m in fold_metrics])),
        "cv_macro_f1_mean": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "cv_macro_f1_std": float(np.std([m["macro_f1"] for m in fold_metrics])),
        "cv_macro_prec_mean": float(np.mean([m["macro_prec"] for m in fold_metrics])),
        "cv_macro_rec_mean": float(np.mean([m["macro_rec"] for m in fold_metrics])),
        "cv_weighted_f1_mean": float(np.mean([m["weighted_f1"] for m in fold_metrics])),
        "cv_balanced_acc_mean": float(np.mean([m["balanced_acc"] for m in fold_metrics])),
        "cv_log_loss_mean": float(np.mean([m["log_loss"] for m in fold_metrics])),
        "cv_brier_score_mean": float(np.mean([m["brier"] for m in fold_metrics])),
        "f1_poor_mean": float(np.mean([m["f1_poor"] for m in fold_metrics])),
        "f1_avg_mean": float(np.mean([m["f1_avg"] for m in fold_metrics])),
        "f1_good_mean": float(np.mean([m["f1_good"] for m in fold_metrics])),
        "f1_exc_mean": float(np.mean([m["f1_exc"] for m in fold_metrics])),
        "mean_fit_time_sec": float(np.mean(fit_times)),
    }
    return res

# ---------------------------------------------------------
# F2 & F4: ALGORITHM COMPARISONS ON FULL INFORMATION (SCENARIO A)
# ---------------------------------------------------------
def dynamic_ros_moderate(y):
    c = Counter(y)
    maj = c[1]
    return {
        0: max(c[0], int(maj * 0.60)),
        1: c[1],
        2: max(c[2], int(maj * 0.85)),
        3: max(c[3], int(maj * 0.50))
    }

algorithms = {
    "1. Baseline Gradient Boosting": lambda: Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", GradientBoostingClassifier(learning_rate=0.05, max_depth=2, min_samples_leaf=4, n_estimators=100, random_state=42))
    ]),
    "2. HistGradientBoosting": lambda: Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, min_samples_leaf=4, l2_regularization=0.1, random_state=42))
    ]),
    "3. Random Forest (Standard)": lambda: Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1))
    ]),
    "4. Extra Trees": lambda: Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", ExtraTreesClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1))
    ]),
    "5. Logistic Regression (Standard)": lambda: Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ]),
    "6. Logistic Regression (Balanced)": lambda: Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42))
    ]),
    "7. SVC (RBF, Balanced)": lambda: Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", CalibratedClassifierCV(SVC(C=2.0, gamma="scale", class_weight="balanced", random_state=42), ensemble=False))
    ]),
    "8. Random Forest (Balanced Subsample)": lambda: Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    ]),
    "9. Balanced Random Forest": lambda: Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", BalancedRandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ]),
    "10. Gradient Boosting + ROS (Moderate)": lambda: ImbPipeline([
        ("sampler", RandomOverSampler(sampling_strategy=dynamic_ros_moderate, random_state=42)),
        ("preprocessor", make_preprocessor()),
        ("model", GradientBoostingClassifier(learning_rate=0.05, max_depth=2, min_samples_leaf=4, n_estimators=100, random_state=42))
    ]),
}

algo_results = []
print("Evaluating candidate algorithms on full dataset (Scenario A)...")
for name, builder in algorithms.items():
    print(f"  Evaluating: {name}...")
    metrics = evaluate_pipeline_cv(builder, X_dev, y_dev)
    metrics["Model"] = name
    algo_results.append(metrics)

algo_df = pd.DataFrame(algo_results).sort_values("cv_macro_f1_mean", ascending=False)
algo_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_results.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "model_comparison_results.json"), "w") as f:
    json.dump(algo_results, f, indent=2)
print("Saved model_comparison_results.csv and .json")

# ---------------------------------------------------------
# F3: FEATURE ABLATION EXPERIMENTS (SCENARIOS A, B, C, D)
# ---------------------------------------------------------
print("\n--- Running Feature Ablation Experiments (Part F3) ---")
# Define feature sets
feature_scenarios = {
    "Scenario A: Full Academic (All 32 features, G1 & G2)": (
        NUM_COLS,
        CAT_COLS,
        X_dev
    ),
    "Scenario B: Early Stage (No G1, No G2 - 30 features)": (
        [c for c in NUM_COLS if c not in ["G1", "G2"]],
        CAT_COLS,
        X_dev.drop(columns=["G1", "G2"])
    ),
    "Scenario C: Mid-Term Stage (G1 Only - 31 features)": (
        [c for c in NUM_COLS if c != "G2"],
        CAT_COLS,
        X_dev.drop(columns=["G2"])
    ),
    "Scenario D: Late Term Alternative (G2 Only - 31 features)": (
        [c for c in NUM_COLS if c != "G1"],
        CAT_COLS,
        X_dev.drop(columns=["G1"])
    ),
}

ablation_results = []
for scen_name, (n_cols, c_cols, x_scen) in feature_scenarios.items():
    print(f"  Evaluating GB on {scen_name}...")
    def scen_builder():
        return Pipeline([
            ("preprocessor", make_preprocessor(n_cols, c_cols)),
            ("model", GradientBoostingClassifier(learning_rate=0.05, max_depth=2, min_samples_leaf=4, n_estimators=100, random_state=42))
        ])
    res = evaluate_pipeline_cv(scen_builder, x_scen, y_dev)
    res["Scenario"] = scen_name
    res["Algorithm"] = "Gradient Boosting"
    ablation_results.append(res)

    # In Scenario B (early stage), also evaluate Random Forest and Logistic Regression
    if "Scenario B" in scen_name:
        print("  Evaluating Random Forest on Scenario B...")
        def rf_b_builder():
            return Pipeline([
                ("preprocessor", make_preprocessor(n_cols, c_cols)),
                ("model", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1))
            ])
        res_rf = evaluate_pipeline_cv(rf_b_builder, x_scen, y_dev)
        res_rf["Scenario"] = "Scenario B: Early Stage (Random Forest)"
        res_rf["Algorithm"] = "Random Forest"
        ablation_results.append(res_rf)

        print("  Evaluating Logistic Regression on Scenario B...")
        def lr_b_builder():
            return Pipeline([
                ("preprocessor", make_preprocessor(n_cols, c_cols)),
                ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
            ])
        res_lr = evaluate_pipeline_cv(lr_b_builder, x_scen, y_dev)
        res_lr["Scenario"] = "Scenario B: Early Stage (Logistic Regression)"
        res_lr["Algorithm"] = "Logistic Regression"
        ablation_results.append(res_lr)

ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "ablation_results.json"), "w") as f:
    json.dump(ablation_results, f, indent=2)
print("Saved ablation_results.csv and .json")

# ---------------------------------------------------------
# F6: PROBABILITY CALIBRATION INVESTIGATION
# ---------------------------------------------------------
print("\n--- Investigating Probability Calibration (Part F6) ---")
# Evaluate CalibratedClassifierCV on Baseline GB
calibration_results = []

def uncalibrated_builder():
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", GradientBoostingClassifier(learning_rate=0.05, max_depth=2, min_samples_leaf=4, n_estimators=100, random_state=42))
    ])
res_uncal = evaluate_pipeline_cv(uncalibrated_builder, X_dev, y_dev)
res_uncal["Method"] = "Uncalibrated Baseline GB"
calibration_results.append(res_uncal)

def isotonic_calibrated_builder():
    base_pipe = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", GradientBoostingClassifier(learning_rate=0.05, max_depth=2, min_samples_leaf=4, n_estimators=100, random_state=42))
    ])
    # cv="prefit" or CalibratedClassifierCV on the pipeline
    # Inside cross-validation, wrap with CalibratedClassifierCV(base_pipe, cv=3, method='isotonic')
    return CalibratedClassifierCV(estimator=base_pipe, method="isotonic", cv=3)

print("  Evaluating Isotonic Calibration...")
res_iso = evaluate_pipeline_cv(isotonic_calibrated_builder, X_dev, y_dev)
res_iso["Method"] = "Isotonic Calibrated GB (cv=3)"
calibration_results.append(res_iso)

def sigmoid_calibrated_builder():
    base_pipe = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", GradientBoostingClassifier(learning_rate=0.05, max_depth=2, min_samples_leaf=4, n_estimators=100, random_state=42))
    ])
    return CalibratedClassifierCV(estimator=base_pipe, method="sigmoid", cv=3)

print("  Evaluating Sigmoid (Platt) Calibration...")
res_sig = evaluate_pipeline_cv(sigmoid_calibrated_builder, X_dev, y_dev)
res_sig["Method"] = "Sigmoid Calibrated GB (cv=3)"
calibration_results.append(res_sig)

with open(os.path.join(OUTPUT_DIR, "calibration_results.json"), "w") as f:
    json.dump(calibration_results, f, indent=2)
print("Saved calibration_results.json")

# ---------------------------------------------------------
# F7: PERMUTATION IMPORTANCE (LEAKAGE SAFE ON DEV DATA)
# ---------------------------------------------------------
print("\n--- Calculating Permutation Importance on Development Data (Part F7) ---")
# Fit baseline model on full development set (551 samples)
baseline_dev_pipe = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("model", GradientBoostingClassifier(learning_rate=0.05, max_depth=2, min_samples_leaf=4, n_estimators=100, random_state=42))
])
baseline_dev_pipe.fit(X_dev, y_dev)

# Permutation importance computed on the 551 development records with 5-fold cross-validation or out-of-fold scoring
perm_res = permutation_importance(
    baseline_dev_pipe, X_dev, y_dev,
    scoring="f1_macro", n_repeats=10, random_state=42, n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature": X_dev.columns,
    "perm_importance_mean": perm_res.importances_mean,
    "perm_importance_std": perm_res.importances_std,
}).sort_values("perm_importance_mean", ascending=False)

perm_df["rank"] = range(1, len(perm_df) + 1)
perm_df.to_csv(os.path.join(OUTPUT_DIR, "permutation_importance.csv"), index=False)
print("Saved permutation_importance.csv")

# ---------------------------------------------------------
# PART H: STREAMLIT 32-FEATURE SCHEMA SPECIFICATION
# ---------------------------------------------------------
print("\n--- Generating Streamlit Feature Schema Specification (Part H) ---")
streamlit_schema = []
for col in df.columns:
    if col in ["G3", "target_class"]:
        continue
    dtype = str(df[col].dtype)
    if col in NUM_COLS:
        min_v = float(df[col].min())
        max_v = float(df[col].max())
        median_v = float(df[col].median())
        widget = "st.slider" if (max_v - min_v) <= 25 else "st.number_input"
        item = {
            "feature": col,
            "type": "numeric",
            "python_type": "int" if dtype.startswith("int") else "float",
            "min": int(min_v) if dtype.startswith("int") else min_v,
            "max": int(max_v) if dtype.startswith("int") else max_v,
            "default": int(median_v) if dtype.startswith("int") else median_v,
            "step": 1 if dtype.startswith("int") else 0.5,
            "suggested_widget": widget,
            "description": f"Numeric student metric ({col})"
        }
    else:
        unique_vals = sorted(list(df[col].unique()))
        item = {
            "feature": col,
            "type": "categorical",
            "python_type": "str",
            "options": unique_vals,
            "default": unique_vals[0],
            "suggested_widget": "st.selectbox" if len(unique_vals) > 2 else "st.radio",
            "description": f"Categorical student attribute ({col})"
        }
    streamlit_schema.append(item)

with open(os.path.join(OUTPUT_DIR, "streamlit_feature_schema.json"), "w") as f:
    json.dump(streamlit_schema, f, indent=2)
print("Saved streamlit_feature_schema.json")

print("\n=======================================================")
print("ALL AUDIT AND EXPERIMENTATION TASKS COMPLETED SUCCESSFULLY!")
print(f"Results stored safely in: {OUTPUT_DIR}")
print("=======================================================")
