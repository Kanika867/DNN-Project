"""
EduPredict - Automated Model Improvement & Hyperparameter Tuning Pipeline
Performs 5-fold Stratified Cross-Validation on the combined development set (Train + Val = 551 samples)
strictly preventing data leakage by encapsulating preprocessing within Scikit-learn Pipelines.
Evaluates Random Forest, Extra Trees, Gradient Boosting, SVC, and Logistic Regression.
Selects the best model based on mean CV Macro F1 and consistency, then evaluates once on the outer test set.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

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

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    GridSearchCV,
    cross_validate
)
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Re-use project preprocessing components
from src.preprocessing import (
    load_dataset,
    convert_grades_to_classes,
    build_preprocessor,
    CLASS_MAPPING,
    TARGET_COLUMN
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'


def get_data_splits(data_path="data/student-por.csv"):
    """
    Reconstructs the exact outer train/val/test partitions from Phase 2.
    Verifies that the reconstructed outer test set matches artifacts/y_test.npy.
    Combines train (454) and val (97) into development set (551 samples).
    """
    df = load_dataset(data_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = convert_grades_to_classes(df[TARGET_COLUMN])

    indices = np.arange(len(df))

    # Phase 2 exact split: 70% train, 30% temp
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # 50/50 split of temp -> 15% val, 15% test
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # Combine train and val into development partition (551 samples)
    idx_dev = np.concatenate([idx_train, idx_val])
    X_dev = X.iloc[idx_dev].reset_index(drop=True)
    y_dev = y[idx_dev]

    X_test = X.iloc[idx_test].reset_index(drop=True)
    y_test = y[idx_test]

    # Verify against saved outer test array
    saved_y_test_path = os.path.join("artifacts", "y_test.npy")
    if os.path.exists(saved_y_test_path):
        saved_y_test = np.load(saved_y_test_path)
        assert np.array_equal(y_test, saved_y_test), "Reconstructed test targets do NOT match artifacts/y_test.npy!"
        print("Reconstructed outer test partition verified against artifacts/y_test.npy (100% exact match).")

    print(f"Dataset Partitions:")
    print(f"  - Total raw dataset:       {len(df)} samples")
    print(f"  - Combined Development:    {len(X_dev)} samples (Train: {len(idx_train)} + Val: {len(idx_val)})")
    print(f"  - Excluded Outer Test Set: {len(X_test)} samples (strictly reserved for final check)")

    return X_dev, y_dev, X_test, y_test


def run_model_experiments(X_dev, y_dev, cv_folds=5, random_state=42):
    """
    Automated model experimentation using Stratified 5-Fold Cross-Validation.
    Encapsulates preprocessing in a Pipeline for every model and candidate configuration.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        'macro_f1': 'f1_macro',
        'accuracy': 'accuracy',
        'macro_precision': 'precision_macro',
        'macro_recall': 'recall_macro'
    }

    results = []

    print("\n" + "=" * 80)
    print("STARTING AUTOMATED MODEL EXPERIMENTATION & HYPERPARAMETER TUNING")
    print("=" * 80)

    # -------------------------------------------------------------
    # 0. Existing Random Forest Baseline (as benchmark)
    # -------------------------------------------------------------
    print("\n[1/6] Evaluating Baseline Random Forest (300 trees, balanced)...")
    rf_baseline_pipe = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=random_state, n_jobs=-1))
    ])
    baseline_cv = cross_validate(rf_baseline_pipe, X_dev, y_dev, cv=cv, scoring=scoring, n_jobs=-1)
    results.append({
        "algorithm": "Random Forest (Baseline)",
        "search_type": "Fixed Baseline",
        "n_candidates_tested": 1,
        "best_params": {"n_estimators": 300, "class_weight": "balanced", "random_state": random_state},
        "mean_cv_macro_f1": float(np.mean(baseline_cv['test_macro_f1'])),
        "std_cv_macro_f1": float(np.std(baseline_cv['test_macro_f1'])),
        "mean_cv_accuracy": float(np.mean(baseline_cv['test_accuracy'])),
        "std_cv_accuracy": float(np.std(baseline_cv['test_accuracy'])),
        "mean_cv_macro_precision": float(np.mean(baseline_cv['test_macro_precision'])),
        "mean_cv_macro_recall": float(np.mean(baseline_cv['test_macro_recall'])),
        "best_estimator_pipeline": rf_baseline_pipe
    })
    print(f"  -> Baseline RF CV Macro F1: {results[-1]['mean_cv_macro_f1']:.4f} (+/- {results[-1]['std_cv_macro_f1']:.4f}), Accuracy: {results[-1]['mean_cv_accuracy']:.4f}")

    # -------------------------------------------------------------
    # 1. Random Forest (Tuned Search)
    # -------------------------------------------------------------
    print("\n[2/6] Tuning Random Forest Classifier (Randomized Search, 20 iterations)...")
    rf_pipe = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', RandomForestClassifier(random_state=random_state, n_jobs=-1))
    ])
    rf_param_dist = {
        'model__n_estimators': [200, 300, 500, 700],
        'model__max_depth': [None, 5, 8, 12, 16],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2'],
        'model__class_weight': ['balanced', 'balanced_subsample', None]
    }
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_param_dist, n_iter=20, cv=cv, scoring=scoring,
        refit='macro_f1', random_state=random_state, n_jobs=-1, verbose=0
    )
    rf_search.fit(X_dev, y_dev)
    best_idx = rf_search.best_index_
    results.append({
        "algorithm": "Random Forest (Tuned)",
        "search_type": "RandomizedSearchCV",
        "n_candidates_tested": 20,
        "best_params": {k.replace('model__', ''): v for k, v in rf_search.best_params_.items()},
        "mean_cv_macro_f1": float(rf_search.cv_results_['mean_test_macro_f1'][best_idx]),
        "std_cv_macro_f1": float(rf_search.cv_results_['std_test_macro_f1'][best_idx]),
        "mean_cv_accuracy": float(rf_search.cv_results_['mean_test_accuracy'][best_idx]),
        "std_cv_accuracy": float(rf_search.cv_results_['std_test_accuracy'][best_idx]),
        "mean_cv_macro_precision": float(rf_search.cv_results_['mean_test_macro_precision'][best_idx]),
        "mean_cv_macro_recall": float(rf_search.cv_results_['mean_test_macro_recall'][best_idx]),
        "best_estimator_pipeline": rf_search.best_estimator_
    })
    print(f"  -> Tuned RF CV Macro F1: {results[-1]['mean_cv_macro_f1']:.4f} (+/- {results[-1]['std_cv_macro_f1']:.4f}), Accuracy: {results[-1]['mean_cv_accuracy']:.4f}")

    # -------------------------------------------------------------
    # 2. Extra Trees Classifier
    # -------------------------------------------------------------
    print("\n[3/6] Tuning Extra Trees Classifier (Randomized Search, 20 iterations)...")
    et_pipe = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', ExtraTreesClassifier(random_state=random_state, n_jobs=-1))
    ])
    et_param_dist = {
        'model__n_estimators': [200, 300, 500],
        'model__max_depth': [None, 5, 8, 12, 16],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2'],
        'model__class_weight': ['balanced', 'balanced_subsample', None]
    }
    et_search = RandomizedSearchCV(
        et_pipe, et_param_dist, n_iter=20, cv=cv, scoring=scoring,
        refit='macro_f1', random_state=random_state, n_jobs=-1, verbose=0
    )
    et_search.fit(X_dev, y_dev)
    best_idx = et_search.best_index_
    results.append({
        "algorithm": "Extra Trees",
        "search_type": "RandomizedSearchCV",
        "n_candidates_tested": 20,
        "best_params": {k.replace('model__', ''): v for k, v in et_search.best_params_.items()},
        "mean_cv_macro_f1": float(et_search.cv_results_['mean_test_macro_f1'][best_idx]),
        "std_cv_macro_f1": float(et_search.cv_results_['std_test_macro_f1'][best_idx]),
        "mean_cv_accuracy": float(et_search.cv_results_['mean_test_accuracy'][best_idx]),
        "std_cv_accuracy": float(et_search.cv_results_['std_test_accuracy'][best_idx]),
        "mean_cv_macro_precision": float(et_search.cv_results_['mean_test_macro_precision'][best_idx]),
        "mean_cv_macro_recall": float(et_search.cv_results_['mean_test_macro_recall'][best_idx]),
        "best_estimator_pipeline": et_search.best_estimator_
    })
    print(f"  -> Extra Trees CV Macro F1: {results[-1]['mean_cv_macro_f1']:.4f} (+/- {results[-1]['std_cv_macro_f1']:.4f}), Accuracy: {results[-1]['mean_cv_accuracy']:.4f}")

    # -------------------------------------------------------------
    # 3. Gradient Boosting Classifier
    # -------------------------------------------------------------
    print("\n[4/6] Tuning Gradient Boosting Classifier (Randomized Search, 20 iterations)...")
    gb_pipe = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', GradientBoostingClassifier(random_state=random_state))
    ])
    gb_param_dist = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'model__max_depth': [1, 2, 3],
        'model__min_samples_leaf': [2, 4, 6]
    }
    gb_search = RandomizedSearchCV(
        gb_pipe, gb_param_dist, n_iter=20, cv=cv, scoring=scoring,
        refit='macro_f1', random_state=random_state, n_jobs=-1, verbose=0
    )
    gb_search.fit(X_dev, y_dev)
    best_idx = gb_search.best_index_
    results.append({
        "algorithm": "Gradient Boosting",
        "search_type": "RandomizedSearchCV",
        "n_candidates_tested": 20,
        "best_params": {k.replace('model__', ''): v for k, v in gb_search.best_params_.items()},
        "mean_cv_macro_f1": float(gb_search.cv_results_['mean_test_macro_f1'][best_idx]),
        "std_cv_macro_f1": float(gb_search.cv_results_['std_test_macro_f1'][best_idx]),
        "mean_cv_accuracy": float(gb_search.cv_results_['mean_test_accuracy'][best_idx]),
        "std_cv_accuracy": float(gb_search.cv_results_['std_test_accuracy'][best_idx]),
        "mean_cv_macro_precision": float(gb_search.cv_results_['mean_test_macro_precision'][best_idx]),
        "mean_cv_macro_recall": float(gb_search.cv_results_['mean_test_macro_recall'][best_idx]),
        "best_estimator_pipeline": gb_search.best_estimator_
    })
    print(f"  -> Gradient Boosting CV Macro F1: {results[-1]['mean_cv_macro_f1']:.4f} (+/- {results[-1]['std_cv_macro_f1']:.4f}), Accuracy: {results[-1]['mean_cv_accuracy']:.4f}")

    # -------------------------------------------------------------
    # 4. Support Vector Classifier (RBF Kernel)
    # -------------------------------------------------------------
    print("\n[5/6] Tuning Support Vector Classifier (Grid Search, 16 combinations)...")
    svc_pipe = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', SVC(random_state=random_state))
    ])
    svc_param_grid = {
        'model__C': [0.1, 1, 10, 50],
        'model__kernel': ['rbf'],
        'model__gamma': ['scale', 'auto'],
        'model__class_weight': ['balanced', None]
    }
    svc_search = GridSearchCV(
        svc_pipe, svc_param_grid, cv=cv, scoring=scoring,
        refit='macro_f1', n_jobs=-1, verbose=0
    )
    svc_search.fit(X_dev, y_dev)
    best_idx = svc_search.best_index_
    results.append({
        "algorithm": "Support Vector Classifier",
        "search_type": "GridSearchCV",
        "n_candidates_tested": 16,
        "best_params": {k.replace('model__', ''): v for k, v in svc_search.best_params_.items()},
        "mean_cv_macro_f1": float(svc_search.cv_results_['mean_test_macro_f1'][best_idx]),
        "std_cv_macro_f1": float(svc_search.cv_results_['std_test_macro_f1'][best_idx]),
        "mean_cv_accuracy": float(svc_search.cv_results_['mean_test_accuracy'][best_idx]),
        "std_cv_accuracy": float(svc_search.cv_results_['std_test_accuracy'][best_idx]),
        "mean_cv_macro_precision": float(svc_search.cv_results_['mean_test_macro_precision'][best_idx]),
        "mean_cv_macro_recall": float(svc_search.cv_results_['mean_test_macro_recall'][best_idx]),
        "best_estimator_pipeline": svc_search.best_estimator_
    })
    print(f"  -> SVC CV Macro F1: {results[-1]['mean_cv_macro_f1']:.4f} (+/- {results[-1]['std_cv_macro_f1']:.4f}), Accuracy: {results[-1]['mean_cv_accuracy']:.4f}")

    # -------------------------------------------------------------
    # 5. Logistic Regression Baseline
    # -------------------------------------------------------------
    print("\n[6/6] Tuning Logistic Regression (Grid Search, 10 combinations)...")
    lr_pipe = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', LogisticRegression(random_state=random_state))
    ])
    lr_param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 50],
        'model__class_weight': ['balanced', None],
        'model__solver': ['lbfgs'],
        'model__max_iter': [3000]
    }
    lr_search = GridSearchCV(
        lr_pipe, lr_param_grid, cv=cv, scoring=scoring,
        refit='macro_f1', n_jobs=-1, verbose=0
    )
    lr_search.fit(X_dev, y_dev)
    best_idx = lr_search.best_index_
    results.append({
        "algorithm": "Logistic Regression",
        "search_type": "GridSearchCV",
        "n_candidates_tested": 10,
        "best_params": {k.replace('model__', ''): v for k, v in lr_search.best_params_.items()},
        "mean_cv_macro_f1": float(lr_search.cv_results_['mean_test_macro_f1'][best_idx]),
        "std_cv_macro_f1": float(lr_search.cv_results_['std_test_macro_f1'][best_idx]),
        "mean_cv_accuracy": float(lr_search.cv_results_['mean_test_accuracy'][best_idx]),
        "std_cv_accuracy": float(lr_search.cv_results_['std_test_accuracy'][best_idx]),
        "mean_cv_macro_precision": float(lr_search.cv_results_['mean_test_macro_precision'][best_idx]),
        "mean_cv_macro_recall": float(lr_search.cv_results_['mean_test_macro_recall'][best_idx]),
        "best_estimator_pipeline": lr_search.best_estimator_
    })
    print(f"  -> Logistic Regression CV Macro F1: {results[-1]['mean_cv_macro_f1']:.4f} (+/- {results[-1]['std_cv_macro_f1']:.4f}), Accuracy: {results[-1]['mean_cv_accuracy']:.4f}")

    return results


def select_and_evaluate_best_model(results, X_dev, y_dev, X_test, y_test, output_dir="assets", models_dir="models"):
    """
    Ranks candidates by Mean CV Macro F1 and consistency (standard deviation).
    Refits winning pipeline on all 551 development records.
    Evaluates once on outer test set (98 records).
    Saves all metrics, predictions, feature importances, and confusion matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Sort candidates primarily by Mean CV Macro F1 (descending), secondarily by std (ascending)
    results_sorted = sorted(results, key=lambda x: (x["mean_cv_macro_f1"], -x["std_cv_macro_f1"]), reverse=True)
    best_candidate = results_sorted[0]

    print("\n" + "=" * 90)
    print("5-FOLD STRATIFIED CROSS-VALIDATION SUMMARY TABLE (DEVELOPMENT SET, N=551)")
    print("=" * 90)
    print(f"{'Algorithm':<28} | {'CV Macro F1':<17} | {'CV Accuracy':<17} | {'CV Precision':<12} | {'CV Recall':<10}")
    print("-" * 90)
    for r in results_sorted:
        f1_str = f"{r['mean_cv_macro_f1']:.4f} (+/- {r['std_cv_macro_f1']:.4f})"
        acc_str = f"{r['mean_cv_accuracy']:.4f} (+/- {r['std_cv_accuracy']:.4f})"
        prec_str = f"{r['mean_cv_macro_precision']:.4f}"
        rec_str = f"{r['mean_cv_macro_recall']:.4f}"
        print(f"{r['algorithm']:<28} | {f1_str:<17} | {acc_str:<17} | {prec_str:<12} | {rec_str:<10}")
    print("=" * 90)

    print(f"\nWinning Candidate Model Selected: {best_candidate['algorithm']}")
    print(f"  Best Hyperparameters: {best_candidate['best_params']}")
    print(f"  Cross-Validation Macro F1: {best_candidate['mean_cv_macro_f1']:.4f} (+/- {best_candidate['std_cv_macro_f1']:.4f})")

    # 1. Refit entire pipeline on full development partition (551 samples)
    print(f"\nRefitting {best_candidate['algorithm']} pipeline on full development partition (N=551)...")
    winning_pipe = best_candidate["best_estimator_pipeline"]
    winning_pipe.fit(X_dev, y_dev)
    print("Refitting complete.")

    # 2. Evaluate once on outer test set (N=98)
    test_preds = winning_pipe.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, test_preds, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_test, test_preds, average='weighted', zero_division=0)
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(y_test, test_preds, average=None, zero_division=0)

    target_names = [CLASS_MAPPING[i]["name"] for i in range(len(CLASS_MAPPING))]
    cm = confusion_matrix(y_test, test_preds)
    clf_report_dict = classification_report(y_test, test_preds, target_names=target_names, output_dict=True, zero_division=0)

    print("\n" + "=" * 80)
    print(f"EVALUATION OF BEST IMPROVED MODEL ON OUTER TEST SET (N=98)")
    print("=" * 80)
    print(f"  Accuracy:        {test_acc:.4f} ({test_acc:.2%})")
    print(f"  Macro Precision: {macro_p:.4f}")
    print(f"  Macro Recall:    {macro_r:.4f}")
    print(f"  Macro F1-Score:  {macro_f1:.4f}  <-- Primary Selection Metric")
    print(f"  Weighted F1:     {weighted_f1:.4f}\n")

    print("Per-Class Metrics on Test Set:")
    for idx, name in enumerate(target_names):
        print(f"  Class {idx} ({name:<9}): Precision = {per_p[idx]:.4f}, Recall = {per_r[idx]:.4f}, F1 = {per_f1[idx]:.4f}, Support = {int(per_sup[idx])}")
    print()

    print("Confusion Matrix:")
    print(cm)
    print("=" * 80)

    # 3. Save Winning Model Pipeline
    model_save_path = os.path.join(models_dir, "best_improved_model.joblib")
    joblib.dump(winning_pipe, model_save_path)
    print(f"\nSaved complete winning pipeline to: {model_save_path}")

    # 4. Save Predictions
    preds_save_path = os.path.join(output_dir, "improved_model_predictions.npy")
    np.save(preds_save_path, test_preds)
    print(f"Saved test predictions to: {preds_save_path}")

    # 5. Save Experiment Results Summary (CSV and JSON)
    exp_records = []
    for r in results_sorted:
        exp_records.append({
            "algorithm": r["algorithm"],
            "search_type": r["search_type"],
            "candidates_tested": r["n_candidates_tested"],
            "mean_cv_macro_f1": r["mean_cv_macro_f1"],
            "std_cv_macro_f1": r["std_cv_macro_f1"],
            "mean_cv_accuracy": r["mean_cv_accuracy"],
            "std_cv_accuracy": r["std_cv_accuracy"],
            "mean_cv_macro_precision": r["mean_cv_macro_precision"],
            "mean_cv_macro_recall": r["mean_cv_macro_recall"],
            "best_params": json.dumps(r["best_params"])
        })
    exp_df = pd.DataFrame(exp_records)
    exp_csv_path = os.path.join(output_dir, "model_experiment_results.csv")
    exp_df.to_csv(exp_csv_path, index=False)
    print(f"Saved experiment results table to: {exp_csv_path}")

    exp_json_path = os.path.join(output_dir, "model_experiment_results.json")
    with open(exp_json_path, "w", encoding="utf-8") as f:
        json.dump([
            {k: v for k, v in r.items() if k != "best_estimator_pipeline"}
            for r in results_sorted
        ], f, indent=2)
    print(f"Saved experiment results JSON to: {exp_json_path}")

    # 6. Save Improved Model Metrics JSON
    metrics_summary = {
        "model_name": best_candidate["algorithm"],
        "hyperparameters": best_candidate["best_params"],
        "cross_validation_metrics_5fold": {
            "mean_macro_f1": best_candidate["mean_cv_macro_f1"],
            "std_macro_f1": best_candidate["std_cv_macro_f1"],
            "mean_accuracy": best_candidate["mean_cv_accuracy"],
            "std_accuracy": best_candidate["std_cv_accuracy"],
            "mean_macro_precision": best_candidate["mean_cv_macro_precision"],
            "mean_macro_recall": best_candidate["mean_cv_macro_recall"]
        },
        "test_metrics_outer": {
            "accuracy": float(test_acc),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(macro_r),
            "weighted_f1": float(weighted_f1),
            "per_class": {
                name: {
                    "class_index": idx,
                    "precision": float(per_p[idx]),
                    "recall": float(per_r[idx]),
                    "f1_score": float(per_f1[idx]),
                    "support": int(per_sup[idx])
                }
                for idx, name in enumerate(target_names)
            }
        },
        "confusion_matrix": cm.tolist()
    }
    improved_metrics_path = os.path.join(output_dir, "improved_model_metrics.json")
    with open(improved_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved improved model metrics JSON to: {improved_metrics_path}")

    # 7. Confusion Matrix Plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=target_names, yticklabels=target_names,
        linewidths=0.8, linecolor="#eeeeee"
    )
    plt.title(f"Best Improved Model ({best_candidate['algorithm']})\nTest Confusion Matrix", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight='bold')
    plt.ylabel("True Label", fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, "improved_model_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved improved confusion matrix plot to: {cm_plot_path}")

    # 8. Feature Importance / Permutation Importance
    print("\nComputing feature importances...")
    fitted_prep = winning_pipe.named_steps['preprocessor']
    fitted_model = winning_pipe.named_steps['model']
    feature_names = list(fitted_prep.get_feature_names_out())

    if hasattr(fitted_model, "feature_importances_"):
        importances = fitted_model.feature_importances_
        fi_method = "Gini Feature Importance"
    else:
        # Use permutation importance on development set
        X_dev_trans = fitted_prep.transform(X_dev)
        perm = permutation_importance(fitted_model, X_dev_trans, y_dev, n_repeats=10, random_state=random_state, scoring='f1_macro', n_jobs=-1)
        importances = perm.importances_mean
        fi_method = "Permutation Feature Importance"

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    fi_df["rank"] = fi_df.index + 1

    fi_csv_path = os.path.join(output_dir, "improved_model_feature_importance.csv")
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"Saved feature importance table ({fi_method}) to: {fi_csv_path}")

    plt.figure(figsize=(10, 7))
    top_fi = fi_df.head(20).sort_values(by="importance", ascending=True)
    colors = ["#8e44ad" if "G1" in f or "G2" in f else "#95a5a6" for f in top_fi["feature"]]
    bars = plt.barh(top_fi["feature"], top_fi["importance"], color=colors, edgecolor="#333333", linewidth=0.6)
    plt.title(f"Improved Model ({best_candidate['algorithm']})\nTop 20 {fi_method}", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Importance Score", fontsize=11, fontweight='bold')
    plt.ylabel("Transformed Feature", fontsize=11)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va='center', fontsize=8.5, fontweight='bold')
    plt.xlim(0, max(top_fi["importance"]) * 1.15)
    plt.tight_layout()
    fi_plot_path = os.path.join(output_dir, "improved_model_feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to: {fi_plot_path}")

    print("\n" + "=" * 80)
    print("MODEL IMPROVEMENT EXPERIMENTATION FULLY COMPLETED!")
    print("=" * 80)

    return best_candidate, metrics_summary


def main():
    X_dev, y_dev, X_test, y_test = get_data_splits()
    results = run_model_experiments(X_dev, y_dev)
    select_and_evaluate_best_model(results, X_dev, y_dev, X_test, y_test)


if __name__ == "__main__":
    main()
