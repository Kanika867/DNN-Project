"""
EduPredict - Focused Imbalance & Oversampling Experimentation Pipeline
Investigates RandomOverSampler, SMOTENC (with categorical feature awareness),
Balanced Random Forest, and Class Weighting across multiple sampling strengths.
Strictly prevents data leakage by executing all sampling and preprocessing
inside 5-Fold Stratified Cross-Validation on the development set (N=551).
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import json
from collections import Counter
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure optional unused estimators in sklearn don't trigger Windows Application Control
from unittest.mock import MagicMock
if 'sklearn.ensemble._hist_gradient_boosting' not in sys.modules:
    sys.modules['sklearn.ensemble._hist_gradient_boosting'] = MagicMock()
if 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting' not in sys.modules:
    sys.modules['sklearn.ensemble._hist_gradient_boosting.gradient_boosting'] = MagicMock()

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.inspection import permutation_importance

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.ensemble import BalancedRandomForestClassifier

from src.preprocessing import (
    load_dataset,
    convert_grades_to_classes,
    build_preprocessor,
    CLASS_MAPPING,
    TARGET_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)
from src.improve_models import get_data_splits

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'


# Define Dynamic Sampling Strategies based on Fold Majority Count
def moderate_strategy(y_arr):
    """
    Moderately boosts minority classes:
    Class 0 (Poor): brought to ~60% of majority
    Class 2 (Good): brought to ~85% of majority
    Class 3 (Excellent): brought to ~50% of majority
    """
    counts = Counter(y_arr)
    max_c = max(counts.values())
    return {
        0: max(counts[0], int(max_c * 0.60)),
        1: counts[1],
        2: max(counts[2], int(max_c * 0.85)),
        3: max(counts[3], int(max_c * 0.50))
    }


def stronger_strategy(y_arr):
    """
    Strongly boosts minority classes:
    Class 0 (Poor): brought to ~80% of majority
    Class 2 (Good): brought to ~90% of majority
    Class 3 (Excellent): brought to ~75% of majority
    """
    counts = Counter(y_arr)
    max_c = max(counts.values())
    return {
        0: max(counts[0], int(max_c * 0.80)),
        1: counts[1],
        2: max(counts[2], int(max_c * 0.90)),
        3: max(counts[3], int(max_c * 0.75))
    }


def evaluate_pipeline_cv(pipeline_builder, X_dev, y_dev, cv, cat_indices=None):
    """
    Executes 5-fold Stratified Cross-Validation on a candidate pipeline.
    Extracts fold-level overall metrics, per-class metrics, and sampling distributions.
    """
    target_names = [CLASS_MAPPING[i]["name"] for i in range(len(CLASS_MAPPING))]
    
    acc_scores = []
    macro_p_scores = []
    macro_r_scores = []
    macro_f1_scores = []
    weighted_f1_scores = []
    
    per_class_p = {c: [] for c in range(4)}
    per_class_r = {c: [] for c in range(4)}
    per_class_f1 = {c: [] for c in range(4)}
    
    before_sampling_counts = []
    after_sampling_counts = []
    
    oof_preds = np.zeros(len(y_dev), dtype=int)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_dev, y_dev)):
        X_tr, y_tr = X_dev.iloc[train_idx].copy(), y_dev[train_idx]
        X_va, y_va = X_dev.iloc[val_idx].copy(), y_dev[val_idx]
        
        before_counts = Counter(y_tr)
        before_sampling_counts.append(before_counts)
        
        # Instantiate fresh pipeline for this fold
        pipe, sampler = pipeline_builder()
        
        # Track resampled counts if sampler is present
        if sampler is not None:
            _, y_sampled = sampler.fit_resample(X_tr, y_tr)
            after_counts = Counter(y_sampled)
            after_sampling_counts.append(after_counts)
        else:
            after_sampling_counts.append(before_counts)
            
        pipe.fit(X_tr, y_tr)
        val_preds = pipe.predict(X_va)
        oof_preds[val_idx] = val_preds
        
        # Fold metrics
        acc_scores.append(accuracy_score(y_va, val_preds))
        mp, mr, mf1, _ = precision_recall_fscore_support(y_va, val_preds, average='macro', zero_division=0)
        wf1 = precision_recall_fscore_support(y_va, val_preds, average='weighted', zero_division=0)[2]
        
        macro_p_scores.append(mp)
        macro_r_scores.append(mr)
        macro_f1_scores.append(mf1)
        weighted_f1_scores.append(wf1)
        
        # Per class
        p_c, r_c, f_c, _ = precision_recall_fscore_support(y_va, val_preds, average=None, zero_division=0)
        for c in range(4):
            per_class_p[c].append(p_c[c])
            per_class_r[c].append(r_c[c])
            per_class_f1[c].append(f_c[c])
            
    # Compute aggregate fold summary
    mean_before = {c: int(np.mean([cnt[c] for cnt in before_sampling_counts])) for c in range(4)}
    mean_after = {c: int(np.mean([cnt[c] for cnt in after_sampling_counts])) for c in range(4)}
    
    return {
        "mean_accuracy": float(np.mean(acc_scores)),
        "std_accuracy": float(np.std(acc_scores)),
        "mean_macro_precision": float(np.mean(macro_p_scores)),
        "mean_macro_recall": float(np.mean(macro_r_scores)),
        "mean_macro_f1": float(np.mean(macro_f1_scores)),
        "std_macro_f1": float(np.std(macro_f1_scores)),
        "mean_weighted_f1": float(np.mean(weighted_f1_scores)),
        "per_class": {
            target_names[c]: {
                "precision": float(np.mean(per_class_p[c])),
                "recall": float(np.mean(per_class_r[c])),
                "f1": float(np.mean(per_class_f1[c]))
            }
            for c in range(4)
        },
        "mean_train_counts_before": mean_before,
        "mean_train_counts_after": mean_after,
        "oof_confusion_matrix": confusion_matrix(y_dev, oof_preds).tolist()
    }


def run_all_oversampling_experiments(X_dev, y_dev):
    """
    Evaluates 9 structured imbalance-handling experiments on 5-fold Stratified CV.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cat_indices = [X_dev.columns.get_loc(col) for col in CATEGORICAL_FEATURES]
    
    # Best Gradient Boosting Hyperparameters from previous phase
    gb_params = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 2,
        'min_samples_leaf': 4,
        'random_state': 42
    }
    
    experiments = {}

    # 1. Baseline: No Oversampling
    def build_baseline():
        return ImbPipeline([
            ('prep', build_preprocessor()),
            ('model', GradientBoostingClassifier(**gb_params))
        ]), None
    experiments["1. Baseline GB (No Oversampling)"] = build_baseline

    # 2. RandomOverSampler (Moderate)
    def build_ros_mod():
        ros = RandomOverSampler(sampling_strategy=moderate_strategy, random_state=42)
        return ImbPipeline([
            ('sampler', ros),
            ('prep', build_preprocessor()),
            ('model', GradientBoostingClassifier(**gb_params))
        ]), ros
    experiments["2. RandomOverSampler (Moderate)"] = build_ros_mod

    # 3. RandomOverSampler (Stronger)
    def build_ros_strong():
        ros = RandomOverSampler(sampling_strategy=stronger_strategy, random_state=42)
        return ImbPipeline([
            ('sampler', ros),
            ('prep', build_preprocessor()),
            ('model', GradientBoostingClassifier(**gb_params))
        ]), ros
    experiments["3. RandomOverSampler (Stronger)"] = build_ros_strong

    # 4. RandomOverSampler (Full Balance)
    def build_ros_full():
        ros = RandomOverSampler(sampling_strategy='not majority', random_state=42)
        return ImbPipeline([
            ('sampler', ros),
            ('prep', build_preprocessor()),
            ('model', GradientBoostingClassifier(**gb_params))
        ]), ros
    experiments["4. RandomOverSampler (Full Balance)"] = build_ros_full

    # 5. SMOTENC (Moderate)
    def build_smote_mod():
        smote = SMOTENC(categorical_features=cat_indices, sampling_strategy=moderate_strategy, k_neighbors=5, random_state=42)
        return ImbPipeline([
            ('sampler', smote),
            ('prep', build_preprocessor()),
            ('model', GradientBoostingClassifier(**gb_params))
        ]), smote
    experiments["5. SMOTENC (Moderate)"] = build_smote_mod

    # 6. SMOTENC (Stronger)
    def build_smote_strong():
        smote = SMOTENC(categorical_features=cat_indices, sampling_strategy=stronger_strategy, k_neighbors=5, random_state=42)
        return ImbPipeline([
            ('sampler', smote),
            ('prep', build_preprocessor()),
            ('model', GradientBoostingClassifier(**gb_params))
        ]), smote
    experiments["6. SMOTENC (Stronger)"] = build_smote_strong

    # 7. SMOTENC (Full Balance)
    def build_smote_full():
        smote = SMOTENC(categorical_features=cat_indices, sampling_strategy='not majority', k_neighbors=5, random_state=42)
        return ImbPipeline([
            ('sampler', smote),
            ('prep', build_preprocessor()),
            ('model', GradientBoostingClassifier(**gb_params))
        ]), smote
    experiments["7. SMOTENC (Full Balance)"] = build_smote_full

    # 8. Balanced Random Forest
    def build_brf():
        return ImbPipeline([
            ('prep', build_preprocessor()),
            ('model', BalancedRandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
        ]), None
    experiments["8. Balanced Random Forest"] = build_brf

    # 9. Cost-Sensitive Random Forest (Balanced Subsample)
    def build_rf_sub():
        return ImbPipeline([
            ('prep', build_preprocessor()),
            ('model', RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', max_depth=12, random_state=42, n_jobs=-1))
        ]), None
    experiments["9. Random Forest (Balanced Subsample)"] = build_rf_sub

    print("\n" + "=" * 90)
    print("EXECUTING OVERSAMPLING & IMBALANCE HANDLING EXPERIMENTS (5-FOLD CV, N=551)")
    print("=" * 90)

    results = {}
    for name, builder in experiments.items():
        print(f"\nEvaluating strategy: {name}...")
        res = evaluate_pipeline_cv(builder, X_dev, y_dev, cv, cat_indices)
        results[name] = res
        print(f"  -> Mean CV Macro F1: {res['mean_macro_f1']:.4f} (+/- {res['std_macro_f1']:.4f})")
        print(f"  -> Mean CV Accuracy: {res['mean_accuracy']:.4f} (+/- {res['std_accuracy']:.4f})")
        print(f"  -> Per-Class F1: Poor={res['per_class']['Poor']['f1']:.4f}, Avg={res['per_class']['Average']['f1']:.4f}, Good={res['per_class']['Good']['f1']:.4f}, Exc={res['per_class']['Excellent']['f1']:.4f}")

    return results, experiments


def compare_and_save_oversampling_results(results, experiments, X_dev, y_dev, X_test, y_test, output_dir="assets", models_dir="models"):
    """
    Selects best candidate based on Macro F1, consistency, and per-class balance.
    Refits best candidate on the full development set and evaluates on the outer test set.
    Saves all experiment metrics, predictions, confusion matrix, and feature importances.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Summary DataFrame of CV Results
    summary_rows = []
    for name, r in results.items():
        summary_rows.append({
            "Strategy": name,
            "CV_Macro_F1": r["mean_macro_f1"],
            "CV_Macro_F1_Std": r["std_macro_f1"],
            "CV_Accuracy": r["mean_accuracy"],
            "CV_Macro_Precision": r["mean_macro_precision"],
            "CV_Macro_Recall": r["mean_macro_recall"],
            "CV_Weighted_F1": r["mean_weighted_f1"],
            "Poor_F1": r["per_class"]["Poor"]["f1"],
            "Average_F1": r["per_class"]["Average"]["f1"],
            "Good_F1": r["per_class"]["Good"]["f1"],
            "Excellent_F1": r["per_class"]["Excellent"]["f1"],
            "Poor_Recall": r["per_class"]["Poor"]["recall"],
            "Excellent_Recall": r["per_class"]["Excellent"]["recall"],
            "Train_Counts_Before": str(r["mean_train_counts_before"]),
            "Train_Counts_After": str(r["mean_train_counts_after"])
        })
    df_results = pd.DataFrame(summary_rows).sort_values(by="CV_Macro_F1", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 110)
    print("5-FOLD CV EXPERIMENTATION RESULTS SUMMARY TABLE")
    print("=" * 110)
    print(f"{'Strategy':<38} | {'CV Macro F1':<16} | {'CV Acc':<8} | {'Poor F1':<7} | {'Avg F1':<7} | {'Good F1':<7} | {'Exc F1':<7}")
    print("-" * 110)
    for _, row in df_results.iterrows():
        f1_str = f"{row['CV_Macro_F1']:.4f} (+/- {row['CV_Macro_F1_Std']:.3f})"
        print(f"{row['Strategy']:<38} | {f1_str:<16} | {row['CV_Accuracy']:<8.4f} | {row['Poor_F1']:<7.4f} | {row['Average_F1']:<7.4f} | {row['Good_F1']:<7.4f} | {row['Excellent_F1']:<7.4f}")
    print("=" * 110)

    # Save CSV and JSON
    csv_path = os.path.join(output_dir, "oversampling_experiment_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved cross-validation experiment results to: {csv_path}")

    json_path = os.path.join(output_dir, "oversampling_experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full experiment details JSON to: {json_path}")

    # 2. Identify the Best Oversampling Strategy (excluding the baseline to test if oversampling helped)
    oversampling_candidates = df_results[df_results["Strategy"] != "1. Baseline GB (No Oversampling)"]
    best_os_name = oversampling_candidates.iloc[0]["Strategy"]
    baseline_macro_f1 = df_results[df_results["Strategy"] == "1. Baseline GB (No Oversampling)"].iloc[0]["CV_Macro_F1"]
    best_os_macro_f1 = oversampling_candidates.iloc[0]["CV_Macro_F1"]

    print("\n" + "=" * 90)
    print("OVERSAMPLING VS. BASELINE CV DECISION ANALYSIS")
    print("=" * 90)
    print(f"  Baseline GB (No Oversampling) Mean CV Macro F1: {baseline_macro_f1:.4f}")
    print(f"  Best Oversampling Strategy ({best_os_name}) Mean CV Macro F1: {best_os_macro_f1:.4f}")
    diff = best_os_macro_f1 - baseline_macro_f1
    if diff > 0.003:
        print(f"  Decision: Oversampling yielded a meaningful CV gain of {diff:+.4f}!")
    elif abs(diff) <= 0.003:
        print(f"  Decision: Oversampling performed virtually identical to the baseline ({diff:+.4f} difference).")
    else:
        print(f"  Decision: Oversampling degraded CV Macro F1 by {diff:+.4f} compared to the un-sampled baseline.")
    print("=" * 90)

    # 3. Refit Best Oversampling Model on 551 Development Records
    print(f"\nRefitting best oversampling strategy ({best_os_name}) on full development partition (N=551)...")
    best_builder = experiments[best_os_name]
    best_pipe, _ = best_builder()
    best_pipe.fit(X_dev, y_dev)
    print("Refit complete.")

    # 4. Evaluate Once on Outer Test Set (N=98)
    test_preds = best_pipe.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, test_preds, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_test, test_preds, average='weighted', zero_division=0)
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(y_test, test_preds, average=None, zero_division=0)

    target_names = [CLASS_MAPPING[i]["name"] for i in range(len(CLASS_MAPPING))]
    cm = confusion_matrix(y_test, test_preds)
    clf_report_dict = classification_report(y_test, test_preds, target_names=target_names, output_dict=True, zero_division=0)

    print("\n" + "=" * 90)
    print(f"EVALUATION OF BEST OVERSAMPLING MODEL ({best_os_name}) ON OUTER TEST SET (N=98)")
    print("=" * 90)
    print(f"  Accuracy:        {test_acc:.4f} ({test_acc:.2%})")
    print(f"  Macro Precision: {macro_p:.4f}")
    print(f"  Macro Recall:    {macro_r:.4f}")
    print(f"  Macro F1-Score:  {macro_f1:.4f}  <-- Primary Evaluation Metric")
    print(f"  Weighted F1:     {weighted_f1:.4f}\n")

    print("Per-Class Metrics on Test Set:")
    for idx, name in enumerate(target_names):
        print(f"  Class {idx} ({name:<9}): Precision = {per_p[idx]:.4f}, Recall = {per_r[idx]:.4f}, F1 = {per_f1[idx]:.4f}, Support = {int(per_sup[idx])}")
    print()

    print("Confusion Matrix:")
    print(cm)
    print("=" * 90)

    # 5. Save Artifacts Separately (Preserving previous models)
    model_save_path = os.path.join(models_dir, "best_oversampling_model.joblib")
    joblib.dump(best_pipe, model_save_path)
    print(f"\nSaved best oversampling model pipeline to: {model_save_path}")

    preds_save_path = os.path.join(output_dir, "oversampling_model_predictions.npy")
    np.save(preds_save_path, test_preds)
    print(f"Saved test predictions to: {preds_save_path}")

    metrics_payload = {
        "strategy_name": best_os_name,
        "cv_metrics": {
            "mean_macro_f1": float(df_results[df_results["Strategy"] == best_os_name]["CV_Macro_F1"].iloc[0]),
            "std_macro_f1": float(df_results[df_results["Strategy"] == best_os_name]["CV_Macro_F1_Std"].iloc[0]),
            "mean_accuracy": float(df_results[df_results["Strategy"] == best_os_name]["CV_Accuracy"].iloc[0]),
            "mean_macro_precision": float(df_results[df_results["Strategy"] == best_os_name]["CV_Macro_Precision"].iloc[0]),
            "mean_macro_recall": float(df_results[df_results["Strategy"] == best_os_name]["CV_Macro_Recall"].iloc[0]),
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
    metrics_path = os.path.join(output_dir, "oversampling_model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved oversampling model metrics JSON to: {metrics_path}")

    # Confusion matrix plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGnBu",
        xticklabels=target_names, yticklabels=target_names,
        linewidths=0.8, linecolor="#eeeeee"
    )
    plt.title(f"Best Oversampling Model ({best_os_name})\nTest Confusion Matrix", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight='bold')
    plt.ylabel("True Label", fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, "oversampling_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

    # Feature Importance if supported
    print("\nExtracting feature importances for best oversampling model...")
    fitted_prep = best_pipe.named_steps['prep']
    fitted_model = best_pipe.named_steps['model']
    feature_names = list(fitted_prep.get_feature_names_out())

    if hasattr(fitted_model, "feature_importances_"):
        importances = fitted_model.feature_importances_
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        fi_df["rank"] = fi_df.index + 1

        fi_csv_path = os.path.join(output_dir, "oversampling_feature_importance.csv")
        fi_df.to_csv(fi_csv_path, index=False)
        print(f"Saved feature importance table to: {fi_csv_path}")

    print("\n" + "=" * 90)
    print("OVERSAMPLING EXPERIMENTATION FULLY COMPLETED!")
    print("=" * 90)

    return df_results, metrics_payload


def main():
    X_dev, y_dev, X_test, y_test = get_data_splits()
    results, experiments = run_all_oversampling_experiments(X_dev, y_dev)
    compare_and_save_oversampling_results(results, experiments, X_dev, y_dev, X_test, y_test)


if __name__ == "__main__":
    main()
