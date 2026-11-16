"""
EduPredict - Model Comparison Script
Compares the performance of the local Random Forest and the Colab Deep Neural Network
on the exact same test dataset. Prioritizes Macro F1-score due to class imbalance.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'


def compare_models(
    rf_metrics_path="assets/rf_metrics.json",
    dnn_metrics_path="assets/dnn_metrics.json",
    output_dir="assets"
):
    print("=" * 75)
    print("EDUPREDICT: MODEL PERFORMANCE COMPARISON (RANDOM FOREST vs DNN)")
    print("=" * 75)

    # 1. Check Random Forest Metrics
    if not os.path.exists(rf_metrics_path):
        raise FileNotFoundError(f"Random Forest metrics file not found at: {rf_metrics_path}")

    with open(rf_metrics_path, "r", encoding="utf-8") as f:
        rf_data = json.load(f)
    rf_test = rf_data["test_metrics"]
    print(f"Loaded Random Forest metrics from: {rf_metrics_path}")

    # 2. Check DNN Metrics
    dnn_available = os.path.exists(dnn_metrics_path)
    if not dnn_available:
        print(f"\n[NOTE] Deep Neural Network metrics not found at: {dnn_metrics_path}")
        print("  Google Colab training has not yet been executed or outputs have not been downloaded.")
        print("\nDisplaying available Random Forest baseline metrics:")
        print(f"  - Test Accuracy:        {rf_test['accuracy']:.4f} ({rf_test['accuracy']:.2%})")
        print(f"  - Test Macro Precision: {rf_test['macro_precision']:.4f}")
        print(f"  - Test Macro Recall:    {rf_test['macro_recall']:.4f}")
        print(f"  - Test Macro F1-Score:  {rf_test['macro_f1']:.4f}  <-- Primary Evaluation Metric")
        print(f"  - Test Weighted F1:     {rf_test['weighted_f1']:.4f}")
        print("\nRun this comparison script again once 'dnn_metrics.json' is downloaded from Colab.")
        print("=" * 75)
        return {
            "status": "partial",
            "rf_metrics": rf_test,
            "dnn_metrics": None
        }

    with open(dnn_metrics_path, "r", encoding="utf-8") as f:
        dnn_data = json.load(f)
    dnn_test = dnn_data.get("test_metrics", dnn_data)
    print(f"Loaded Deep Neural Network metrics from: {dnn_metrics_path}\n")

    # 3. Build Comparison DataFrame
    metrics_to_compare = [
        ("Accuracy", "accuracy"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Macro F1-Score", "macro_f1"),
        ("Weighted F1-Score", "weighted_f1"),
    ]

    records = []
    for label, key in metrics_to_compare:
        rf_val = rf_test.get(key, 0.0)
        dnn_val = dnn_test.get(key, 0.0)
        diff = dnn_val - rf_val
        winner = "DNN" if diff > 0.0005 else ("RF" if diff < -0.0005 else "Tie")
        records.append({
            "Metric": label,
            "Random Forest": rf_val,
            "Deep Neural Network": dnn_val,
            "Difference (DNN - RF)": diff,
            "Leading Model": winner
        })

    comp_df = pd.DataFrame(records)

    print("Model Comparison Summary Table:")
    print("-" * 75)
    print(f"{'Metric':<20} | {'Random Forest':<14} | {'Deep Neural Net':<16} | {'Diff':<8} | {'Leader':<6}")
    print("-" * 75)
    for _, r in comp_df.iterrows():
        print(f"{r['Metric']:<20} | {r['Random Forest']:<14.4f} | {r['Deep Neural Network']:<16.4f} | {r['Difference (DNN - RF)']:+7.4f} | {r['Leading Model']:<6}")
    print("-" * 75)

    rf_macro = rf_test["macro_f1"]
    dnn_macro = dnn_test["macro_f1"]
    print("\nPrimary Metric Analysis (Macro F1-Score):")
    if abs(dnn_macro - rf_macro) < 0.005:
        print(f"  Models perform comparably on Macro F1 (RF: {rf_macro:.4f} vs DNN: {dnn_macro:.4f}).")
    elif dnn_macro > rf_macro:
        pct_gain = ((dnn_macro - rf_macro) / rf_macro) * 100
        print(f"  Deep Neural Network outperforms Random Forest on Macro F1 by {pct_gain:+.2f}% ({dnn_macro:.4f} vs {rf_macro:.4f}).")
    else:
        pct_gain = ((rf_macro - dnn_macro) / dnn_macro) * 100
        print(f"  Random Forest outperforms Deep Neural Network on Macro F1 by {pct_gain:+.2f}% ({rf_macro:.4f} vs {dnn_macro:.4f}).")

    # 4. Save Comparison CSV and JSON
    comp_csv_path = os.path.join(output_dir, "model_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\nSaved comparison CSV to: {comp_csv_path}")

    # Per-class F1 comparison
    rf_per_class = rf_test.get("per_class", {})
    dnn_per_class = dnn_test.get("per_class", {})
    per_class_records = []
    print("\nPer-Class F1-Score Comparison:")
    print("-" * 75)
    print(f"{'Class':<15} | {'Random Forest F1':<18} | {'Deep Neural Net F1':<20} | {'Leader':<6}")
    print("-" * 75)
    for cls_name in ["Poor", "Average", "Good", "Excellent"]:
        rf_f1 = rf_per_class.get(cls_name, {}).get("f1_score", 0.0)
        dnn_f1 = dnn_per_class.get(cls_name, {}).get("f1_score", 0.0)
        c_diff = dnn_f1 - rf_f1
        c_winner = "DNN" if c_diff > 0.005 else ("RF" if c_diff < -0.005 else "Tie")
        per_class_records.append({
            "Class": cls_name,
            "RF_F1": rf_f1,
            "DNN_F1": dnn_f1,
            "Diff (DNN - RF)": c_diff,
            "Leader": c_winner
        })
        print(f"{cls_name:<15} | {rf_f1:<18.4f} | {dnn_f1:<20.4f} | {c_winner:<6}")
    print("-" * 75)

    comp_json_path = os.path.join(output_dir, "model_comparison.json")
    comparison_payload = {
        "metrics_comparison": comp_df.to_dict(orient="records"),
        "per_class_comparison": per_class_records,
        "primary_metric": "Macro F1-Score",
        "rf_macro_f1": float(rf_macro),
        "dnn_macro_f1": float(dnn_macro),
        "macro_f1_winner": "Random Forest" if rf_macro > dnn_macro else "Deep Neural Network",
        "rf_accuracy": float(rf_test.get("accuracy", 0.0)),
        "dnn_accuracy": float(dnn_test.get("accuracy", 0.0)),
        "accuracy_winner": "Deep Neural Network" if dnn_test.get("accuracy", 0.0) > rf_test.get("accuracy", 0.0) else "Random Forest",
        "detailed_rf": rf_test,
        "detailed_dnn": dnn_test
    }
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_payload, f, indent=2)
    print(f"Saved comparison JSON to: {comp_json_path}")

    # 5. Generate Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comp_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, comp_df["Random Forest"], width, label="Random Forest", color="#2980b9", edgecolor="#333333", linewidth=0.6)
    bars2 = ax.bar(x + width/2, comp_df["Deep Neural Network"], width, label="Deep Neural Network", color="#27ae60", edgecolor="#333333", linewidth=0.6)

    ax.set_title("Model Comparison: Random Forest vs Deep Neural Network (Test Set)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Metric"], fontsize=10, fontweight='bold')
    ax.set_ylabel("Score", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, loc='lower right')

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    comp_plot_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot to: {comp_plot_path}")
    print("=" * 75)

    return comparison_payload


if __name__ == "__main__":
    compare_models()
