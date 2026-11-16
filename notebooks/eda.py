"""
EduPredict - Exploratory Data Analysis (EDA) Script
Generates comprehensive statistical summaries and 7 high-resolution visualizations
saved to the assets/ directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

CLASS_PALETTE = {
    "Poor (0)": "#e74c3c",      # Red
    "Average (1)": "#f39c12",   # Orange
    "Good (2)": "#3498db",      # Blue
    "Excellent (3)": "#2ecc71"  # Green
}

def run_eda(data_path="data/student-por.csv", assets_dir="assets"):
    os.makedirs(assets_dir, exist_ok=True)
    df = pd.read_csv(data_path, sep=';')

    print("=" * 70)
    print("EDUPREDICT: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # Target class derivation
    bins = [-1, 9, 12, 15, 20]
    labels = ["Poor (0)", "Average (1)", "Good (2)", "Excellent (3)"]
    df['performance_class'] = pd.cut(df['G3'], bins=bins, labels=labels)

    # ---------------------------------------------------------
    # Chart 1: Target Grade and Class Distribution
    # ---------------------------------------------------------
    print("Generating Chart 1: Target Grade & Class Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Continuous G3 distribution
    sns.histplot(df['G3'], kde=True, bins=21, ax=axes[0], color="#2980b9", edgecolor="white")
    axes[0].axvline(df['G3'].mean(), color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Mean: {df['G3'].mean():.2f}")
    axes[0].axvline(df['G3'].median(), color="#27ae60", linestyle=":", linewidth=2, label=f"Median: {df['G3'].median():.2f}")
    axes[0].set_title("Distribution of Final Grades (G3: 0–20)", fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel("G3 Final Grade", fontsize=11)
    axes[0].set_ylabel("Student Count", fontsize=11)
    axes[0].set_xlim(-0.5, 20.5)
    axes[0].legend(frameon=True)

    # Categorical 4-Class Distribution
    class_counts = df['performance_class'].value_counts()[labels]
    bars = axes[1].bar(
        [str(l) for l in labels],
        class_counts.values,
        color=[CLASS_PALETTE[l] for l in labels],
        edgecolor="#333333",
        linewidth=0.8,
        width=0.55
    )
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 4, f"{h}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].set_title("Target Performance Classes (4-Category Breakdown)", fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel("Performance Category", fontsize=11)
    axes[1].set_ylabel("Student Count", fontsize=11)
    axes[1].set_ylim(0, max(class_counts.values) * 1.18)

    plt.tight_layout()
    chart1_path = os.path.join(assets_dir, "eda_01_target_distribution.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"  -> Saved {chart1_path}")

    # ---------------------------------------------------------
    # Chart 2: Relationships Between G1, G2 and G3
    # ---------------------------------------------------------
    print("Generating Chart 2: G1 and G2 vs G3 Progression...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    corr_g1_g3 = df['G1'].corr(df['G3'])
    corr_g2_g3 = df['G2'].corr(df['G3'])

    # G1 vs G3
    sns.regplot(
        data=df, x='G1', y='G3', ax=axes[0],
        scatter_kws={'alpha': 0.55, 'color': '#3498db'},
        line_kws={'color': '#e74c3c', 'linewidth': 2}
    )
    axes[0].set_title(f"Period 1 (G1) vs Final Grade (G3)\nPearson Correlation r = {corr_g1_g3:.3f}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("First Period Grade (G1)", fontsize=11)
    axes[0].set_ylabel("Final Grade (G3)", fontsize=11)
    axes[0].set_xlim(-1, 21)
    axes[0].set_ylim(-1, 21)

    # G2 vs G3
    sns.regplot(
        data=df, x='G2', y='G3', ax=axes[1],
        scatter_kws={'alpha': 0.55, 'color': '#2ecc71'},
        line_kws={'color': '#e74c3c', 'linewidth': 2}
    )
    axes[1].set_title(f"Period 2 (G2) vs Final Grade (G3)\nPearson Correlation r = {corr_g2_g3:.3f}", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Second Period Grade (G2)", fontsize=11)
    axes[1].set_ylabel("Final Grade (G3)", fontsize=11)
    axes[1].set_xlim(-1, 21)
    axes[1].set_ylim(-1, 21)

    plt.tight_layout()
    chart2_path = os.path.join(assets_dir, "eda_02_g1_g2_g3_relationships.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"  -> Saved {chart2_path}")

    # ---------------------------------------------------------
    # Chart 3: Academic Performance by Study Time
    # ---------------------------------------------------------
    print("Generating Chart 3: Performance by Study Time...")
    study_labels = {1: "< 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "> 10 hours"}
    df['studytime_label'] = df['studytime'].map(study_labels)
    study_order = ["< 2 hours", "2 to 5 hours", "5 to 10 hours", "> 10 hours"]

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = ["#f8c471", "#85c1e9", "#7dcea0", "#bb8fce"]
    sns.boxplot(data=df, x='studytime_label', y='G3', order=study_order, hue='studytime_label', palette=palette, legend=False, ax=ax, width=0.55)
    sns.stripplot(data=df, x='studytime_label', y='G3', order=study_order, color='#2c3e50', alpha=0.3, jitter=0.2, size=4, ax=ax)

    # Medians display
    medians = df.groupby('studytime_label', observed=False)['G3'].median()[study_order]
    counts = df['studytime_label'].value_counts()[study_order]
    for idx, (name, med) in enumerate(medians.items()):
        ax.text(idx, med + 0.4, f"Med: {med:.1f}\n(n={counts[name]})", ha='center', va='bottom', fontsize=9, fontweight='bold', color="#111111")

    ax.set_title("Academic Performance (G3) by Weekly Study Time", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Weekly Study Time Category", fontsize=11)
    ax.set_ylabel("Final Grade (G3)", fontsize=11)
    ax.set_ylim(-1, 22)

    plt.tight_layout()
    chart3_path = os.path.join(assets_dir, "eda_03_performance_by_studytime.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"  -> Saved {chart3_path}")

    # ---------------------------------------------------------
    # Chart 4: Academic Performance by Past Failures
    # ---------------------------------------------------------
    print("Generating Chart 4: Performance by Past Class Failures...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot of G3 vs failures
    fail_palette = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    sns.boxplot(data=df, x='failures', y='G3', hue='failures', palette=fail_palette, legend=False, ax=axes[0], width=0.5)
    axes[0].set_title("Final Grade Distribution Across Past Class Failures", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Number of Past Failures (0 to 3)", fontsize=11)
    axes[0].set_ylabel("Final Grade (G3)", fontsize=11)

    # Proportion of classes by failures
    fail_ct = pd.crosstab(df['failures'], df['performance_class'], normalize='index')[labels] * 100
    fail_ct.plot(
        kind='bar', stacked=True, ax=axes[1],
        color=[CLASS_PALETTE[l] for l in labels],
        edgecolor='#333333', linewidth=0.6, width=0.55
    )
    axes[1].set_title("Performance Class Breakdown by Past Failures (%)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Number of Past Failures", fontsize=11)
    axes[1].set_ylabel("Percentage (%)", fontsize=11)
    axes[1].legend(title="Class", loc='upper right', frameon=True)
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    chart4_path = os.path.join(assets_dir, "eda_04_performance_by_past_failures.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"  -> Saved {chart4_path}")

    # ---------------------------------------------------------
    # Chart 5: Performance by School Absences
    # ---------------------------------------------------------
    print("Generating Chart 5: Performance by School Absences...")
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = sns.scatterplot(
        data=df, x='absences', y='G3', hue='performance_class',
        hue_order=labels, palette=CLASS_PALETTE, alpha=0.75, s=55, ax=ax
    )
    sns.regplot(
        data=df, x='absences', y='G3', scatter=False, ax=ax,
        color="#34495e", line_kws={'linestyle': '--', 'linewidth': 2, 'label': 'Linear Trend'}
    )
    corr_abs_g3 = df['absences'].corr(df['G3'])
    ax.set_title(f"Student Absences vs Final Grade (G3)\nPearson Correlation r = {corr_abs_g3:.3f}", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Number of School Absences", fontsize=11)
    ax.set_ylabel("Final Grade (G3)", fontsize=11)
    ax.legend(title="Performance Class", loc='upper right', frameon=True)

    plt.tight_layout()
    chart5_path = os.path.join(assets_dir, "eda_05_performance_by_absences.png")
    plt.savefig(chart5_path, dpi=300)
    plt.close()
    print(f"  -> Saved {chart5_path}")

    # ---------------------------------------------------------
    # Chart 6: Performance by Home Internet Access
    # ---------------------------------------------------------
    print("Generating Chart 6: Performance by Home Internet Access...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    internet_labels = {"yes": "With Internet", "no": "Without Internet"}
    df['internet_desc'] = df['internet'].map(internet_labels)
    int_order = ["With Internet", "Without Internet"]

    sns.boxplot(
        data=df, x='internet_desc', y='G3', order=int_order,
        hue='internet_desc', palette=["#5dade2", "#f5b7b1"], legend=False, width=0.45, ax=axes[0]
    )
    axes[0].set_title("G3 Final Grades by Home Internet Access", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Home Internet Status", fontsize=11)
    axes[0].set_ylabel("Final Grade (G3)", fontsize=11)

    int_medians = df.groupby('internet_desc', observed=False)['G3'].mean()[int_order]
    for idx, (name, m) in enumerate(int_medians.items()):
        axes[0].text(idx, m + 0.3, f"Mean: {m:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    int_ct = pd.crosstab(df['internet_desc'], df['performance_class'], normalize='index')[labels] * 100
    int_ct.loc[int_order].plot(
        kind='bar', stacked=True, ax=axes[1],
        color=[CLASS_PALETTE[l] for l in labels],
        edgecolor='#333333', linewidth=0.6, width=0.45
    )
    axes[1].set_title("Performance Class Proportions by Internet Access (%)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Home Internet Status", fontsize=11)
    axes[1].set_ylabel("Percentage (%)", fontsize=11)
    axes[1].legend(title="Class", loc='upper right', frameon=True)
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    chart6_path = os.path.join(assets_dir, "eda_06_performance_by_internet_access.png")
    plt.savefig(chart6_path, dpi=300)
    plt.close()
    print(f"  -> Saved {chart6_path}")

    # ---------------------------------------------------------
    # Chart 7: Numerical Feature Correlation Heatmap
    # ---------------------------------------------------------
    print("Generating Chart 7: Correlation Heatmap for Numerical Features...")
    num_cols = [
        "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        "absences", "G1", "G2", "G3"
    ]
    corr_matrix = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-0.4, vmax=1.0, cbar_kws={'label': 'Pearson Correlation Coefficient'},
        linewidths=0.5, linecolor="#eeeeee", ax=ax, annot_kws={"size": 8.5}
    )
    ax.set_title("Pearson Correlation Matrix of Numerical Attributes", fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    chart7_path = os.path.join(assets_dir, "eda_07_numerical_correlation_heatmap.png")
    plt.savefig(chart7_path, dpi=300)
    plt.close()
    print(f"  -> Saved {chart7_path}")

    print("=" * 70)
    print("EDA COMPLETED SUCCESSFULLY! ALL 7 CHARTS GENERATED IN 'assets/'")
    print("=" * 70)

if __name__ == "__main__":
    run_eda()
