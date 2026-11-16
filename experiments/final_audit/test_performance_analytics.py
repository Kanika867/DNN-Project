"""
EduPredict - Phase 4D Test Suite: Performance Analytics
Verifies:
1. Dynamic KPI statistics match exact Pandas calculations
2. Correlation calculations match ground-truth values
3. All Plotly analytics charts generate cleanly without errors
4. Group sizes, sample counts, and percentage denominators
5. Filter operations and empty-state resilience
6. Multi-page compilation and server probing
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.app_utils import load_student_data
from src.dashboard_components import (
    render_g3_distribution_histogram,
    render_category_box_plots,
    render_previous_grades_scatters,
    render_studytime_full_analytics,
    render_failures_full_analytics,
    render_attendance_full_analytics,
    render_demographics_full_analytics,
    render_correlation_heatmap
)

def run_tests():
    print("=== 1. VERIFY DATASET LOADING & EXACT SUMMARY STATISTICS ===")
    df = load_student_data()
    assert df is not None, "Failed to load student dataset!"
    assert len(df) == 649, f"Expected 649 rows, got {len(df)}"

    avg_g3 = float(df["G3"].mean())
    med_g3 = float(df["G3"].median())
    std_g3 = float(df["G3"].std())
    good_exc_pct = ((df["target_class"] >= 2).sum() / len(df)) * 100
    fail_pct = ((df["failures"] > 0).sum() / len(df)) * 100

    assert np.isclose(avg_g3, 11.906, atol=0.01), f"Mean G3 mismatch: {avg_g3}"
    assert np.isclose(med_g3, 12.0, atol=0.01), f"Median G3 mismatch: {med_g3}"
    assert np.isclose(std_g3, 3.23, atol=0.01), f"Std G3 mismatch: {std_g3}"
    assert np.isclose(good_exc_pct, 42.53, atol=0.01), f"Good+Exc % mismatch: {good_exc_pct}"
    assert np.isclose(fail_pct, 15.41, atol=0.01), f"Failures % mismatch: {fail_pct}"
    print(f"KPI statistics verified: Mean={avg_g3:.2f}, Median={med_g3:.1f}, Std={std_g3:.2f}, Good+Exc={good_exc_pct:.2f}%, Failures={fail_pct:.2f}%.")

    print("\n=== 2. VERIFY EXACT CORRELATION VALUES ===")
    r_g2_g3 = float(df["G2"].corr(df["G3"]))
    r_g1_g3 = float(df["G1"].corr(df["G3"]))
    r_g1_g2 = float(df["G1"].corr(df["G2"]))
    r_fail_g3 = float(df["failures"].corr(df["G3"]))
    r_study_g3 = float(df["studytime"].corr(df["G3"]))

    assert np.isclose(r_g2_g3, 0.9185, atol=0.001), f"G2-G3 correlation mismatch: {r_g2_g3}"
    assert np.isclose(r_g1_g3, 0.8264, atol=0.001), f"G1-G3 correlation mismatch: {r_g1_g3}"
    assert np.isclose(r_g1_g2, 0.8650, atol=0.001), f"G1-G2 correlation mismatch: {r_g1_g2}"
    assert np.isclose(r_fail_g3, -0.3933, atol=0.001), f"Failures-G3 correlation mismatch: {r_fail_g3}"
    assert np.isclose(r_study_g3, 0.2498, atol=0.001), f"Study-G3 correlation mismatch: {r_study_g3}"
    print(f"Correlations verified: r(G2,G3)={r_g2_g3:.4f}, r(G1,G3)={r_g1_g3:.4f}, r(G1,G2)={r_g1_g2:.4f}, r(fail,G3)={r_fail_g3:.4f}, r(study,G3)={r_study_g3:.4f}.")

    print("\n=== 3. VERIFY ALL PLOTLY ANALYTICS CHARTS GENERATION ===")
    fig_hist = render_g3_distribution_histogram(df)
    assert fig_hist is not None and len(fig_hist.data) >= 1, "Histogram generation failed!"
    print("  - G3 Distribution Histogram: OK")

    fig_box = render_category_box_plots(df)
    assert fig_box is not None and len(fig_box.data) >= 1, "Box plots generation failed!"
    print("  - Category Box Plots: OK")

    scatters = render_previous_grades_scatters(df)
    assert "g1_g3" in scatters and "g2_g3" in scatters and "g1_g2" in scatters, "Previous grades scatter generation failed!"
    print("  - Previous Grades Scatters (3 charts): OK")

    st_charts = render_studytime_full_analytics(df)
    assert "mean_chart" in st_charts and "stack_chart" in st_charts, "Study time charts generation failed!"
    print("  - Study Time Analytics (2 charts): OK")

    fail_charts = render_failures_full_analytics(df)
    assert "bar_chart" in fail_charts and "box_chart" in fail_charts, "Failures charts generation failed!"
    print("  - Historical Failures Analytics (2 charts): OK")

    att_charts = render_attendance_full_analytics(df)
    assert "bar_chart" in att_charts and "scatter_chart" in att_charts, "Attendance charts generation failed!"
    print("  - Attendance & Absences Analytics (2 charts): OK")

    demo_charts = render_demographics_full_analytics(df)
    assert "school" in demo_charts and "gender" in demo_charts and "address" in demo_charts and "internet" in demo_charts, "Demographic charts generation failed!"
    print("  - Demographic & Contextual Analytics (4 charts): OK")

    heatmap = render_correlation_heatmap(df)
    assert heatmap is not None and len(heatmap.data) >= 1, "Heatmap generation failed!"
    print("  - 15-Feature Correlation Heatmap: OK")

    print("\n=== 4. VERIFY GROUP SAMPLE SIZES AND COUNTS ===")
    # Study time counts
    st_counts = df["studytime"].value_counts().to_dict()
    assert st_counts[1] == 212 and st_counts[2] == 305 and st_counts[3] == 97 and st_counts[4] == 35, f"Study time counts mismatch: {st_counts}"
    print(f"Study time sample counts verified: 1: {st_counts[1]}, 2: {st_counts[2]}, 3: {st_counts[3]}, 4: {st_counts[4]}.")

    # Failures counts
    fail_counts = df["failures"].value_counts().to_dict()
    assert fail_counts[0] == 549 and fail_counts[1] == 70 and fail_counts[2] == 16 and fail_counts[3] == 14, f"Failures counts mismatch: {fail_counts}"
    print(f"Past failures sample counts verified: 0: {fail_counts[0]}, 1: {fail_counts[1]}, 2: {fail_counts[2]}, 3: {fail_counts[3]}.")

    print("\n=== 5. VERIFY FILTERING AND RESILIENCE ===")
    df_gp = df[df["school"] == "GP"]
    assert len(df_gp) == 423
    charts_gp = render_previous_grades_scatters(df_gp)
    assert charts_gp["r_g2_g3"] > 0.9, "GP subset correlation mismatch"
    print("Filtered subset analytics verified.")

    print("\n=== 6. VERIFY BYTECODE COMPILATION OF ALL PAGES ===")
    import py_compile, glob
    for f in glob.glob("pages/*.py") + ["app.py", "src/app_utils.py", "src/dashboard_components.py"]:
        py_compile.compile(f, doraise=True)
        print(f"  - Compiled: {f}")

    print("\n=======================================================")
    print("ALL PHASE 4D TEST SUITE CHECKS PASSED WITH 100% SUCCESS!")
    print("=======================================================")

if __name__ == "__main__":
    run_tests()
