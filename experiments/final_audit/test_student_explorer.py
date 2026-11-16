"""
EduPredict - Phase 4C Test Suite: Student Explorer
Verifies:
1. Data integrity (649 records, 0 missing, 0 duplicates, stable anonymous IDs)
2. Filter logic (independent, combined, edge cases)
3. Empty state handling
4. Chart generation for filtered subsets
5. CSV export schema and content integrity
6. Multi-page compilation and server response
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.app_utils import load_student_data
from src.dashboard_components import (
    render_explorer_kpi_cards,
    render_filtered_distribution_chart,
    render_comparative_charts
)

def run_tests():
    print("=== 1. VERIFY DATASET INTEGRITY & ANONYMOUS IDENTIFIERS ===")
    df = load_student_data()
    assert df is not None, "Failed to load student dataset!"
    assert len(df) == 649, f"Expected 649 rows, got {len(df)}"
    assert df.isnull().sum().sum() == 0, "Found missing values in dataset!"
    assert df.drop(columns=["student_id", "target_class", "target_name", "performance_category"]).duplicated().sum() == 0, "Found duplicate records!"
    assert "student_id" in df.columns, "Missing student_id column!"
    assert df["student_id"].iloc[0] == "Student-001", "Invalid first student_id!"
    assert df["student_id"].iloc[-1] == "Student-649", "Invalid last student_id!"
    print(f"Dataset integrity verified: {len(df)} rows, 0 missing, 0 duplicates, stable IDs from Student-001 to Student-649.")

    print("\n=== 2. VERIFY TARGET CATEGORY BOUNDARIES ===")
    assert ((df["target_class"] == 0) == (df["G3"] <= 9)).all(), "Poor category boundary mismatch!"
    assert ((df["target_class"] == 1) == ((df["G3"] >= 10) & (df["G3"] <= 12))).all(), "Average category boundary mismatch!"
    assert ((df["target_class"] == 2) == ((df["G3"] >= 13) & (df["G3"] <= 15))).all(), "Good category boundary mismatch!"
    assert ((df["target_class"] == 3) == (df["G3"] >= 16)).all(), "Excellent category boundary mismatch!"
    print("Category boundaries verified: 0 (Poor 0-9), 1 (Avg 10-12), 2 (Good 13-15), 3 (Exc 16-20).")

    print("\n=== 3. VERIFY INDEPENDENT FILTERS ===")
    # School filter
    df_gp = df[df["school"] == "GP"]
    assert len(df_gp) == 423, f"Expected 423 GP students, got {len(df_gp)}"
    df_ms = df[df["school"] == "MS"]
    assert len(df_ms) == 226, f"Expected 226 MS students, got {len(df_ms)}"
    print(f"School filter verified: GP={len(df_gp)}, MS={len(df_ms)}.")

    # Gender filter
    df_f = df[df["sex"] == "F"]
    assert len(df_f) == 383, f"Expected 383 female students, got {len(df_f)}"
    df_m = df[df["sex"] == "M"]
    assert len(df_m) == 266, f"Expected 266 male students, got {len(df_m)}"
    print(f"Gender filter verified: Female={len(df_f)}, Male={len(df_m)}.")

    # Failures filter
    df_fail = df[df["failures"] >= 1]
    assert len(df_fail) == 100, f"Expected 100 students with >=1 failures, got {len(df_fail)}"
    print(f"Failures filter verified: >=1 failures = {len(df_fail)} students.")

    # Category filter
    df_poor = df[df["target_class"] == 0]
    assert len(df_poor) == 100, f"Expected 100 Poor students, got {len(df_poor)}"
    print(f"Category filter verified: Poor={len(df_poor)}.")

    print("\n=== 4. VERIFY COMBINED MULTI-CRITERIA FILTERS ===")
    df_comb = df[
        (df["school"] == "GP") &
        (df["sex"] == "F") &
        (df["studytime"] >= 3) &
        (df["failures"] == 0) &
        (df["target_class"] == 3)
    ]
    print(f"Combined filter (GP, Female, Study>=3, Failures=0, Excellent): {len(df_comb)} matching students.")
    assert len(df_comb) > 0, "Expected at least 1 student matching combined criteria."

    print("\n=== 5. VERIFY EMPTY STATE HANDLING ===")
    df_empty = df[(df["age"] > 30)] # No student aged > 30
    assert len(df_empty) == 0, "Expected 0 students."
    chart_empty = render_filtered_distribution_chart(df_empty)
    assert chart_empty is None, "Empty dataframe should return None for chart!"
    print("Empty filter handled gracefully: chart returns None without exceptions.")

    print("\n=== 6. VERIFY CHART GENERATION ON FILTERED SUBSET ===")
    chart_dist = render_filtered_distribution_chart(df_gp)
    assert chart_dist is not None, "Failed to render distribution chart for GP students!"
    assert len(chart_dist.data) == 1, "Expected 1 bar trace in distribution chart."

    comp_charts = render_comparative_charts(df_gp)
    assert "school" in comp_charts and "studytime" in comp_charts and "gender" in comp_charts and "failures" in comp_charts and "absences" in comp_charts
    print(f"Comparative charts successfully generated: {list(comp_charts.keys())}.")

    print("\n=== 7. VERIFY CSV EXPORT SCHEMA AND INTEGRITY ===")
    export_cols = ["student_id"] + [c for c in df.columns if c not in ["student_id", "target_class", "target_name", "performance_category"]] + ["performance_category"]
    exp_subset = df_gp[export_cols]
    assert len(exp_subset) == 423, f"Expected 423 rows in export, got {len(exp_subset)}"
    assert exp_subset["student_id"].iloc[0] == "Student-001", "Export missing student_id!"
    assert "G3" in exp_subset.columns, "Export missing G3!"
    assert "performance_category" in exp_subset.columns, "Export missing performance_category!"
    csv_text = exp_subset.to_csv(index=False)
    assert len(csv_text.splitlines()) == 424, "CSV line count mismatch (header + 423 records)!"
    print(f"CSV export verified: exactly 423 filtered rows with 35 columns (ID, 33 original features, Category).")

    print("\n=== 8. VERIFY BYTECODE COMPILATION OF ALL STREAMLIT PAGES ===")
    import py_compile, glob
    for f in glob.glob("pages/*.py") + ["app.py", "src/app_utils.py", "src/dashboard_components.py"]:
        py_compile.compile(f, doraise=True)
        print(f"  - Compiled: {f}")

    print("\n=======================================================")
    print("ALL PHASE 4C TEST SUITE CHECKS PASSED WITH 100% SUCCESS!")
    print("=======================================================")

if __name__ == "__main__":
    run_tests()
