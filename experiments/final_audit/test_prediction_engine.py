"""
EduPredict - Phase 4B Test Suite: Student Prediction Engine
Verifies:
1. Model loading without preprocessing duplication
2. 32-feature canonical schema and order
3. Exact matching between direct pipeline and app predict_student across multiple real records
4. Class score properties (sum to 1.0, 4 classes)
5. Graceful exception handling for invalid/missing inputs
6. Non-judgmental educational recommendation generation
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.app_utils import (
    load_production_model,
    predict_student,
    generate_academic_recommendations,
    ORIGINAL_FEATURE_NAMES,
    DEFAULT_CLASS_MAPPING
)

def run_tests():
    print("=== 1. VERIFY MODEL LOADING ===")
    model = load_production_model()
    assert model is not None, "Model failed to load"
    print("Model loaded successfully:", type(model))

    print("\n=== 2. VERIFY EXACT 32 FEATURES AND CANONICAL ORDER ===")
    raw_df = pd.read_csv("data/student-por.csv", sep=";")
    dataset_cols = [c for c in raw_df.columns if c != "G3"]
    assert ORIGINAL_FEATURE_NAMES == dataset_cols, "Canonical feature list does not match dataset!"
    print(f"Original feature names match dataset exactly ({len(ORIGINAL_FEATURE_NAMES)} features).")

    print("\n=== 3. COMPARE PREDICTIONS ON 5 REAL DATASET RECORDS ===")
    test_indices = [0, 42, 120, 250, 500]
    for idx in test_indices:
        raw_row = raw_df.iloc[[idx]].drop(columns=["G3"])
        actual_g3 = raw_df.iloc[idx]["G3"]
        
        # Direct pipeline inference
        direct_pred = int(model.predict(raw_row)[0])
        direct_proba = model.predict_proba(raw_row)[0]
        
        # App predict_student inference
        app_res = predict_student(raw_row, model=model)
        app_pred = app_res["predicted_class"]
        app_proba = [app_res["probabilities"][i] for i in range(4)]
        
        assert direct_pred == app_pred, f"Mismatch in prediction for row {idx}!"
        assert np.allclose(direct_proba, app_proba, atol=1e-6), f"Mismatch in probas for row {idx}!"
        assert np.isclose(np.sum(app_proba), 1.0, atol=1e-5), f"Probas do not sum to 1 for row {idx}!"
        
        name = app_res["predicted_name"]
        conf = app_res["confidence"] * 100
        print(f"Row {idx:3d} (Actual G3={actual_g3:2d}): Pred={app_pred} ({name}), Conf={conf:.2f}%, Match=TRUE")

    print("\n=== 4. VERIFY INVALID INPUT HANDLING ===")
    invalid_dict = {"school": "GP", "sex": "F"} # Missing other 30 features
    try:
        predict_student(invalid_dict, model=model)
        raise AssertionError("ERROR: Missing cols did not raise ValueError!")
    except ValueError as e:
        print("Graceful failure verified for invalid input:", str(e)[:60] + "...")

    print("\n=== 5. VERIFY ACADEMIC RECOMMENDATIONS ===")
    sample_inputs = {"G1": 8, "G2": 9, "failures": 1, "studytime": 1, "absences": 12, "schoolsup": "no", "famsup": "no"}
    recs = generate_academic_recommendations(sample_inputs, 0)
    print(f"Generated {len(recs)} recommendations for at-risk sample:")
    for r in recs:
        print(f"  - [{r['type']}] {r['icon']} {r['title']}")
    assert len(recs) >= 3, "Expected at least 3 recommendations"

    print("\n=== 6. VERIFY COMPILATION OF ALL STREAMLIT PAGES ===")
    import py_compile, glob
    for f in glob.glob("pages/*.py") + ["app.py", "src/app_utils.py", "src/dashboard_components.py"]:
        py_compile.compile(f, doraise=True)
        print(f"  - Compiled: {f}")

    print("\n=======================================================")
    print("ALL PHASE 4B TEST SUITE CHECKS PASSED WITH 100% SUCCESS!")
    print("=======================================================")

if __name__ == "__main__":
    run_tests()
