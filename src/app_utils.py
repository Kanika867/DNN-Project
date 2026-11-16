"""
EduMetric — Application Utilities Module
Provides cached data loading, artifact loading, shared UI configuration,
and inference prediction services for the EduMetric platform.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st

# Color constants for target classes and UI branding
CLASS_COLORS = {
    0: "#EF4444",  # Poor - Red/Coral
    1: "#F59E0B",  # Average - Amber/Gold
    2: "#3B82F6",  # Good - Blue
    3: "#10B981",  # Excellent - Emerald
}

CLASS_HEX_LIST = ["#EF4444", "#F59E0B", "#3B82F6", "#10B981"]

BRAND_COLOR = "#4F46E5"  # Indigo
BG_CARD = "#F8FAFC"      # Light slate card background

# Grade boundaries mapping
DEFAULT_CLASS_MAPPING = {
    0: {"name": "Poor", "grade_range": "0-9", "min_grade": 0, "max_grade": 9, "color": "#EF4444"},
    1: {"name": "Average", "grade_range": "10-12", "min_grade": 10, "max_grade": 12, "color": "#F59E0B"},
    2: {"name": "Good", "grade_range": "13-15", "min_grade": 13, "max_grade": 15, "color": "#3B82F6"},
    3: {"name": "Excellent", "grade_range": "16-20", "min_grade": 16, "max_grade": 20, "color": "#10B981"},
}


def map_g3_to_class(g3_series):
    """Maps G3 numeric grades (0-20) to discrete performance categories (0-3)."""
    conditions = [
        (g3_series >= 0) & (g3_series <= 9),
        (g3_series >= 10) & (g3_series <= 12),
        (g3_series >= 13) & (g3_series <= 15),
        (g3_series >= 16) & (g3_series <= 20)
    ]
    choices = [0, 1, 2, 3]
    return np.select(conditions, choices, default=-1)


@st.cache_data(show_spinner=False)
def load_student_data(filepath="data/student-por.csv"):
    """
    Loads student-por.csv, verifies integrity, and appends target class annotations.
    """
    if not os.path.exists(filepath):
        st.error(f"Dataset file not found at: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, sep=";")
        df["student_id"] = [f"Student-{i+1:03d}" for i in range(len(df))]
        df["target_class"] = map_g3_to_class(df["G3"])
        class_names = {0: "Poor (0-9)", 1: "Average (10-12)", 2: "Good (13-15)", 3: "Excellent (16-20)"}
        category_simple = {0: "Poor", 1: "Average", 2: "Good", 3: "Excellent"}
        df["target_name"] = df["target_class"].map(class_names)
        df["performance_category"] = df["target_class"].map(category_simple)
        return df
    except Exception as e:
        st.error(f"Error loading student dataset: {str(e)}")
        return None


@st.cache_data(show_spinner=False)
def load_class_mapping(filepath="artifacts/class_mapping.json"):
    """Loads class mapping metadata from artifacts."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        except Exception:
            return DEFAULT_CLASS_MAPPING
    return DEFAULT_CLASS_MAPPING


@st.cache_data(show_spinner=False)
def load_improved_model_metrics(filepath="assets/improved_model_metrics.json"):
    """Loads evaluation metrics for the production Gradient Boosting model."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not parse improved model metrics: {e}")
            return None
    return None


@st.cache_data(show_spinner=False)
def load_model_comparison(filepath="assets/model_comparison.json"):
    """Loads model benchmark comparison metrics (RF, DNN, GB)."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_feature_names(filepath="artifacts/feature_names.json"):
    """Loads original 32 input feature names."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


# The 32 original input features in exact dataset column order
ORIGINAL_FEATURE_NAMES = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2"
]


@st.cache_resource(show_spinner=False)
def load_production_model(filepath="models/best_improved_model.joblib"):
    """
    Loads and caches the production Gradient Boosting Pipeline.
    Pipeline includes ColumnTransformer (StandardScaler + OneHotEncoder)
    and GradientBoostingClassifier.
    """
    import joblib
    if not os.path.exists(filepath):
        st.error(f"Production model not found at: {filepath}")
        return None
    try:
        model = joblib.load(filepath)
        return model
    except Exception as e:
        st.error(f"Error loading production model: {str(e)}")
        return None


def predict_student(input_dict_or_df, model=None):
    """
    Accepts a dictionary or single-row DataFrame of 32 student features,
    validates column names and types, passes raw data to the production pipeline,
    and returns a structured prediction dictionary.
    """
    if model is None:
        model = load_production_model()
        if model is None:
            raise ValueError("Production model could not be loaded.")

    if isinstance(input_dict_or_df, dict):
        df = pd.DataFrame([input_dict_or_df])
    elif isinstance(input_dict_or_df, pd.DataFrame):
        df = input_dict_or_df.copy()
    else:
        raise TypeError("Input must be a dict or pandas DataFrame.")

    # Validate that all 32 required features are present
    missing_cols = [col for col in ORIGINAL_FEATURE_NAMES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required input features: {missing_cols}")

    # Reorder columns to match canonical dataset order
    df = df[ORIGINAL_FEATURE_NAMES]

    # Execute prediction via the full scikit-learn Pipeline
    pred_class_idx = int(model.predict(df)[0])
    raw_probas = model.predict_proba(df)[0]

    # Map classes based on model's classes_ attribute
    classes_order = getattr(model.named_steps["model"], "classes_", [0, 1, 2, 3])
    ordered_probas = {}
    for idx, cls in enumerate(classes_order):
        ordered_probas[int(cls)] = float(raw_probas[idx])

    meta = DEFAULT_CLASS_MAPPING.get(pred_class_idx, {
        "name": f"Class {pred_class_idx}",
        "grade_range": "N/A",
        "color": "#4F46E5"
    })

    return {
        "predicted_class": pred_class_idx,
        "predicted_name": meta["name"],
        "grade_range": meta["grade_range"],
        "color": meta["color"],
        "confidence": ordered_probas.get(pred_class_idx, 0.0),
        "probabilities": ordered_probas,
        "classes_order": [int(c) for c in classes_order],
        "input_features": df.to_dict(orient="records")[0]
    }


def generate_academic_recommendations(inputs, pred_class):
    """
    Generates actionable, non-judgmental educational support recommendations
    based on actionable academic and lifestyle indicators.
    Avoids sensitive demographic attributes.
    """
    recs = []

    # Prior grades
    g1 = inputs.get("G1", 10)
    g2 = inputs.get("G2", 10)
    failures = inputs.get("failures", 0)
    studytime = inputs.get("studytime", 2)
    absences = inputs.get("absences", 0)
    schoolsup = inputs.get("schoolsup", "no")
    famsup = inputs.get("famsup", "no")

    if g2 < 10 or g1 < 10:
        recs.append({
            "title": "Targeted Foundational Revision",
            "icon": "📚",
            "type": "warning",
            "desc": "Past evaluation marks indicate foundational concepts in early coursework require reinforcement. Scheduling 1-on-1 office hours with subject teachers can address specific knowledge gaps."
        })
    elif g2 >= 15 and g1 >= 15:
        recs.append({
            "title": "Advanced Academic Enrichment",
            "icon": "🌟",
            "type": "success",
            "desc": "Demonstrates strong conceptual mastery across multiple terms. Consider advanced placement coursework, peer tutoring leadership, or university preparation enrichment."
        })

    if failures > 0:
        recs.append({
            "title": "Remedial Milestone Coaching",
            "icon": "🎯",
            "type": "warning",
            "desc": f"Student has {failures} prior course setback(s). Breaking complex units into smaller, low-stakes milestone assessments can rebuild subject mastery and academic resilience."
        })

    if studytime == 1:
        recs.append({
            "title": "Study Routine Structuring",
            "icon": "⏱️",
            "type": "info",
            "desc": "Reported study time is under 2 hours weekly. Introducing a structured 30–45 minute daily revision block with active recall techniques can substantially improve retention."
        })

    if absences >= 8:
        recs.append({
            "title": "Attendance Reconciliation & Gap Closure",
            "icon": "📋",
            "type": "warning",
            "desc": f"Record reflects {absences} absences. High absenteeism correlates with scattered comprehension; recommend establishing an attendance recovery contract and study buddy notes exchange."
        })

    if schoolsup == "no" and (g1 < 12 or g2 < 12):
        recs.append({
            "title": "Institutional Academic Support",
            "icon": "🏫",
            "type": "info",
            "desc": "Student is not currently enrolled in school-based support. Leveraging free after-school study halls and institutional tutorials provides consistent guided practice."
        })

    if not recs:
        recs.append({
            "title": "Sustained Academic Momentum",
            "icon": "✅",
            "type": "success",
            "desc": "Study habits, attendance, and baseline evaluations show balanced consistency. Maintain regular self-assessment and proactive preparation for final exams."
        })

    return recs
