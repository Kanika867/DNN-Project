"""
EduMetric — Academic Performance Prediction & Intelligence Platform
Module 1: Student Academic Performance Prediction Engine & Advisory
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.ui_theme import (
    BRAND_NAME,
    SEMANTIC_COLORS,
    CLASS_HEX_LIST,
    inject_global_theme,
    render_unified_sidebar,
    render_page_header,
    render_page_footer,
    apply_plotly_theme,
    render_safe_dataframe
)
from src.app_utils import (
    load_production_model,
    predict_student,
    generate_academic_recommendations,
    DEFAULT_CLASS_MAPPING,
    ORIGINAL_FEATURE_NAMES
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"Student Prediction | {BRAND_NAME}",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# INJECT UNIFIED DESIGN SYSTEM & APPLICATION SHELL
# -----------------------------------------------------------------------------
inject_global_theme()
render_unified_sidebar(current_page="prediction")

# -----------------------------------------------------------------------------
# HERO BANNER / PAGE HEADER
# -----------------------------------------------------------------------------
render_page_header(
    title="Student Academic Performance Prediction Engine",
    subtitle="Evaluate student performance cohorts using multi-dimensional demographic, behavioral, and interim academic markers. Generate calibrated probabilistic predictions and actionable advisory recommendations.",
    badge_text="Real-Time Inference & Educational Advisory",
    icon="🔮"
)

# Contextual guidance notice
st.markdown(
    f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #4F46E5; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.1rem;">💡</span>
            <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #0F172A; font-size: 0.92rem;">
                Calibrated Academic Inference Context
            </span>
        </div>
        <div style="color: #475569; font-size: 0.85rem; margin-top: 4px; line-height: 1.5;">
            The production <strong>Gradient Boosting Pipeline</strong> forecasts final examination outcomes (<strong>G3</strong>: <em>Poor</em>, <em>Average</em>, <em>Good</em>, or <em>Excellent</em>) from 32 holistic indicators. Load a sample preset below or enter specific student parameters, then click <strong>Generate Performance Forecast</strong>.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# SAMPLE PROFILE PRESETS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.92rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;">
        ⚡ Fast-Load Student Profile Presets
    </div>
    """,
    unsafe_allow_html=True
)

preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

PRESET_DEFAULT = {
    "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
    "Medu": 2, "Fedu": 2, "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
    "traveltime": 1, "studytime": 2, "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
    "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
    "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2, "health": 3, "absences": 2,
    "G1": 11, "G2": 11
}

PRESET_HIGH_ACHIEVER = {
    "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "GT3", "Pstatus": "T",
    "Medu": 4, "Fedu": 4, "Mjob": "health", "Fjob": "services", "reason": "reputation", "guardian": "mother",
    "traveltime": 1, "studytime": 3, "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
    "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
    "famrel": 5, "freetime": 3, "goout": 2, "Dalc": 1, "Walc": 1, "health": 4, "absences": 1,
    "G1": 16, "G2": 17
}

PRESET_AT_RISK = {
    "school": "MS", "sex": "M", "age": 18, "address": "R", "famsize": "LE3", "Pstatus": "A",
    "Medu": 1, "Fedu": 1, "Mjob": "at_home", "Fjob": "other", "reason": "other", "guardian": "other",
    "traveltime": 3, "studytime": 1, "failures": 2, "schoolsup": "no", "famsup": "no", "paid": "no",
    "activities": "no", "nursery": "no", "higher": "no", "internet": "no", "romantic": "yes",
    "famrel": 2, "freetime": 4, "goout": 4, "Dalc": 3, "Walc": 4, "health": 2, "absences": 14,
    "G1": 7, "G2": 8
}

if "student_form_state" not in st.session_state:
    st.session_state["student_form_state"] = PRESET_DEFAULT.copy()

with preset_col1:
    if st.button("👤 Average Baseline Profile", use_container_width=True):
        st.session_state["student_form_state"] = PRESET_DEFAULT.copy()
        st.session_state.pop("prediction_result", None)
        st.rerun()

with preset_col2:
    if st.button("🌟 High-Achieving Profile", use_container_width=True):
        st.session_state["student_form_state"] = PRESET_HIGH_ACHIEVER.copy()
        st.session_state.pop("prediction_result", None)
        st.rerun()

with preset_col3:
    if st.button("⚠️ Academic Support Profile", use_container_width=True):
        st.session_state["student_form_state"] = PRESET_AT_RISK.copy()
        st.session_state.pop("prediction_result", None)
        st.rerun()

with preset_col4:
    if st.button("🔄 Reset Profile Defaults", use_container_width=True):
        st.session_state["student_form_state"] = PRESET_DEFAULT.copy()
        st.session_state.pop("prediction_result", None)
        st.rerun()

vals = st.session_state["student_form_state"]

# -----------------------------------------------------------------------------
# STUDENT PROFILE INPUT FORM (32 CANONICAL FEATURES)
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

with st.form("student_prediction_form"):
    # SECTION A: PERSONAL & INSTITUTIONAL DEMOGRAPHICS
    st.markdown(
        """
        <div class="edum-section-heading" style="margin-top: 4px;">
            <span>🏛️</span>
            <span>Section A — Personal & Institutional Demographics</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    col_a1, col_a2, col_a3, col_a4, col_a5, col_a6 = st.columns(6)

    with col_a1:
        school = st.selectbox(
            "School Campus",
            options=["GP", "MS"],
            index=0 if vals.get("school") == "GP" else 1,
            help="GP: Gabriel Pereira | MS: Mousinho da Silveira"
        )
    with col_a2:
        sex = st.selectbox(
            "Gender",
            options=["F", "M"],
            index=0 if vals.get("sex") == "F" else 1,
            format_func=lambda x: "Female (F)" if x == "F" else "Male (M)",
            help="Student gender"
        )
    with col_a3:
        age = st.slider(
            "Student Age",
            min_value=15,
            max_value=22,
            value=int(vals.get("age", 17)),
            help="Student chronological age (15 to 22 years)"
        )
    with col_a4:
        address = st.selectbox(
            "Address Location",
            options=["U", "R"],
            index=0 if vals.get("address") == "U" else 1,
            format_func=lambda x: "Urban (U)" if x == "U" else "Rural (R)",
            help="Urban or Rural residential location"
        )
    with col_a5:
        famsize = st.selectbox(
            "Household Size",
            options=["GT3", "LE3"],
            index=0 if vals.get("famsize") == "GT3" else 1,
            format_func=lambda x: "> 3 Members (GT3)" if x == "GT3" else "≤ 3 Members (LE3)",
            help="Family household member count"
        )
    with col_a6:
        Pstatus = st.selectbox(
            "Parental Status",
            options=["T", "A"],
            index=0 if vals.get("Pstatus") == "T" else 1,
            format_func=lambda x: "Together (T)" if x == "T" else "Apart (A)",
            help="T: Living together | A: Living apart"
        )

    # SECTION B: FAMILY, SOCIAL & SUPPORT ENVIRONMENT
    st.markdown(
        """
        <div class="edum-section-heading">
            <span>👨‍👩‍👧</span>
            <span>Section B — Family Background & Academic Support Infrastructure</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)

    medu_options = {
        0: "0 — None",
        1: "1 — Primary (4th Grade)",
        2: "2 — 5th to 9th Grade",
        3: "3 — Secondary School",
        4: "4 — Higher Education"
    }

    with col_b1:
        Medu = st.selectbox(
            "Mother's Education (Medu)",
            options=[0, 1, 2, 3, 4],
            index=int(vals.get("Medu", 2)),
            format_func=lambda x: medu_options[x],
            help="Highest formal education level completed by mother"
        )
        Mjob = st.selectbox(
            "Mother's Profession (Mjob)",
            options=["at_home", "health", "other", "services", "teacher"],
            index=["at_home", "health", "other", "services", "teacher"].index(vals.get("Mjob", "other")),
            format_func=lambda x: x.replace("_", " ").capitalize(),
            help="Mother's occupational category"
        )
        schoolsup = st.selectbox(
            "School Academic Support (schoolsup)",
            options=["no", "yes"],
            index=0 if vals.get("schoolsup") == "no" else 1,
            format_func=lambda x: "Yes" if x == "yes" else "No",
            help="Enrolled in extra academic support programs at school"
        )
        activities = st.selectbox(
            "Extracurricular Activities",
            options=["no", "yes"],
            index=0 if vals.get("activities") == "no" else 1,
            format_func=lambda x: "Yes" if x == "yes" else "No",
            help="Participation in school sports, arts, or clubs"
        )

    with col_b2:
        Fedu = st.selectbox(
            "Father's Education (Fedu)",
            options=[0, 1, 2, 3, 4],
            index=int(vals.get("Fedu", 2)),
            format_func=lambda x: medu_options[x],
            help="Highest formal education level completed by father"
        )
        Fjob = st.selectbox(
            "Father's Profession (Fjob)",
            options=["at_home", "health", "other", "services", "teacher"],
            index=["at_home", "health", "other", "services", "teacher"].index(vals.get("Fjob", "other")),
            format_func=lambda x: x.replace("_", " ").capitalize(),
            help="Father's occupational category"
        )
        famsup = st.selectbox(
            "Family Educational Support (famsup)",
            options=["no", "yes"],
            index=0 if vals.get("famsup") == "no" else 1,
            format_func=lambda x: "Yes" if x == "yes" else "No",
            help="Academic assistance provided by family members"
        )
        nursery = st.selectbox(
            "Attended Nursery School",
            options=["no", "yes"],
            index=0 if vals.get("nursery") == "no" else 1,
            format_func=lambda x: "Yes" if x == "yes" else "No",
            help="Attended preschool or nursery prior to primary school"
        )

    with col_b3:
        guardian = st.selectbox(
            "Primary Legal Guardian",
            options=["mother", "father", "other"],
            index=["mother", "father", "other"].index(vals.get("guardian", "mother")),
            format_func=lambda x: x.capitalize(),
            help="Primary responsible legal guardian"
        )
        reason = st.selectbox(
            "Reason for Choosing School",
            options=["course", "home", "reputation", "other"],
            index=["course", "home", "reputation", "other"].index(vals.get("reason", "course")),
            format_func=lambda x: x.capitalize(),
            help="Primary factor determining school selection"
        )
        paid = st.selectbox(
            "Private Tutoring Classes (paid)",
            options=["no", "yes"],
            index=0 if vals.get("paid") == "no" else 1,
            format_func=lambda x: "Yes" if x == "yes" else "No",
            help="Enrolled in paid commercial tutoring courses"
        )
        higher = st.selectbox(
            "Higher Education Aspiration",
            options=["no", "yes"],
            index=0 if vals.get("higher") == "no" else 1,
            format_func=lambda x: "Yes" if x == "yes" else "No",
            help="Student aims to pursue university or higher education"
        )

    with col_b4:
        famrel = st.slider(
            "Family Relationship Quality (1-5)",
            min_value=1,
            max_value=5,
            value=int(vals.get("famrel", 4)),
            help="1: Poor | 3: Average | 5: Excellent"
        )
        internet = st.selectbox(
            "Home Internet Connectivity",
            options=["no", "yes"],
            index=0 if vals.get("internet") == "no" else 1,
            format_func=lambda x: "Yes" if x == "yes" else "No",
            help="Reliable home broadband internet access"
        )
        romantic = st.selectbox(
            "Romantic Relationship",
            options=["no", "yes"],
            index=0 if vals.get("romantic") == "no" else 1,
            format_func=lambda x: "Yes" if x == "yes" else "No",
            help="Involved in a romantic relationship"
        )

    # SECTION C: ACADEMIC HISTORY & INTERIM EVALUATIONS
    st.markdown(
        """
        <div class="edum-section-heading">
            <span>📚</span>
            <span>Section C — Academic History & Interim Term Marks (Core Signals)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    col_c1, col_c2, col_c3, col_c4, col_c5, col_c6 = st.columns(6)

    travel_options = {1: "< 15 min", 2: "15–30 min", 3: "30–60 min", 4: "> 60 min"}
    study_options = {1: "< 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "> 10 hours"}

    with col_c1:
        traveltime = st.selectbox(
            "Daily Commute Time",
            options=[1, 2, 3, 4],
            index=[1, 2, 3, 4].index(vals.get("traveltime", 1)),
            format_func=lambda x: travel_options[x],
            help="Transit duration from home to school"
        )
    with col_c2:
        studytime = st.selectbox(
            "Weekly Dedicated Study",
            options=[1, 2, 3, 4],
            index=[1, 2, 3, 4].index(vals.get("studytime", 2)),
            format_func=lambda x: study_options[x],
            help="Dedicated non-classroom weekly revision hours"
        )
    with col_c3:
        failures = st.selectbox(
            "Past Course Failures",
            options=[0, 1, 2, 3],
            index=int(vals.get("failures", 0)),
            format_func=lambda x: str(x) if x < 3 else "3 or more",
            help="Total count of previous academic course setbacks"
        )
    with col_c4:
        G1 = st.slider(
            "Period 1 Grade (G1)",
            min_value=0,
            max_value=20,
            value=int(vals.get("G1", 11)),
            help="First term evaluation mark (0–20 scale)"
        )
    with col_c5:
        G2 = st.slider(
            "Period 2 Grade (G2)",
            min_value=0,
            max_value=20,
            value=int(vals.get("G2", 11)),
            help="Second term evaluation mark (0–20 scale)"
        )
    with col_c6:
        absences = st.number_input(
            "Recorded School Absences",
            min_value=0,
            max_value=32,
            value=int(vals.get("absences", 2)),
            step=1,
            help="Total recorded school absences (days)"
        )

    # SECTION D: LIFESTYLE, BEHAVIORAL & WELLBEING FACTORS
    st.markdown(
        """
        <div class="edum-section-heading">
            <span>🌿</span>
            <span>Section D — Lifestyle, Social Habits & Student Wellbeing</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)

    with col_d1:
        freetime = st.slider(
            "Free Time After School",
            min_value=1,
            max_value=5,
            value=int(vals.get("freetime", 3)),
            help="1: Very low | 3: Moderate | 5: Very high"
        )
    with col_d2:
        goout = st.slider(
            "Socializing / Going Out",
            min_value=1,
            max_value=5,
            value=int(vals.get("goout", 3)),
            help="1: Very low | 3: Moderate | 5: Very high"
        )
    with col_d3:
        Dalc = st.slider(
            "Workday Alcohol (1-5)",
            min_value=1,
            max_value=5,
            value=int(vals.get("Dalc", 1)),
            help="1: Very low | 5: Very high"
        )
    with col_d4:
        Walc = st.slider(
            "Weekend Alcohol (1-5)",
            min_value=1,
            max_value=5,
            value=int(vals.get("Walc", 2)),
            help="1: Very low | 5: Very high"
        )
    with col_d5:
        health = st.slider(
            "Current Health Status",
            min_value=1,
            max_value=5,
            value=int(vals.get("health", 3)),
            help="1: Very bad | 3: Normal | 5: Very good"
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    col_sub1, col_sub2 = st.columns([1.5, 4])
    with col_sub1:
        submit_btn = st.form_submit_button(
            "🚀 Generate Performance Forecast",
            type="primary",
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# PREDICTION INFERENCE LOGIC (EXACT CANONICAL 32 FEATURES)
# -----------------------------------------------------------------------------
if submit_btn:
    input_payload = {
        "school": school,
        "sex": sex,
        "age": int(age),
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": int(Medu),
        "Fedu": int(Fedu),
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": int(traveltime),
        "studytime": int(studytime),
        "failures": int(failures),
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": int(famrel),
        "freetime": int(freetime),
        "goout": int(goout),
        "Dalc": int(Dalc),
        "Walc": int(Walc),
        "health": int(health),
        "absences": int(absences),
        "G1": int(G1),
        "G2": int(G2),
    }

    st.session_state["student_form_state"] = input_payload.copy()

    with st.spinner("Executing calibrated production pipeline inference..."):
        try:
            prod_model = load_production_model()
            if prod_model is None:
                st.error("Production model could not be loaded from `models/best_improved_model.joblib`.")
            else:
                result = predict_student(input_payload, model=prod_model)
                st.session_state["prediction_result"] = result
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# -----------------------------------------------------------------------------
# RENDER PREDICTION RESULTS & ADVISORY
# -----------------------------------------------------------------------------
if "prediction_result" in st.session_state:
    res = st.session_state["prediction_result"]
    pred_cls = res["predicted_class"]
    pred_name = res["predicted_name"]
    grade_range = res["grade_range"]
    conf = res["confidence"]
    probas = res["probabilities"]
    inputs = res["input_features"]

    sem = SEMANTIC_COLORS[pred_cls]

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="edum-section-heading">
            <span>📊</span>
            <span>Forecast Outcome & Performance Advisory</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    res_col_left, res_col_right = st.columns([1.15, 1.4])

    with res_col_left:
        st.markdown(
            f"""
            <div style="background-color: {sem['soft_bg']}; border: 1.5px solid {sem['border']}; border-radius: 14px; padding: 24px; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05); height: 100%;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: {sem['text']}; letter-spacing: 0.06em;">
                        Forecasted Outcome Bracket
                    </span>
                    <span style="font-size: 1.6rem;">{sem['icon']}</span>
                </div>
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.3rem; font-weight: 800; color: {sem['text']}; margin-top: 6px; letter-spacing: -0.03em;">
                    {pred_name}
                </div>
                <div style="font-size: 1.05rem; color: {sem['text']}; font-weight: 600; margin-top: 4px;">
                    Estimated Final Grade: <strong>G3 = {grade_range} Marks</strong> (0–20 Scale)
                </div>
                <hr style="margin: 14px 0; border: none; border-top: 1px solid {sem['border']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.85rem; color: {sem['text']}; font-weight: 600;">Calibrated Category Score</span>
                    <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.4rem; font-weight: 800; color: {sem['text']};">{conf * 100:.1f}%</span>
                </div>
                <div style="margin-top: 14px; background: rgba(255, 255, 255, 0.65); border: 1px solid {sem['border']}; border-radius: 8px; padding: 10px 12px; font-size: 0.78rem; color: #475569; line-height: 1.45;">
                    <strong>Advisory Guidance Note:</strong> This outcome represents statistical classification within historical student cohorts. Model outputs are probabilistic guidance indicators to empower targeted tutoring, not deterministic ceilings on student growth.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with res_col_right:
        cat_labels = ["Poor (0–9)", "Average (10–12)", "Good (13–15)", "Excellent (16–20)"]
        cat_probs = [probas.get(i, 0.0) * 100 for i in range(4)]

        fig_scores = go.Figure()
        fig_scores.add_trace(go.Bar(
            y=cat_labels,
            x=cat_probs,
            orientation="h",
            text=[f"<b>{p:.1f}%</b>" for p in cat_probs],
            textposition="outside",
            marker=dict(
                color=CLASS_HEX_LIST,
                line=dict(color="#0F172A", width=0.8)
            ),
            hovertemplate="<b>%{y}</b><br>Calibrated Model Probability: <b>%{x:.2f}%</b><extra></extra>"
        ))

        apply_plotly_theme(fig_scores, title="4-Class Calibrated Probability Distribution", height=255)
        fig_scores.update_layout(
            xaxis=dict(title="Estimated Probability (%)", range=[0, max(max(cat_probs) * 1.3, 45)]),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#334155")),
            margin=dict(l=10, r=40, t=40, b=30)
        )

        st.plotly_chart(fig_scores, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            "📌 *Probability scores reflect relative gradient-boosted likelihood across the 4 discrete grade categories. Scores sum to exactly 100.0%.*"
        )

    # SUBMITTED PROFILE SNAPSHOT
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.98rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;">
            📝 Submitted Student Profile Summary
        </div>
        """,
        unsafe_allow_html=True
    )

    sum_c1, sum_c2, sum_c3, sum_c4, sum_c5, sum_c6 = st.columns(6)
    with sum_c1:
        st.metric("Period 1 Grade", f"{inputs['G1']} / 20")
    with sum_c2:
        st.metric("Period 2 Grade", f"{inputs['G2']} / 20")
    with sum_c3:
        study_hours_txt = {1: "< 2 hrs", 2: "2–5 hrs", 3: "5–10 hrs", 4: "> 10 hrs"}[inputs['studytime']]
        st.metric("Weekly Study", study_hours_txt)
    with sum_c4:
        st.metric("Past Failures", inputs['failures'])
    with sum_c5:
        st.metric("Absences", f"{inputs['absences']} days")
    with sum_c6:
        st.metric("Higher Ed Aim", "Yes" if inputs['higher'] == "yes" else "No")

    with st.expander("Inspect Full 32-Feature Canonical Parameter Vector"):
        feature_df = pd.DataFrame([inputs]).T.reset_index()
        feature_df.columns = ["Feature Identifier", "Submitted Value"]
        render_safe_dataframe(feature_df, use_container_width=True, height=250)

    # ACTIONABLE ACADEMIC RECOMMENDATIONS
    st.markdown(
        """
        <div class="edum-section-heading">
            <span>💡</span>
            <span>Tailored Academic Support & Early Warning Guidance</span>
        </div>
        <p style="color: #64748B; font-size: 0.88rem; margin-top: -10px; margin-bottom: 14px;">
            Actionable educational support recommendations formulated strictly from academic habits, study duration, and attendance indicators. Sensitive demographic markers are strictly excluded.
        </p>
        """,
        unsafe_allow_html=True
    )

    recs = generate_academic_recommendations(inputs, pred_cls)
    rec_cols = st.columns(len(recs)) if len(recs) <= 3 else st.columns(3)

    for idx, rec in enumerate(recs):
        target_col = rec_cols[idx % len(rec_cols)]
        with target_col:
            border_c = "#E11D48" if rec["type"] == "warning" else ("#059669" if rec["type"] == "success" else "#4F46E5")
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid {border_c}; border-radius: 10px; padding: 16px; margin-bottom: 12px; height: 100%; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #0F172A; font-size: 0.94rem;">
                        {rec['icon']} {rec['title']}
                    </div>
                    <div style="color: #475569; font-size: 0.85rem; margin-top: 8px; line-height: 1.5;">
                        {rec['desc']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Clear Prediction & Enter New Student", type="secondary"):
        st.session_state.pop("prediction_result", None)
        st.rerun()

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
render_page_footer()
