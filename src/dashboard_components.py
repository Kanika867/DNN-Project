"""
EduMetric — Dashboard Components Module
Provides modular, enterprise-grade UI components and Plotly charts
for the EduMetric Academic Intelligence Platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui_theme import (
    BRAND_NAME,
    COLOR_NAVY_DARK,
    COLOR_NAVY_CARD,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED,
    COLOR_ACCENT_PRIMARY,
    SEMANTIC_COLORS,
    CLASS_HEX_LIST,
    apply_plotly_theme
)
from src.app_utils import DEFAULT_CLASS_MAPPING


# =============================================================================
# 1. OVERVIEW DASHBOARD COMPONENTS (app.py)
# =============================================================================

def render_kpi_cards(total_records=649, feature_count=32, class_count=4,
                     model_name="Gradient Boosting", test_acc="72.45%", test_macro_f1="0.7516"):
    """
    Renders 6 styled metric KPI cards in a responsive grid for the Overview Dashboard.
    """
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    cards = [
        (col1, "COHORT POPULATION", str(total_records), "649 Students Analyzed", "#4F46E5", "👥"),
        (col2, "CANONICAL FEATURES", str(feature_count), "15 Num · 17 Categorical", "#2563EB", "📐"),
        (col3, "PERFORMANCE BRACKETS", str(class_count), "Poor to Excellent", "#D97706", "🏷️"),
        (col4, "PRODUCTION PIPELINE", model_name, "Calibrated Ensemble", "#059669", "⚙️"),
        (col5, "HELD-OUT ACCURACY", test_acc, "Outer Test (N = 98)", "#059669", "🎯"),
        (col6, "TEST MACRO F1", test_macro_f1, "Balanced 4-Class Score", "#4F46E5", "⚖️"),
    ]

    for col, label, value, subtext, accent, icon in cards:
        with col:
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid {accent}; border-radius: 12px; padding: 16px 14px; text-align: left; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04); transition: transform 0.15s ease, box-shadow 0.15s ease;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.7rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em;">{label}</span>
                        <span style="font-size: 1rem;">{icon}</span>
                    </div>
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.55rem; font-weight: 800; color: #0F172A; margin-top: 6px; letter-spacing: -0.02em;">
                        {value}
                    </div>
                    <div style="display: inline-block; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 9999px; padding: 2px 8px; font-size: 0.7rem; font-weight: 600; color: #475569; margin-top: 6px;">
                        {subtext}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_target_distribution_chart(df):
    """
    Creates an interactive Plotly bar breakdown for the 4 performance categories.
    """
    counts = df["target_class"].value_counts().sort_index()
    names = ["Poor (0–9)", "Average (10–12)", "Good (13–15)", "Excellent (16–20)"]
    pcts = (counts / len(df) * 100).round(1)

    chart_df = pd.DataFrame({
        "Category": names,
        "Class": counts.index,
        "Students": counts.values,
        "Percentage": pcts.values,
        "Color": CLASS_HEX_LIST
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=chart_df["Category"],
        y=chart_df["Students"],
        text=[f"<b>{val}</b> ({pct}%)" for val, pct in zip(chart_df["Students"], chart_df["Percentage"])],
        textposition="outside",
        marker=dict(
            color=CLASS_HEX_LIST,
            line=dict(color="#0F172A", width=0.8)
        ),
        hovertemplate="<b>%{x}</b><br>Cohort Count: <b>%{y} students</b><br>Proportion: <b>%{text}</b><extra></extra>"
    ))

    apply_plotly_theme(
        fig,
        title="Student Academic Performance Distribution (G3 Target Outcome)",
        height=320
    )
    fig.update_layout(
        xaxis=dict(title=None, tickfont=dict(size=11, color="#334155", family="Inter, sans-serif")),
        yaxis=dict(title="Enrolled Students", range=[0, max(chart_df["Students"]) * 1.18]),
        margin=dict(l=35, r=20, t=50, b=35)
    )
    return fig


def render_eda_summary_cards():
    """
    Renders structured exploratory findings cards distinguishing correlation from causation.
    """
    st.markdown(
        """
        <div class="edum-section-heading">
            <span>🔍</span>
            <span>Key Exploratory & Empirical Insights</span>
        </div>
        <p style="color: #64748B; font-size: 0.9rem; margin-top: -10px; margin-bottom: 16px;">
            Evidence-based observations derived from systematic data auditing across the 649-student Portuguese secondary cohort.
            <em>Note: Observed statistical associations highlight longitudinal correlation, not unilateral mechanical causation.</em>
        </p>
        """,
        unsafe_allow_html=True
    )

    findings = [
        ("1. Prior Academic Trajectory Dominance (G1 & G2)",
         "Period 1 (<strong>G1</strong>, r = +0.83) and Period 2 (<strong>G2</strong>, r = +0.92) interim evaluations account for over <strong>92%</strong> of built-in predictive importance. Students who sustain stable revision routines through mid-term rarely drop more than one category bracket.",
         "#4F46E5", "📈"),
        ("2. Weekly Dedicated Study Hours",
         "Weekly study duration displays a distinct positive association (r = +0.25). Students dedicating <strong>&gt; 5 hours weekly</strong> achieve Good or Excellent grades at over 2.4× the rate of students studying under 2 hours.",
         "#059669", "⏱️"),
        ("3. Prior Academic Setbacks as Diagnostic Signal",
         "Historical course failures constitute the strongest single non-grade risk factor (r = -0.39). Students with <strong>&ge; 1 previous failures</strong> have an 81% empirical probability of completing in Poor or Average brackets without targeted tutoring.",
         "#E11D48", "⚠️"),
        ("4. Attendance Regularity & Absenteeism",
         "Absences exhibit an inflection threshold. Moderate absences (1–4 days) reflect routine variation, whereas chronic absenteeism (<strong>&gt; 15 days</strong>, 21 cohort outliers) shows substantial grade dispersion and elevated failure probability.",
         "#D97706", "📋"),
        ("5. Digital Infrastructure & Household Resources",
         "Home internet connectivity is present for 83.7% of students. Students with reliable digital access demonstrate 1.8× greater representation in upper proficiency tiers, highlighting the structural value of digital educational equity.",
         "#2563EB", "🌐"),
        ("6. Parental Educational Attainment",
         "Maternal education (Medu, r = +0.24) and Paternal education (Fedu, r = +0.21) correlate positively with final marks, reflecting the compounding institutional advantages of supportive home learning environments.",
         "#7C3AED", "🎓"),
    ]

    col1, col2 = st.columns(2)
    for idx, (title, desc, border_color, icon) in enumerate(findings):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid {border_color}; border-radius: 10px; padding: 16px 18px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.1rem;">{icon}</span>
                        <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #0F172A; font-size: 0.95rem;">{title}</div>
                    </div>
                    <div style="color: #475569; font-size: 0.86rem; margin-top: 8px; line-height: 1.5;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_model_snapshot(metrics=None):
    """
    Renders an architectural snapshot of the selected Gradient Boosting production pipeline.
    """
    st.markdown(
        """
        <div class="edum-section-heading">
            <span>🏆</span>
            <span>Production Inference Architecture</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.25, 1])

    with col1:
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 22px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.15rem; font-weight: 700; color: #0F172A;">Gradient Boosting Pipeline</span>
                        <div style="font-size: 0.78rem; color: #64748B; margin-top: 2px;">Production Pipeline: <code>models/best_improved_model.joblib</code></div>
                    </div>
                    <span style="background-color: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; font-size: 0.74rem; font-weight: 700; padding: 4px 10px; border-radius: 9999px;">Active Production Model</span>
                </div>
                <div style="color: #475569; font-size: 0.85rem; margin-top: 10px; line-height: 1.5;">
                    Self-contained Scikit-Learn <code>Pipeline</code> encapsulating a leak-free <code>ColumnTransformer</code> (StandardScaler + OneHotEncoder) and regularized shallow-depth boosting.
                </div>
                <hr style="margin: 14px 0; border: none; border-top: 1px solid #F1F5F9;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div>
                        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600;">Algorithm Architecture</div>
                        <div style="font-size: 0.92rem; font-weight: 700; color: #0F172A; margin-top: 2px;">Regularized Gradient Boosting</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600;">Optimal Hyperparameters</div>
                        <div style="font-size: 0.92rem; font-weight: 700; color: #0F172A; margin-top: 2px;">100 Trees · Depth 2 · LR 0.05</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600;">Outer Test Accuracy</div>
                        <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800; color: #059669; margin-top: 2px;">72.45%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600;">Outer Test Macro F1</div>
                        <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800; color: #4F46E5; margin-top: 2px;">0.7516</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px; padding: 22px; height: 100%; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.98rem; font-weight: 700; color: #0F172A;">
                    Validation & Engineering Rationale
                </div>
                <ul style="font-size: 0.85rem; color: #475569; margin-top: 10px; padding-left: 18px; line-height: 1.55;">
                    <li><strong>Superior Generalization:</strong> Outperformed Deep Neural Network (68.37% acc) and Random Forest (66.33% acc) across held-out test records.</li>
                    <li><strong>Zero Severe Errors:</strong> Produced <strong>0 severe misclassifications</strong> (|Error| &gt; 1 bracket) on the outer test set; 100% of residual errors are adjacent.</li>
                    <li><strong>Optimal Calibration:</strong> Achieved lowest log loss (0.4692) and Brier score (0.2688) among 10 evaluated candidate pipelines.</li>
                    <li><strong>Zero Native C-Extension Friction:</strong> Executes natively in Scikit-Learn without local TensorFlow security blockages.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_disclaimer_box():
    """
    Renders the ethical disclaimer and responsible AI governance advisory.
    """
    st.markdown(
        f"""
        <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-left: 5px solid #D97706; border-radius: 12px; padding: 18px 22px; margin-top: 24px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.3rem;">🛡️</span>
                <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1rem; font-weight: 700; color: #92400E;">
                    {BRAND_NAME} Responsible AI Governance & Ethical Usage Charter
                </span>
            </div>
            <div style="font-size: 0.85rem; color: #78350F; margin-top: 8px; line-height: 1.55;">
                • <strong>Early Advisory Mandate:</strong> {BRAND_NAME} is engineered exclusively as an exploratory diagnostic decision-support system for academic advisors, counselors, and educators. It must <strong>never</strong> be utilized for punitive tracking, admissions gating, or automated disciplinary determinations.<br>
                • <strong>Dataset Scope & Boundaries:</strong> Models are calibrated on secondary school cohorts in Portugal enrolled in Portuguese language curricula. Cultural, curricular, and grading parameters vary across external jurisdictions.<br>
                • <strong>Human-in-the-Loop Oversight:</strong> Model outputs represent calibrated statistical cohort tendencies, never fixed determinations of an individual student's intellectual capability or personal potential.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# 2. STUDENT EXPLORER COMPONENTS (pages/2_Student_Explorer.py)
# =============================================================================

def render_explorer_kpi_cards(total_records=649, filtered_records=649,
                              avg_g3=11.91, median_g3=12.0,
                              school_count=2, class_count=4):
    """
    Renders 6 styled KPI cards for the Student Explorer,
    distinguishing full dataset statistics from filtered subset statistics.
    """
    pct_filtered = (filtered_records / total_records * 100) if total_records > 0 else 0
    avg_str = f"{avg_g3:.2f}" if avg_g3 is not None else "N/A"
    med_str = f"{median_g3:.1f}" if median_g3 is not None else "N/A"

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    cards = [
        (col1, "TOTAL COHORT", str(total_records), "Full Registry", "#4F46E5"),
        (col2, "FILTERED COHORT", str(filtered_records), f"{pct_filtered:.1f}% Active", "#2563EB" if filtered_records < total_records else "#059669"),
        (col3, "FILTERED MEAN G3", avg_str, "0–20 Scale", "#059669"),
        (col4, "FILTERED MEDIAN G3", med_str, "50th Percentile", "#7C3AED"),
        (col5, "SCHOOLS ACTIVE", str(school_count), "GP / MS Campuses", "#0F172A"),
        (col6, "CLASSES ACTIVE", str(class_count), "Target Brackets", "#D97706"),
    ]

    for col, label, value, subtext, color in cards:
        with col:
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid {color}; border-radius: 12px; padding: 14px 12px; text-align: left; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">
                    <div style="font-size: 0.68rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em;">{label}</div>
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.45rem; font-weight: 800; color: #0F172A; margin-top: 4px; letter-spacing: -0.02em;">
                        {value}
                    </div>
                    <div style="font-size: 0.72rem; font-weight: 600; color: {color}; margin-top: 4px;">
                        {subtext}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_filtered_distribution_chart(df_filtered):
    """
    Renders an interactive Plotly bar chart for the performance categories
    of the currently filtered student records.
    """
    if len(df_filtered) == 0:
        return None

    cat_order = [0, 1, 2, 3]
    cat_names = ["Poor (0–9)", "Average (10–12)", "Good (13–15)", "Excellent (16–20)"]
    counts = []
    pcts = []
    for cls in cat_order:
        cnt = int((df_filtered["target_class"] == cls).sum())
        counts.append(cnt)
        pcts.append((cnt / len(df_filtered) * 100) if len(df_filtered) > 0 else 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cat_names,
        y=counts,
        text=[f"<b>{cnt}</b> ({pct:.1f}%)" for cnt, pct in zip(counts, pcts)],
        textposition="outside",
        marker=dict(color=CLASS_HEX_LIST, line=dict(color="#0F172A", width=0.8)),
        hovertemplate="<b>%{x}</b><br>Filtered Students: <b>%{y}</b><br>Proportion: <b>%{text}</b><extra></extra>"
    ))

    max_c = max(counts) if max(counts) > 0 else 10
    apply_plotly_theme(
        fig,
        title=f"Performance Category Distribution (Filtered Cohort N = {len(df_filtered):,})",
        height=300
    )
    fig.update_layout(
        xaxis=dict(title=None, tickfont=dict(size=11, color="#334155")),
        yaxis=dict(title="Student Count", range=[0, max_c * 1.25]),
        margin=dict(l=35, r=20, t=45, b=35)
    )
    return fig


def render_comparative_charts(df_filtered):
    """
    Generates comparative visual analytics for the filtered subset:
    1. Average G3 by School
    2. Average G3 by Study Time Tier
    3. Performance Category Breakdown by Gender
    4. G3 Distribution by Past Failures (Box Plot)
    5. G3 vs Absences (Scatter / Trend)
    """
    if len(df_filtered) == 0:
        return {}

    charts = {}

    # 1. School Comparison
    school_stats = df_filtered.groupby("school")["G3"].agg(["mean", "count"]).reset_index()
    fig_school = go.Figure()
    fig_school.add_trace(go.Bar(
        x=school_stats["school"],
        y=school_stats["mean"].round(2),
        text=[f"<b>{m:.2f}</b> (N={c})" for m, c in zip(school_stats["mean"], school_stats["count"])],
        textposition="outside",
        marker=dict(color=["#4F46E5", "#0EA5E9"][:len(school_stats)], line=dict(color="#0F172A", width=0.8)),
        hovertemplate="School: <b>%{x}</b><br>Mean Grade G3: <b>%{y}</b><extra></extra>"
    ))
    apply_plotly_theme(fig_school, title="Mean Final Grade by Institution", height=280)
    fig_school.update_layout(yaxis=dict(title="Mean G3", range=[0, 20]))
    charts["school"] = fig_school

    # 2. Study Time Comparison
    st_names = {1: "< 2 hrs", 2: "2–5 hrs", 3: "5–10 hrs", 4: "> 10 hrs"}
    st_stats = df_filtered.groupby("studytime")["G3"].agg(["mean", "count"]).reset_index()
    st_stats["Tier"] = st_stats["studytime"].map(st_names)
    fig_st = go.Figure()
    fig_st.add_trace(go.Bar(
        x=st_stats["Tier"],
        y=st_stats["mean"].round(2),
        text=[f"<b>{m:.2f}</b> (N={c})" for m, c in zip(st_stats["mean"], st_stats["count"])],
        textposition="outside",
        marker=dict(color="#059669", line=dict(color="#0F172A", width=0.8)),
        hovertemplate="Study Tier: <b>%{x}</b><br>Mean Grade G3: <b>%{y}</b><extra></extra>"
    ))
    apply_plotly_theme(fig_st, title="Mean Final Grade by Weekly Study Tier", height=280)
    fig_st.update_layout(yaxis=dict(title="Mean G3", range=[0, 20]))
    charts["studytime"] = fig_st

    # 3. Gender Comparison
    gender_ct = pd.crosstab(df_filtered["sex"], df_filtered["target_class"], normalize="index") * 100
    fig_gender = go.Figure()
    names = ["Poor", "Average", "Good", "Excellent"]
    for col_idx in range(4):
        if col_idx in gender_ct.columns:
            fig_gender.add_trace(go.Bar(
                name=names[col_idx],
                x=["Female", "Male"],
                y=[gender_ct.loc["F", col_idx] if "F" in gender_ct.index else 0,
                   gender_ct.loc["M", col_idx] if "M" in gender_ct.index else 0],
                marker_color=CLASS_HEX_LIST[col_idx]
            ))
    apply_plotly_theme(fig_gender, title="Category Share by Gender (%)", height=280, show_legend=True)
    fig_gender.update_layout(barmode="stack", yaxis=dict(title="Share (%)", range=[0, 100]))
    charts["gender"] = fig_gender

    # 4. Failures Box Plot
    fig_fail = px.box(
        df_filtered,
        x="failures",
        y="G3",
        color="failures",
        labels={"failures": "Past Failures", "G3": "Final Grade G3"},
        color_discrete_sequence=["#059669", "#D97706", "#E11D48", "#9F1239"]
    )
    apply_plotly_theme(fig_fail, title="Grade Dispersion by Past Academic Failures", height=280)
    fig_fail.update_layout(yaxis=dict(title="G3 Grade (0–20)", range=[-0.5, 20.5]))
    charts["failures"] = fig_fail

    # 5. Absences Scatter
    fig_abs = px.scatter(
        df_filtered,
        x="absences",
        y="G3",
        color="performance_category",
        category_orders={"performance_category": ["Poor", "Average", "Good", "Excellent"]},
        color_discrete_map={"Poor": "#E11D48", "Average": "#D97706", "Good": "#2563EB", "Excellent": "#059669"},
        labels={"absences": "Recorded Absences (Days)", "G3": "Final Grade G3"}
    )
    apply_plotly_theme(fig_abs, title="Final Grade G3 vs. School Absences", height=280)
    charts["absences"] = fig_abs

    return charts


# =============================================================================
# 3. PERFORMANCE ANALYTICS COMPONENTS (pages/3_Performance_Analytics.py)
# =============================================================================

def render_analytics_kpi_cards(df):
    """
    Renders 6 dynamic KPI cards for Performance Analytics calculated from the active dataset.
    """
    total_students = len(df)
    if total_students == 0:
        return

    avg_g3 = float(df["G3"].mean())
    med_g3 = float(df["G3"].median())
    std_g3 = float(df["G3"].std()) if total_students > 1 else 0.0
    good_exc_cnt = int((df["target_class"] >= 2).sum())
    good_exc_pct = (good_exc_cnt / total_students) * 100
    fail_cnt = int((df["failures"] > 0).sum())
    fail_pct = (fail_cnt / total_students) * 100

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    cards = [
        (col1, "ANALYZED COHORT", f"{total_students:,}", "Active Population", "#4F46E5"),
        (col2, "COHORT MEAN G3", f"{avg_g3:.2f}", "0–20 Grade Scale", "#2563EB"),
        (col3, "COHORT MEDIAN G3", f"{med_g3:.1f}", "50th Percentile", "#7C3AED"),
        (col4, "GRADE DISPERSION", f"±{std_g3:.2f}", "Standard Deviation", "#64748B"),
        (col5, "PROFICIENT RATIO", f"{good_exc_pct:.1f}%", f"{good_exc_cnt} Students (G3 ≥ 13)", "#059669"),
        (col6, "SETBACK RISK RATIO", f"{fail_pct:.1f}%", f"{fail_cnt} Students (Failures ≥ 1)", "#E11D48"),
    ]

    for col, label, value, subtext, color in cards:
        with col:
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid {color}; border-radius: 12px; padding: 14px 12px; text-align: left; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">
                    <div style="font-size: 0.68rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em;">{label}</div>
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.45rem; font-weight: 800; color: #0F172A; margin-top: 4px; letter-spacing: -0.02em;">
                        {value}
                    </div>
                    <div style="font-size: 0.72rem; font-weight: 600; color: {color}; margin-top: 4px;">
                        {subtext}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_g3_distribution_histogram(df):
    """
    Renders an interactive histogram of G3 with clearly marked category boundaries
    and vertical dashed reference lines for mean and median.
    """
    avg_g3 = float(df["G3"].mean())
    med_g3 = float(df["G3"].median())

    g3_vals = sorted(df["G3"].unique())
    bar_colors = []
    for g in g3_vals:
        if g <= 9:
            bar_colors.append(SEMANTIC_COLORS[0]["primary"])
        elif g <= 12:
            bar_colors.append(SEMANTIC_COLORS[1]["primary"])
        elif g <= 15:
            bar_colors.append(SEMANTIC_COLORS[2]["primary"])
        else:
            bar_colors.append(SEMANTIC_COLORS[3]["primary"])

    counts = [int((df["G3"] == g).sum()) for g in g3_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=g3_vals,
        y=counts,
        text=[f"<b>{c}</b>" for c in counts],
        textposition="outside",
        marker=dict(color=bar_colors, line=dict(color="#0F172A", width=0.8)),
        hovertemplate="Grade G3: <b>%{x}</b><br>Cohort Count: <b>%{y} students</b><extra></extra>"
    ))

    fig.add_vline(
        x=avg_g3, line_width=2, line_dash="dash", line_color="#DC2626",
        annotation_text=f"Mean: {avg_g3:.2f}", annotation_position="top left",
        annotation_font=dict(color="#DC2626", size=10, family="Inter, sans-serif")
    )
    fig.add_vline(
        x=med_g3, line_width=2, line_dash="dot", line_color="#4F46E5",
        annotation_text=f"Median: {med_g3:.1f}", annotation_position="top right",
        annotation_font=dict(color="#4F46E5", size=10, family="Inter, sans-serif")
    )

    apply_plotly_theme(
        fig,
        title="Final Examination Grade (G3) Distribution & Grade Category Bands",
        height=320
    )
    fig.update_layout(
        xaxis=dict(title="G3 Final Grade (0–20 Scale)", dtick=1, range=[-0.5, 20.5]),
        yaxis=dict(title="Student Count", range=[0, max(counts) * 1.25])
    )
    return fig


def render_category_box_plots(df):
    """
    Renders interactive box plots showing G3 distribution across the 4 performance categories.
    """
    fig = px.box(
        df,
        x="performance_category",
        y="G3",
        color="performance_category",
        category_orders={"performance_category": ["Poor", "Average", "Good", "Excellent"]},
        color_discrete_map={"Poor": "#E11D48", "Average": "#D97706", "Good": "#2563EB", "Excellent": "#059669"},
        labels={"performance_category": "Category", "G3": "Final Grade G3"},
        points="all"
    )
    apply_plotly_theme(
        fig,
        title="Grade Distribution Dispersion by Target Category",
        height=320
    )
    fig.update_layout(yaxis=dict(range=[-0.5, 20.5]))
    return fig


def render_previous_grades_scatters(df):
    """
    Renders 3 interactive scatter plots with trend lines:
    G1 vs G3, G2 vs G3, and G1 vs G2 with calculated Pearson r correlations.
    """
    r_g1_g3 = float(df["G1"].corr(df["G3"]))
    r_g2_g3 = float(df["G2"].corr(df["G3"]))
    r_g1_g2 = float(df["G1"].corr(df["G2"]))

    def build_scatter_with_trend(x_col, y_col, title_txt, r_val):
        fig = px.scatter(
            df, x=x_col, y=y_col, color="performance_category",
            category_orders={"performance_category": ["Poor", "Average", "Good", "Excellent"]},
            color_discrete_map={"Poor": "#E11D48", "Average": "#D97706", "Good": "#2563EB", "Excellent": "#059669"},
            labels={x_col: x_col, y_col: y_col, "performance_category": "Category"}
        )
        if len(df) > 1:
            try:
                poly = np.polyfit(df[x_col], df[y_col], 1)
                x_trend = np.linspace(df[x_col].min(), df[x_col].max(), 50)
                y_trend = np.polyval(poly, x_trend)
                fig.add_trace(go.Scatter(
                    x=x_trend, y=y_trend, mode="lines",
                    name=f"Trend (r={r_val:.2f})",
                    line=dict(color="#0F172A", width=1.8, dash="dash")
                ))
            except Exception:
                pass

        apply_plotly_theme(fig, title=title_txt, height=270)
        fig.update_layout(
            xaxis=dict(range=[-0.5, 20.5]),
            yaxis=dict(range=[-0.5, 20.5]),
            margin=dict(l=30, r=20, t=40, b=30)
        )
        return fig

    return {
        "g1_g3": build_scatter_with_trend("G1", "G3", f"G1 vs G3 (r = {r_g1_g3:.2f})", r_g1_g3),
        "g2_g3": build_scatter_with_trend("G2", "G3", f"G2 vs G3 (r = {r_g2_g3:.2f})", r_g2_g3),
        "g1_g2": build_scatter_with_trend("G1", "G2", f"G1 vs G2 (r = {r_g1_g2:.2f})", r_g1_g2),
        "r_g1_g3": r_g1_g3,
        "r_g2_g3": r_g2_g3,
        "r_g1_g2": r_g1_g2
    }


def render_studytime_full_analytics(df):
    """
    Renders 2 complementary charts for study time analysis.
    """
    st_names = {1: "< 2 hrs (Tier 1)", 2: "2–5 hrs (Tier 2)", 3: "5–10 hrs (Tier 3)", 4: "> 10 hrs (Tier 4)"}
    st_stats = df.groupby("studytime")["G3"].agg(["mean", "count"]).reset_index()
    st_stats["Tier"] = st_stats["studytime"].map(st_names)

    fig_mean = go.Figure()
    fig_mean.add_trace(go.Bar(
        x=st_stats["Tier"],
        y=st_stats["mean"].round(2),
        text=[f"<b>{m:.2f}</b> (N={c})" for m, c in zip(st_stats["mean"], st_stats["count"])],
        textposition="outside",
        marker=dict(color="#059669", line=dict(color="#0F172A", width=0.8)),
        hovertemplate="Study Tier: <b>%{x}</b><br>Cohort Mean G3: <b>%{y}</b><extra></extra>"
    ))
    apply_plotly_theme(fig_mean, title="Mean Final Grade by Weekly Study Tier", height=280)
    fig_mean.update_layout(yaxis=dict(title="Mean G3", range=[0, 20]))

    st_ct = pd.crosstab(df["studytime"], df["target_class"], normalize="index") * 100
    st_ct.index = [st_names.get(i, str(i)) for i in st_ct.index]
    fig_stack = go.Figure()
    names = ["Poor", "Average", "Good", "Excellent"]
    for col_idx in range(4):
        if col_idx in st_ct.columns:
            fig_stack.add_trace(go.Bar(
                name=names[col_idx],
                x=st_ct.index,
                y=st_ct[col_idx],
                marker_color=CLASS_HEX_LIST[col_idx]
            ))
    apply_plotly_theme(fig_stack, title="Performance Category Proportions by Study Tier (%)", height=280, show_legend=True)
    fig_stack.update_layout(barmode="stack", yaxis=dict(title="Share (%)", range=[0, 100]))

    return {"mean_chart": fig_mean, "stack_chart": fig_stack}


def render_failures_full_analytics(df):
    """
    Renders 2 complementary charts for historical failures analysis.
    """
    fail_stats = df.groupby("failures")["G3"].agg(["mean", "count"]).reset_index()
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=[f"{f} Failure(s)" for f in fail_stats["failures"]],
        y=fail_stats["mean"].round(2),
        text=[f"<b>{m:.2f}</b> (N={c})" for m, c in zip(fail_stats["mean"], fail_stats["count"])],
        textposition="outside",
        marker=dict(color=["#059669", "#D97706", "#E11D48", "#9F1239"][:len(fail_stats)], line=dict(color="#0F172A", width=0.8)),
        hovertemplate="Setbacks: <b>%{x}</b><br>Mean Grade G3: <b>%{y}</b><extra></extra>"
    ))
    apply_plotly_theme(fig_bar, title="Mean Final Grade by Prior Class Failures", height=280)
    fig_bar.update_layout(yaxis=dict(title="Mean G3", range=[0, 20]))

    fig_box = px.box(
        df, x="failures", y="G3", color="failures",
        labels={"failures": "Prior Failures", "G3": "Final Grade G3"},
        color_discrete_sequence=["#059669", "#D97706", "#E11D48", "#9F1239"]
    )
    apply_plotly_theme(fig_box, title="Grade Dispersion Across Failure Counts", height=280)
    fig_box.update_layout(yaxis=dict(range=[-0.5, 20.5]))

    return {"bar_chart": fig_bar, "box_chart": fig_box}


def render_attendance_full_analytics(df):
    """
    Renders 2 complementary charts for attendance analysis.
    """
    df_temp = df.copy()
    bins = [-1, 0, 4, 10, 15, 40]
    labels = ["0 Days", "1–4 Days", "5–10 Days", "11–15 Days", "> 15 Days"]
    df_temp["abs_bracket"] = pd.cut(df_temp["absences"], bins=bins, labels=labels)

    abs_stats = df_temp.groupby("abs_bracket", observed=False)["G3"].agg(["mean", "count"]).reset_index()
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=abs_stats["abs_bracket"],
        y=abs_stats["mean"].round(2),
        text=[f"<b>{m:.2f}</b> (N={c})" for m, c in zip(abs_stats["mean"], abs_stats["count"])],
        textposition="outside",
        marker=dict(color="#2563EB", line=dict(color="#0F172A", width=0.8)),
        hovertemplate="Absence Bracket: <b>%{x}</b><br>Mean Grade G3: <b>%{y}</b><extra></extra>"
    ))
    apply_plotly_theme(fig_bar, title="Mean Final Grade by Absence Bracket", height=280)
    fig_bar.update_layout(yaxis=dict(title="Mean G3", range=[0, 20]))

    fig_scatter = px.scatter(
        df, x="absences", y="G3", color="performance_category",
        category_orders={"performance_category": ["Poor", "Average", "Good", "Excellent"]},
        color_discrete_map={"Poor": "#E11D48", "Average": "#D97706", "Good": "#2563EB", "Excellent": "#059669"},
        labels={"absences": "Recorded Absences", "G3": "Final Grade G3"}
    )
    apply_plotly_theme(fig_scatter, title="Final Grade G3 vs. Total Absences", height=280)

    return {"bar_chart": fig_bar, "scatter_chart": fig_scatter}


def render_demographics_full_analytics(df):
    """
    Renders 4 socio-demographic cohort comparison charts.
    """
    charts = {}

    def make_mean_bar(cat_col, title_txt, label_map=None, colors=None):
        grouped = df.groupby(cat_col)["G3"].agg(["mean", "count"]).reset_index()
        x_vals = [label_map.get(v, v) if label_map else v for v in grouped[cat_col]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_vals,
            y=grouped["mean"].round(2),
            text=[f"<b>{m:.2f}</b> (N={c})" for m, c in zip(grouped["mean"], grouped["count"])],
            textposition="outside",
            marker=dict(color=colors or ["#4F46E5", "#0EA5E9"], line=dict(color="#0F172A", width=0.8)),
            hovertemplate="Cohort: <b>%{x}</b><br>Mean Grade: <b>%{y}</b><extra></extra>"
        ))
        apply_plotly_theme(fig, title=title_txt, height=260)
        fig.update_layout(yaxis=dict(title="Mean G3", range=[0, 20]), margin=dict(l=30, r=20, t=40, b=30))
        return fig

    charts["school"] = make_mean_bar("school", "By Secondary Institution", {"GP": "Gabriel Pereira", "MS": "Mousinho"})
    charts["gender"] = make_mean_bar("sex", "By Gender", {"F": "Female (F)", "M": "Male (M)"}, ["#EC4899", "#3B82F6"])
    charts["address"] = make_mean_bar("address", "By Residential Location", {"U": "Urban (U)", "R": "Rural (R)"}, ["#059669", "#D97706"])
    charts["internet"] = make_mean_bar("internet", "By Home Internet Access", {"yes": "Internet Yes", "no": "Internet No"}, ["#2563EB", "#64748B"])

    return charts


def render_correlation_heatmap(df):
    """
    Renders an interactive correlation heatmap across 15 numerical features.
    """
    num_cols = [
        "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        "absences", "G1", "G2", "G3"
    ]
    avail_cols = [c for c in num_cols if c in df.columns]
    corr_matrix = df[avail_cols].corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate="%{text}",
        textfont={"size": 10, "family": "Inter, sans-serif"},
        colorbar=dict(title="Pearson r", len=0.85, thickness=12),
        hovertemplate="X: <b>%{x}</b><br>Y: <b>%{y}</b><br>Pearson r: <b>%{z:.2f}</b><extra></extra>"
    ))

    apply_plotly_theme(fig, title="15-Variable Pearson Correlation Heatmap", height=500)
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        margin=dict(l=40, r=30, t=50, b=60)
    )
    return fig


def render_calculated_insights(df):
    """
    Dynamically generates non-judgmental, empirical insight cards calculated directly
    from the current dataset slice.
    """
    total = len(df)
    if total == 0:
        return

    r_g2_g3 = float(df["G2"].corr(df["G3"]))
    st_low = df[df["studytime"] == 1]
    st_high = df[df["studytime"] >= 3]
    avg_st_low = float(st_low["G3"].mean()) if len(st_low) > 0 else 0
    avg_st_high = float(st_high["G3"].mean()) if len(st_high) > 0 else 0
    diff_st = avg_st_high - avg_st_low

    fail_zero = df[df["failures"] == 0]
    fail_has = df[df["failures"] >= 1]
    avg_fail_zero = float(fail_zero["G3"].mean()) if len(fail_zero) > 0 else 0
    avg_fail_has = float(fail_has["G3"].mean()) if len(fail_has) > 0 else 0
    diff_fail = avg_fail_zero - avg_fail_has

    net_yes = df[df["internet"] == "yes"]
    net_no = df[df["internet"] == "no"]
    avg_net_yes = float(net_yes["G3"].mean()) if len(net_yes) > 0 else 0
    avg_net_no = float(net_no["G3"].mean()) if len(net_no) > 0 else 0

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #4F46E5; border-radius: 10px; padding: 16px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #0F172A; font-size: 0.95rem;">
                    1. Interim-to-Final Grade Trajectory Alignment
                </div>
                <div style="color: #475569; font-size: 0.86rem; margin-top: 6px; line-height: 1.5;">
                    Period 2 evaluation (<strong>G2</strong>) demonstrates a near-deterministic Pearson correlation of <strong>r = {r_g2_g3:.2f}</strong> with final outcomes (G3). Stabilizing study habits before mid-term evaluations creates lasting momentum.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #059669; border-radius: 10px; padding: 16px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #0F172A; font-size: 0.95rem;">
                    2. Study Time Bracket Trajectory Margin
                </div>
                <div style="color: #475569; font-size: 0.86rem; margin-top: 6px; line-height: 1.5;">
                    Students committing <strong>&gt; 5 hours weekly</strong> show an observed cohort mean of <strong>{avg_st_high:.2f} G3 points</strong>, compared to <strong>{avg_st_low:.2f} points</strong> for those studying &lt; 2 hours (an observed margin of <strong>+{diff_st:.2f} points</strong>).
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #E11D48; border-radius: 10px; padding: 16px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #0F172A; font-size: 0.95rem;">
                    3. Prior Academic Setback Penalty
                </div>
                <div style="color: #475569; font-size: 0.86rem; margin-top: 6px; line-height: 1.5;">
                    Students with zero prior failures average <strong>{avg_fail_zero:.2f} points</strong> in G3, versus <strong>{avg_fail_has:.2f} points</strong> for students with &ge; 1 previous failures (an observed disparity of <strong>-{diff_fail:.2f} points</strong>).
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #D97706; border-radius: 10px; padding: 16px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #0F172A; font-size: 0.95rem;">
                    4. Household Digital Infrastructure Impact
                </div>
                <div style="color: #475569; font-size: 0.86rem; margin-top: 6px; line-height: 1.5;">
                    Students reporting home internet access average <strong>{avg_net_yes:.2f} points</strong> versus <strong>{avg_net_no:.2f} points</strong> for those without. Digital connectivity acts as a key enabler for academic research and assignment completion.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =============================================================================
# 4. MODEL LABORATORY COMPONENTS (pages/4_Model_Laboratory.py)
# =============================================================================

def render_model_lab_overview_cards(rf_m, dnn_m, gb_m):
    """
    Renders 3 detailed model cards comparing Random Forest, DNN, and Gradient Boosting.
    """
    col1, col2, col3 = st.columns(3)

    models_info = [
        (col1, "Random Forest", "Baseline Bagging Ensemble", rf_m, "Phase 2 Baseline", "#64748B", False),
        (col2, "Deep Neural Net", "Colab Keras 3-Layer Dense", dnn_m, "Phase 3 Deep Learning", "#6366F1", False),
        (col3, "Gradient Boosting", "Production Pipeline", gb_m, "Selected Production Model", "#059669", True),
    ]

    for col, title, arch, m, badge, badge_color, is_selected in models_info:
        acc = m.get("accuracy", 0.0) * 100
        f1 = m.get("macro_f1", 0.0)
        rec = m.get("macro_recall", 0.0) * 100
        prec = m.get("macro_precision", 0.0) * 100

        top_border = f"3px solid {badge_color}" if is_selected else "1px solid #E2E8F0"
        card_bg = "#FFFFFF"

        with col:
            st.markdown(
                f"""
                <div style="background-color: {card_bg}; border: 1px solid #E2E8F0; border-top: {top_border}; border-radius: 14px; padding: 20px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04); height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.15rem; font-weight: 800; color: #0F172A;">{title}</span>
                            <div style="color: #64748B; font-size: 0.76rem; margin-top: 2px;">{arch}</div>
                        </div>
                        <span style="background-color: #F8FAFC; border: 1px solid #E2E8F0; color: {badge_color}; font-size: 0.7rem; font-weight: 700; padding: 3px 8px; border-radius: 9999px;">
                            {badge}
                        </span>
                    </div>
                    <hr style="margin: 12px 0; border: none; border-top: 1px solid #F1F5F9;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div>
                            <div style="font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600;">Test Accuracy</div>
                            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800; color: #0F172A; margin-top: 2px;">{acc:.2f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600;">Macro F1-Score</div>
                            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800; color: {badge_color}; margin-top: 2px;">{f1:.4f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600;">Macro Recall</div>
                            <div style="font-size: 0.95rem; font-weight: 700; color: #334155; margin-top: 2px;">{rec:.1f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600;">Macro Precision</div>
                            <div style="font-size: 0.95rem; font-weight: 700; color: #334155; margin-top: 2px;">{prec:.1f}%</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_model_comparison_bar_chart(rf_m, dnn_m, gb_m):
    """
    Renders grouped bar chart comparing RF, DNN, and Gradient Boosting across key metrics.
    """
    metrics_list = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]

    rf_scores = [
        rf_m.get("accuracy", 0.6633) * 100,
        rf_m.get("macro_precision", 0.6601) * 100,
        rf_m.get("macro_recall", 0.7262) * 100,
        rf_m.get("macro_f1", 0.6779) * 100,
        rf_m.get("weighted_f1", 0.6603) * 100,
    ]
    dnn_scores = [
        dnn_m.get("accuracy", 0.6837) * 100,
        dnn_m.get("macro_precision", 0.6638) * 100,
        dnn_m.get("macro_recall", 0.6989) * 100,
        dnn_m.get("macro_f1", 0.6747) * 100,
        dnn_m.get("weighted_f1", 0.6867) * 100,
    ]
    gb_scores = [
        gb_m.get("accuracy", 0.7245) * 100,
        gb_m.get("macro_precision", 0.7513) * 100,
        gb_m.get("macro_recall", 0.7753) * 100,
        gb_m.get("macro_f1", 0.7516) * 100,
        gb_m.get("weighted_f1", 0.7206) * 100,
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Random Forest", x=metrics_list, y=rf_scores,
        marker_color="#94A3B8", text=[f"{v:.1f}%" for v in rf_scores], textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="Deep Neural Net", x=metrics_list, y=dnn_scores,
        marker_color="#6366F1", text=[f"{v:.1f}%" for v in dnn_scores], textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="Gradient Boosting (Production)", x=metrics_list, y=gb_scores,
        marker_color="#059669", text=[f"<b>{v:.1f}%</b>" for v in gb_scores], textposition="outside"
    ))

    apply_plotly_theme(
        fig,
        title="Principal Model Architecture Benchmark (Outer Test Set, N = 98)",
        height=340,
        show_legend=True
    )
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score (%)", range=[50, 88])
    )
    return fig


def render_confusion_matrix_heatmap(y_true, y_pred, title="Confusion Matrix", model_name="Model"):
    """
    Renders an annotated Plotly heatmap confusion matrix with class names:
    0: Poor, 1: Average, 2: Good, 3: Excellent.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    labels = ["Poor (0)", "Average (1)", "Good (2)", "Excellent (3)"]
    z_text = [[str(val) for val in row] for row in cm]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=z_text,
        texttemplate="<b>%{text}</b>",
        textfont={"size": 13, "family": "Plus Jakarta Sans, sans-serif"},
        colorscale="Blues",
        showscale=False,
        hovertemplate="Actual Class: <b>%{y}</b><br>Predicted Class: <b>%{x}</b><br>Cohort Count: <b>%{z}</b><extra></extra>"
    ))

    apply_plotly_theme(fig, title=title, height=280)
    fig.update_layout(
        xaxis=dict(title="Predicted Classification", tickfont=dict(size=10.5)),
        yaxis=dict(title="Actual Ground Truth", autorange="reversed", tickfont=dict(size=10.5)),
        margin=dict(l=35, r=20, t=40, b=30)
    )
    return fig


def render_feature_importance_charts(fi_df, perm_df=None):
    """
    Renders two feature importance charts:
    1. Built-in Gradient Boosting Gini Importance (top 12)
    2. Model-Agnostic Permutation Importance on Development Set (top 12)
    """
    charts = {}

    top_fi = fi_df.head(12).iloc[::-1]
    fig_gini = go.Figure(go.Bar(
        x=top_fi["importance"],
        y=top_fi["feature"],
        orientation="h",
        text=[f"<b>{v:.3f}</b>" for v in top_fi["importance"]],
        textposition="outside",
        marker=dict(color="#059669", line=dict(color="#0F172A", width=0.8))
    ))
    apply_plotly_theme(fig_gini, title="Built-in Gradient Boosting Feature Importance (Top 12 Gini Splits)", height=340)
    fig_gini.update_layout(
        xaxis=dict(title="Gini Impurity Reduction Share", range=[0, max(top_fi["importance"]) * 1.25]),
        yaxis=dict(title=None)
    )
    charts["gini"] = fig_gini

    if perm_df is not None:
        top_perm = perm_df.head(12).iloc[::-1]
        fig_perm = go.Figure(go.Bar(
            x=top_perm["perm_importance_mean"],
            y=top_perm["feature"],
            orientation="h",
            text=[f"<b>{v:.3f}</b>" for v in top_perm["perm_importance_mean"]],
            textposition="outside",
            marker=dict(color="#4F46E5", line=dict(color="#0F172A", width=0.8))
        ))
        apply_plotly_theme(fig_perm, title="Permutation Importance (Top 12 on Development Set)", height=340)
        fig_perm.update_layout(
            xaxis=dict(title="Macro F1 Score Drop When Shuffled", range=[0, max(top_perm["perm_importance_mean"]) * 1.25]),
            yaxis=dict(title=None)
        )
        charts["perm"] = fig_perm

    return charts


def render_cv_comparison_chart(cv_df):
    """
    Renders ranked horizontal bar chart of 5-fold cross-validation Macro F1 across candidate algorithms.
    """
    df_sorted = cv_df.sort_values("cv_macro_f1_mean", ascending=True)
    colors = ["#059669" if "Baseline Gradient Boosting" in m else "#94A3B8" for m in df_sorted["Model"]]

    fig = go.Figure(go.Bar(
        x=df_sorted["cv_macro_f1_mean"],
        y=df_sorted["Model"],
        orientation="h",
        text=[f"<b>{v:.4f}</b> (±{s:.3f})" for v, s in zip(df_sorted["cv_macro_f1_mean"], df_sorted["cv_macro_f1_std"])],
        textposition="outside",
        marker=dict(color=colors, line=dict(color="#0F172A", width=0.8))
    ))

    apply_plotly_theme(fig, title="5-Fold Cross-Validation Performance Ranking (Dev Set N = 551)", height=380)
    fig.update_layout(
        xaxis=dict(title="Mean CV Macro F1-Score", range=[0.60, 0.88]),
        yaxis=dict(title=None)
    )
    return fig


def render_oversampling_comparison_chart(os_df):
    """
    Renders comparative chart for the 9-strategy oversampling experiment.
    """
    df_sorted = os_df.sort_values("CV_Macro_F1", ascending=True)
    colors = ["#059669" if "Baseline GB" in s else "#64748B" for s in df_sorted["Strategy"]]

    fig = go.Figure(go.Bar(
        x=df_sorted["CV_Macro_F1"],
        y=df_sorted["Strategy"],
        orientation="h",
        text=[f"<b>{v:.4f}</b>" for v in df_sorted["CV_Macro_F1"]],
        textposition="outside",
        marker=dict(color=colors, line=dict(color="#0F172A", width=0.8))
    ))

    apply_plotly_theme(fig, title="Oversampling Strategies vs. Baseline (5-Fold CV Macro F1)", height=350)
    fig.update_layout(
        xaxis=dict(title="CV Macro F1", range=[0.74, 0.86]),
        yaxis=dict(title=None)
    )
    return fig


def render_dnn_history_charts(th):
    """
    Renders two Plotly charts showing DNN training curves over epochs:
    1. Training Accuracy vs. Validation Accuracy
    2. Training Loss vs. Validation Loss
    """
    epochs = list(range(1, len(th["loss"]) + 1))
    best_epoch = int(np.argmin(th["val_loss"])) + 1
    min_val_loss = float(np.min(th["val_loss"]))

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=epochs, y=th["accuracy"], mode="lines", name="Train Accuracy",
        line=dict(color="#4F46E5", width=2)
    ))
    fig_acc.add_trace(go.Scatter(
        x=epochs, y=th["val_accuracy"], mode="lines", name="Val Accuracy",
        line=dict(color="#059669", width=2, dash="dash")
    ))
    apply_plotly_theme(fig_acc, title=f"Deep Neural Network Accuracy Curves ({len(epochs)} Epochs)", height=280, show_legend=True)
    fig_acc.update_layout(xaxis=dict(title="Epoch"), yaxis=dict(title="Accuracy", range=[0.4, 0.9]))

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=th["loss"], mode="lines", name="Train Loss",
        line=dict(color="#4F46E5", width=2)
    ))
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=th["val_loss"], mode="lines", name="Val Loss",
        line=dict(color="#E11D48", width=2, dash="dash")
    ))
    fig_loss.add_vline(
        x=best_epoch, line_width=1.5, line_dash="dot", line_color="#D97706",
        annotation_text=f"Min Val Loss ({min_val_loss:.4f})", annotation_position="top right",
        annotation_font=dict(size=10, family="Inter, sans-serif")
    )
    apply_plotly_theme(fig_loss, title=f"DNN Convergence Loss Curves (Min at Epoch {best_epoch})", height=280, show_legend=True)
    fig_loss.update_layout(xaxis=dict(title="Epoch"), yaxis=dict(title="Loss", range=[0.2, 1.2]))

    return {
        "acc_chart": fig_acc,
        "loss_chart": fig_loss,
        "best_epoch": best_epoch,
        "min_val_loss": min_val_loss
    }
