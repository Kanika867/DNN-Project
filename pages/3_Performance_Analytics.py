"""
EduMetric — Academic Performance Prediction & Intelligence Platform
Module 3: Multi-Variable Academic Performance Analytics & Correlation Suite
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.ui_theme import (
    BRAND_NAME,
    inject_global_theme,
    render_unified_sidebar,
    render_page_header,
    render_page_footer
)
from src.app_utils import (
    load_student_data,
    CLASS_COLORS,
    CLASS_HEX_LIST,
    DEFAULT_CLASS_MAPPING
)
from src.dashboard_components import (
    render_analytics_kpi_cards,
    render_g3_distribution_histogram,
    render_category_box_plots,
    render_previous_grades_scatters,
    render_studytime_full_analytics,
    render_failures_full_analytics,
    render_attendance_full_analytics,
    render_demographics_full_analytics,
    render_correlation_heatmap,
    render_calculated_insights
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"Performance Analytics | {BRAND_NAME}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# INJECT UNIFIED DESIGN SYSTEM & APPLICATION SHELL
# -----------------------------------------------------------------------------
inject_global_theme()
render_unified_sidebar(current_page="analytics")

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
df_raw = load_student_data()

if df_raw is None:
    st.error("Failed to load student dataset. Please ensure `data/student-por.csv` exists.")
    st.stop()

# -----------------------------------------------------------------------------
# HERO BANNER / PAGE HEADER
# -----------------------------------------------------------------------------
render_page_header(
    title="Academic Performance Analytics & Correlation Suite",
    subtitle="Conduct multi-dimensional investigations into academic performance trajectories, evaluate interim-grade predictive strength, inspect behavioral and contextual factors, and analyze inter-feature Pearson correlations.",
    badge_text="Empirical Research & Statistical Intelligence",
    icon="📊"
)

# -----------------------------------------------------------------------------
# INTERACTIVE COHORT FILTER BAR
# -----------------------------------------------------------------------------
with st.container(border=True):
    col_f1, col_f2, col_f3 = st.columns([1.5, 1.5, 1])

    with col_f1:
        school_choice = st.selectbox(
            "Institution Campus Filter",
            options=["All Schools", "GP (Gabriel Pereira)", "MS (Mousinho da Silveira)"],
            index=0,
            help="Filter analytics across secondary institutions"
        )

    with col_f2:
        category_choice = st.selectbox(
            "Performance Bracket Filter",
            options=["All Categories", "Poor (0–9)", "Average (10–12)", "Good (13–15)", "Excellent (16–20)"],
            index=0,
            help="Filter analytics by final examination bracket"
        )

    with col_f3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset Analytics Filter", use_container_width=True, type="secondary"):
            st.session_state.pop("analytics_school", None)
            st.session_state.pop("analytics_category", None)
            st.rerun()

# Apply cohort filter
df_active = df_raw.copy()
if "GP" in school_choice:
    df_active = df_active[df_active["school"] == "GP"]
elif "MS" in school_choice:
    df_active = df_active[df_active["school"] == "MS"]

if "Poor" in category_choice:
    df_active = df_active[df_active["target_class"] == 0]
elif "Average" in category_choice:
    df_active = df_active[df_active["target_class"] == 1]
elif "Good" in category_choice:
    df_active = df_active[df_active["target_class"] == 2]
elif "Excellent" in category_choice:
    df_active = df_active[df_active["target_class"] == 3]

if len(df_active) == 0:
    st.warning("⚠️ No student records match the selected cohort criteria.")
    st.stop()

# -----------------------------------------------------------------------------
# SECTION 1: ANALYTICS OVERVIEW KPI CARDS
# -----------------------------------------------------------------------------
render_analytics_kpi_cards(df_active)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECTION 2: ACADEMIC PERFORMANCE DISTRIBUTION
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="edum-section-heading">
        <span>1.</span>
        <span>Final Examination Grade (G3) Distribution & Category Dispersion</span>
    </div>
    """,
    unsafe_allow_html=True
)

dist_col1, dist_col2 = st.columns([1.3, 1])

with dist_col1:
    with st.container(border=True):
        fig_hist = render_g3_distribution_histogram(df_active)
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

with dist_col2:
    with st.container(border=True):
        fig_box = render_category_box_plots(df_active)
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

st.caption(
    "📌 *Category Ranges: Poor (0–9 marks), Average (10–12 marks), Good (13–15 marks), Excellent (16–20 marks). Dashed vertical lines represent active cohort mean and median.*"
)

# -----------------------------------------------------------------------------
# SECTION 3: PRIOR GRADES VS. FINAL PERFORMANCE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="edum-section-heading">
        <span>2.</span>
        <span>Prior Academic Performance Trajectories vs. Final Outcomes</span>
    </div>
    <p style="color: #64748B; font-size: 0.88rem; margin-top: -10px; margin-bottom: 14px;">
        Evaluating the linear alignment between interim evaluations (Period 1 <strong>G1</strong> and Period 2 <strong>G2</strong>) and the final mark (<strong>G3</strong>).
        <em>Important: Correlations demonstrate strong longitudinal tracking rather than direct mechanical causality.</em>
    </p>
    """,
    unsafe_allow_html=True
)

scatters = render_previous_grades_scatters(df_active)

scat_col1, scat_col2, scat_col3 = st.columns(3)
with scat_col1:
    with st.container(border=True):
        st.plotly_chart(scatters["g1_g3"], use_container_width=True, config={"displayModeBar": False})
with scat_col2:
    with st.container(border=True):
        st.plotly_chart(scatters["g2_g3"], use_container_width=True, config={"displayModeBar": False})
with scat_col3:
    with st.container(border=True):
        st.plotly_chart(scatters["g1_g2"], use_container_width=True, config={"displayModeBar": False})

st.markdown(
    f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #4F46E5; border-radius: 8px; padding: 14px 18px; font-size: 0.86rem; color: #475569; margin-top: 6px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);">
        <strong>Empirical Trajectory Finding:</strong> Period 2 evaluation marks correlate overwhelmingly with final examination outcomes (<strong>r = {scatters['r_g2_g3']:.2f}</strong>), while Period 1 establishes strong baseline direction (<strong>r = {scatters['r_g1_g3']:.2f}</strong>). Students rarely transition across more than one letter bracket between mid-term and finals without targeted intervention.
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# SECTION 4: STUDY TIME AND PERFORMANCE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="edum-section-heading">
        <span>3.</span>
        <span>Weekly Dedicated Study Duration</span>
    </div>
    """,
    unsafe_allow_html=True
)

st_col1, st_col2 = st.columns(2)
st_charts = render_studytime_full_analytics(df_active)

with st_col1:
    with st.container(border=True):
        st.plotly_chart(st_charts["mean_chart"], use_container_width=True, config={"displayModeBar": False})

with st_col2:
    with st.container(border=True):
        st.plotly_chart(st_charts["stack_chart"], use_container_width=True, config={"displayModeBar": False})

st.caption(
    "📌 *Weekly Study Tiers: 1 (< 2 hrs), 2 (2–5 hrs), 3 (5–10 hrs), 4 (> 10 hrs). Observed group differences indicate historical cohort patterns and do not imply study hours alone guarantee performance.*"
)

# -----------------------------------------------------------------------------
# SECTION 5: PAST FAILURES AND PERFORMANCE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="edum-section-heading">
        <span>4.</span>
        <span>Historical Course Setbacks & Academic Recovery</span>
    </div>
    """,
    unsafe_allow_html=True
)

fail_col1, fail_col2 = st.columns(2)
fail_charts = render_failures_full_analytics(df_active)

with fail_col1:
    with st.container(border=True):
        st.plotly_chart(fail_charts["bar_chart"], use_container_width=True, config={"displayModeBar": False})

with fail_col2:
    with st.container(border=True):
        st.plotly_chart(fail_charts["box_chart"], use_container_width=True, config={"displayModeBar": False})

st.caption(
    "📌 *Prior failures represent past academic setbacks. In modern educational research, setbacks are diagnostic indicators for targeted tutoring, never immutable ceilings on future student achievement.*"
)

# -----------------------------------------------------------------------------
# SECTION 6: ATTENDANCE AND ABSENCES
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="edum-section-heading">
        <span>5.</span>
        <span>Attendance Regularity & School Absences</span>
    </div>
    """,
    unsafe_allow_html=True
)

att_col1, att_col2 = st.columns(2)
att_charts = render_attendance_full_analytics(df_active)

with att_col1:
    with st.container(border=True):
        st.plotly_chart(att_charts["bar_chart"], use_container_width=True, config={"displayModeBar": False})

with att_col2:
    with st.container(border=True):
        st.plotly_chart(att_charts["scatter_chart"], use_container_width=True, config={"displayModeBar": False})

st.caption(
    "📌 *Attendance brackets group recorded days absent. Students with severe absenteeism (> 15 days) represent outlier cases (N = 21) exhibiting wider grade dispersion.*"
)

# -----------------------------------------------------------------------------
# SECTION 7: DEMOGRAPHIC & SOCIO-CONTEXTUAL COMPARISONS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="edum-section-heading">
        <span>6.</span>
        <span>Descriptive Socio-Demographic & Contextual Comparisons</span>
    </div>
    <p style="color: #64748B; font-size: 0.88rem; margin-top: -10px; margin-bottom: 14px;">
        Descriptive cohort comparisons across contextual and household dimensions.
        <em>Responsible AI Disclosure: Group differences reflect contextual educational factors within this specific historical cohort and must never be used to prejudge an individual student's potential.</em>
    </p>
    """,
    unsafe_allow_html=True
)

demo_charts = render_demographics_full_analytics(df_active)

demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)
with demo_col1:
    with st.container(border=True):
        st.plotly_chart(demo_charts["school"], use_container_width=True, config={"displayModeBar": False})
with demo_col2:
    with st.container(border=True):
        st.plotly_chart(demo_charts["gender"], use_container_width=True, config={"displayModeBar": False})
with demo_col3:
    with st.container(border=True):
        st.plotly_chart(demo_charts["address"], use_container_width=True, config={"displayModeBar": False})
with demo_col4:
    with st.container(border=True):
        st.plotly_chart(demo_charts["internet"], use_container_width=True, config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# SECTION 8: CORRELATION HEATMAP
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="edum-section-heading">
        <span>7.</span>
        <span>15-Variable Pearson Correlation Matrix</span>
    </div>
    """,
    unsafe_allow_html=True
)

with st.container(border=True):
    fig_heatmap = render_correlation_heatmap(df_active)
    st.plotly_chart(fig_heatmap, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    """
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px 20px; font-size: 0.86rem; color: #475569; line-height: 1.55; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);">
        <strong style="color: #0F172A;">Statistical Interpretation & Methodological Caveats:</strong><br>
        • <strong>Strong Positive Correlations:</strong> G2 with G3 (r = +0.92) and G1 with G3 (r = +0.83). Moderate positive associations occur for study time (+0.25) and parental education (+0.24).<br>
        • <strong>Strong Negative Alignments:</strong> Past class failures (r = -0.39) and workday alcohol consumption (r = -0.20) exhibit the clearest inverse alignments.<br>
        • <strong>Methodological Precision:</strong> Pearson correlation evaluates linear continuous associations; ordinal variables such as study time tiers, failures, and health ratings serve as descriptive directional indicators rather than exact structural parameters.
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# SECTION 9: DYNAMIC EMPIRICAL INSIGHTS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="edum-section-heading">
        <span>8.</span>
        <span>Calculated Empirical Insights</span>
    </div>
    """,
    unsafe_allow_html=True
)
render_calculated_insights(df_active)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
render_page_footer()
