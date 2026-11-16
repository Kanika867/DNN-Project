"""
EduMetric — Academic Performance Prediction & Intelligence Platform
Module 0: Executive Overview Dashboard
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
    BRAND_DESCRIPTOR,
    inject_global_theme,
    render_unified_sidebar,
    render_page_header,
    render_page_footer
)
from src.app_utils import (
    load_student_data,
    load_improved_model_metrics,
)
from src.dashboard_components import (
    render_kpi_cards,
    render_target_distribution_chart,
    render_eda_summary_cards,
    render_model_snapshot,
    render_disclaimer_box
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{BRAND_NAME} | Academic Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# INJECT UNIFIED DESIGN SYSTEM & APPLICATION SHELL
# -----------------------------------------------------------------------------
inject_global_theme()
render_unified_sidebar(current_page="overview")

# -----------------------------------------------------------------------------
# DATA & ARTIFACT LOADING
# -----------------------------------------------------------------------------
df = load_student_data()
model_metrics = load_improved_model_metrics()

if df is None:
    st.error("Failed to load dataset. Please ensure `data/student-por.csv` exists.")
    st.stop()

# -----------------------------------------------------------------------------
# HERO BANNER / PAGE HEADER
# -----------------------------------------------------------------------------
render_page_header(
    title=f"{BRAND_NAME} — Academic Performance Prediction & Intelligence Platform",
    subtitle="An enterprise academic intelligence and early warning platform. Leveraging calibrated machine learning to help educators, academic advisors, and institutions forecast student outcomes, detect academic risk early, and deliver targeted interventions.",
    badge_text="Institutional Intelligence Suite · Production Architecture",
    icon="📊"
)

# -----------------------------------------------------------------------------
# SECTION 1: KEY PERFORMANCE INDICATORS
# -----------------------------------------------------------------------------
test_acc_str = "72.45%"
macro_f1_str = "0.7516"

if model_metrics and "test_metrics_outer" in model_metrics:
    outer_m = model_metrics["test_metrics_outer"]
    test_acc_str = f"{outer_m.get('accuracy', 0.7245) * 100:.2f}%"
    macro_f1_str = f"{outer_m.get('macro_f1', 0.7516):.4f}"

render_kpi_cards(
    total_records=len(df),
    feature_count=32,
    class_count=4,
    model_name="Gradient Boosting",
    test_acc=test_acc_str,
    test_macro_f1=macro_f1_str
)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECTION 2: DATASET PROFILE & TARGET DISTRIBUTION
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="edum-section-heading">
        <span>📋</span>
        <span>Cohort Profile & Outcome Distribution</span>
    </div>
    """,
    unsafe_allow_html=True
)

col_overview, col_chart = st.columns([1, 1.4])

with col_overview:
    st.markdown(
        """
        <div class="edum-card" style="height: 100%;">
            <div class="edum-card-header">
                <div>
                    <h3 class="edum-card-title">UCI Portuguese Secondary Cohort</h3>
                    <div class="edum-card-subtitle">Secondary language course academic records</div>
                </div>
                <span style="background-color: #EEF2FF; color: #4F46E5; border: 1px solid #C7D2FE; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">
                    Benchmark Dataset
                </span>
            </div>
            <div style="font-size: 0.86rem; color: #475569; line-height: 1.55;">
                Calibrated on real-world academic data tracking secondary school students across two regional institutions in Portugal: <strong>Escola Secundária Gabriel Pereira (GP)</strong> and <strong>Escola Secundária Mousinho da Silveira (MS)</strong>.
            </div>
            <hr style="margin: 14px 0; border: none; border-top: 1px solid #F1F5F9;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.83rem;">
                <div><span style="color: #64748B;">Total Records:</span> <strong>649 Students</strong></div>
                <div><span style="color: #64748B;">Total Inputs:</span> <strong>32 Features</strong></div>
                <div><span style="color: #64748B;">Missing Cells:</span> <strong style="color: #059669;">0 (100% Clean)</strong></div>
                <div><span style="color: #64748B;">Duplicate Rows:</span> <strong style="color: #059669;">0</strong></div>
                <div><span style="color: #64748B;">Numerical Fields:</span> <strong>15 Metrics</strong></div>
                <div><span style="color: #64748B;">Categorical Fields:</span> <strong>17 Attributes</strong></div>
                <div><span style="color: #64748B;">Target Metric:</span> <strong>G3 (0–20 Grade)</strong></div>
                <div><span style="color: #64748B;">Target Schema:</span> <strong>4 Discrete Classes</strong></div>
            </div>
            <div style="margin-top: 16px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; font-size: 0.78rem; color: #475569; line-height: 1.6;">
                <div style="font-weight: 700; color: #0F172A; margin-bottom: 4px;">4-Tier Classification Schema:</div>
                <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                    <span style="background-color: #FFF1F2; border: 1px solid #FECDD3; color: #9F1239; padding: 2px 8px; border-radius: 4px; font-weight: 600;">
                        • Poor (0): 0–9 Marks (15.4%)
                    </span>
                    <span style="background-color: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; padding: 2px 8px; border-radius: 4px; font-weight: 600;">
                        • Average (1): 10–12 Marks (42.1%)
                    </span>
                    <span style="background-color: #EFF6FF; border: 1px solid #BFDBFE; color: #1E40AF; padding: 2px 8px; border-radius: 4px; font-weight: 600;">
                        • Good (2): 13–15 Marks (29.9%)
                    </span>
                    <span style="background-color: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; padding: 2px 8px; border-radius: 4px; font-weight: 600;">
                        • Excellent (3): 16–20 Marks (12.6%)
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_chart:
    with st.container(border=True):
        fig_dist = render_target_distribution_chart(df)
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# SECTION 3: KEY EXPLORATORY FINDINGS
# -----------------------------------------------------------------------------
render_eda_summary_cards()

# -----------------------------------------------------------------------------
# SECTION 4: PRODUCTION MODEL SNAPSHOT
# -----------------------------------------------------------------------------
render_model_snapshot(model_metrics)

# -----------------------------------------------------------------------------
# SECTION 5: ETHICAL GOVERNANCE & RESPONSIBLE AI
# -----------------------------------------------------------------------------
render_disclaimer_box()

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
render_page_footer()
