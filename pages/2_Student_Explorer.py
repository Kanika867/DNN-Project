"""
EduMetric — Academic Performance Prediction & Intelligence Platform
Module 2: In-Memory Student Dataset Explorer & Cohort Analytics
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
    render_page_footer,
    render_safe_dataframe
)
from src.app_utils import (
    load_student_data,
    CLASS_COLORS,
    CLASS_HEX_LIST,
    DEFAULT_CLASS_MAPPING
)
from src.dashboard_components import (
    render_explorer_kpi_cards,
    render_filtered_distribution_chart,
    render_comparative_charts
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"Student Explorer | {BRAND_NAME}",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# INJECT UNIFIED DESIGN SYSTEM & APPLICATION SHELL
# -----------------------------------------------------------------------------
inject_global_theme()
render_unified_sidebar(current_page="explorer")

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
df_raw = load_student_data()

if df_raw is None:
    st.error("Failed to load student dataset. Please verify `data/student-por.csv` exists.")
    st.stop()

# -----------------------------------------------------------------------------
# HERO BANNER / PAGE HEADER
# -----------------------------------------------------------------------------
render_page_header(
    title="Student Academic Dataset Explorer",
    subtitle="Inspect individualized student academic records, apply multi-dimensional cohort filters, examine comparative cross-tabulations, and export custom cohort subsets for institutional analysis.",
    badge_text="Cohort Exploration & In-Memory Analytics",
    icon="🔎"
)

# -----------------------------------------------------------------------------
# SEARCH & MULTI-CRITERIA FILTER CONSOLE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="edum-section-heading">
        <span>⚙️</span>
        <span>Search & Multi-Criteria Cohort Filter Console</span>
    </div>
    """,
    unsafe_allow_html=True
)

filter_keys = [
    "exp_search", "exp_school", "exp_sex", "exp_age", "exp_address",
    "exp_famsize", "exp_studytime", "exp_failures", "exp_internet",
    "exp_category", "exp_absences"
]

with st.container(border=True):
    # Row 1: Search query and primary demographics
    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns([1.5, 1, 1, 1.1])
with r1_c1:
    search_query = st.text_input(
        "🔍 Search Anonymous Student ID",
        placeholder="e.g. Student-042 or 42",
        key="exp_search",
        help="Search by stable anonymous student identifier"
    )
with r1_c2:
    school_filter = st.selectbox(
        "School Campus",
        options=["All", "GP", "MS"],
        key="exp_school",
        format_func=lambda x: "All Campuses" if x == "All" else ("Gabriel Pereira (GP)" if x == "GP" else "Mousinho (MS)"),
        help="GP: Gabriel Pereira | MS: Mousinho da Silveira"
    )
with r1_c3:
    gender_filter = st.selectbox(
        "Gender",
        options=["All", "F", "M"],
        key="exp_sex",
        format_func=lambda x: "All Genders" if x == "All" else ("Female (F)" if x == "F" else "Male (M)")
    )
with r1_c4:
    category_filter = st.selectbox(
        "Performance Category",
        options=["All", "Poor (0-9)", "Average (10-12)", "Good (13-15)", "Excellent (16-20)"],
        key="exp_category",
        help="Final grade G3 classification category"
    )

# Row 2: Socio-demographic & Academic filters
r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
with r2_c1:
    studytime_filter = st.selectbox(
        "Weekly Study Time",
        options=["All", "1 (< 2 hrs)", "2 (2–5 hrs)", "3 (5–10 hrs)", "4 (> 10 hrs)"],
        key="exp_studytime",
        help="Dedicated non-classroom weekly revision hours"
    )
with r2_c2:
    failures_filter = st.selectbox(
        "Past Course Failures",
        options=["All", "0", "1", "2", "3 or more"],
        key="exp_failures",
        help="Historical academic setbacks"
    )
with r2_c3:
    address_filter = st.selectbox(
        "Residential Location",
        options=["All", "U", "R"],
        key="exp_address",
        format_func=lambda x: "All Locations" if x == "All" else ("Urban (U)" if x == "U" else "Rural (R)")
    )
with r2_c4:
    internet_filter = st.selectbox(
        "Home Internet Access",
        options=["All", "yes", "no"],
        key="exp_internet",
        format_func=lambda x: "All Digital Statuses" if x == "All" else ("Internet Connected" if x == "yes" else "No Internet")
    )

# Row 3: Sliders for Age and Absences
r3_c1, r3_c2 = st.columns(2)
with r3_c1:
    age_range = st.slider(
        "Student Age Range",
        min_value=int(df_raw["age"].min()),
        max_value=int(df_raw["age"].max()),
        value=(int(df_raw["age"].min()), int(df_raw["age"].max())),
        key="exp_age",
        help="Student chronological age range"
    )
with r3_c2:
    absence_range = st.slider(
        "Recorded School Absences Range (Days)",
        min_value=int(df_raw["absences"].min()),
        max_value=int(df_raw["absences"].max()),
        value=(int(df_raw["absences"].min()), int(df_raw["absences"].max())),
        key="exp_absences",
        help="Total recorded school absences"
    )

col_reset, _ = st.columns([1.5, 4])
with col_reset:
    if st.button("🔄 Reset All Filters to Cohort Default", type="secondary"):
        for k in filter_keys:
            st.session_state.pop(k, None)
        st.rerun()

# -----------------------------------------------------------------------------
# APPLY FILTERING LOGIC
# -----------------------------------------------------------------------------
df_filtered = df_raw.copy()

if search_query:
    q = search_query.strip().lower()
    df_filtered = df_filtered[df_filtered["student_id"].str.lower().str.contains(q)]

if school_filter != "All":
    df_filtered = df_filtered[df_filtered["school"] == school_filter]

if gender_filter != "All":
    df_filtered = df_filtered[df_filtered["sex"] == gender_filter]

if category_filter != "All":
    df_filtered = df_filtered[df_filtered["target_name"] == category_filter]

if studytime_filter != "All":
    st_val = int(studytime_filter.split()[0])
    df_filtered = df_filtered[df_filtered["studytime"] == st_val]

if failures_filter != "All":
    if failures_filter == "3 or more":
        df_filtered = df_filtered[df_filtered["failures"] >= 3]
    else:
        df_filtered = df_filtered[df_filtered["failures"] == int(failures_filter)]

if address_filter != "All":
    df_filtered = df_filtered[df_filtered["address"] == address_filter]

if internet_filter != "All":
    df_filtered = df_filtered[df_filtered["internet"] == internet_filter]

df_filtered = df_filtered[(df_filtered["age"] >= age_range[0]) & (df_filtered["age"] <= age_range[1])]
df_filtered = df_filtered[(df_filtered["absences"] >= absence_range[0]) & (df_filtered["absences"] <= absence_range[1])]

# -----------------------------------------------------------------------------
# DATASET OVERVIEW KPI CARDS
# -----------------------------------------------------------------------------
total_cnt = len(df_raw)
filtered_cnt = len(df_filtered)

avg_g3 = float(df_filtered["G3"].mean()) if filtered_cnt > 0 else None
median_g3 = float(df_filtered["G3"].median()) if filtered_cnt > 0 else None
school_cnt = int(df_filtered["school"].nunique()) if filtered_cnt > 0 else 0
class_cnt = int(df_filtered["target_class"].nunique()) if filtered_cnt > 0 else 0

render_explorer_kpi_cards(
    total_records=total_cnt,
    filtered_records=filtered_cnt,
    avg_g3=avg_g3,
    median_g3=median_g3,
    school_count=school_cnt,
    class_count=class_cnt
)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# EMPTY STATE HANDLING
# -----------------------------------------------------------------------------
if filtered_cnt == 0:
    st.markdown(
        """
        <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 12px; padding: 24px; text-align: center; margin: 20px 0;">
            <div style="font-size: 2rem;">⚠️</div>
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.15rem; font-weight: 700; color: #92400E; margin-top: 8px;">
                No Student Records Match the Active Filter Criteria
            </div>
            <div style="color: #78350F; font-size: 0.88rem; margin-top: 6px; max-width: 520px; margin-left: auto; margin-right: auto;">
                Please broaden your selected parameters or click the button below to restore the complete 649-student cohort.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("🔄 Reset All Filters to Restore Full Cohort", type="primary"):
        for k in filter_keys:
            st.session_state.pop(k, None)
        st.rerun()
    st.stop()

# -----------------------------------------------------------------------------
# STUDENT RECORDS TABLE SECTION
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="edum-section-heading">
        <span>📋</span>
        <span>Filtered Student Records ({filtered_cnt:,} Students Active)</span>
    </div>
    """,
    unsafe_allow_html=True
)

col_table_header, col_export = st.columns([3, 1.2])

with col_export:
    export_cols = ["student_id"] + [c for c in df_raw.columns if c not in ["student_id", "target_class", "target_name", "performance_category"]] + ["performance_category"]
    export_df = df_filtered[export_cols]
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Export Cohort (CSV)",
        data=csv_bytes,
        file_name="edumetric_filtered_students.csv",
        mime="text/csv",
        use_container_width=True,
        help="Export all 33 original features plus anonymous ID and category for the filtered cohort."
    )

with col_table_header:
    view_mode = st.radio(
        "Table Column View",
        options=["Core Academic Summary (11 Columns)", "All Original Features (34 Columns)"],
        horizontal=True
    )

if "Core Academic" in view_mode:
    display_cols = [
        "student_id", "school", "sex", "age", "studytime",
        "failures", "absences", "G1", "G2", "G3", "performance_category"
    ]
    column_renames = {
        "student_id": "Student ID",
        "school": "Campus",
        "sex": "Gender",
        "age": "Age",
        "studytime": "Study Tier",
        "failures": "Past Failures",
        "absences": "Absences",
        "G1": "Period 1 (G1)",
        "G2": "Period 2 (G2)",
        "G3": "Final Grade (G3)",
        "performance_category": "Performance Category"
    }
    table_df = df_filtered[display_cols].rename(columns=column_renames)
else:
    table_df = df_filtered

render_safe_dataframe(
    table_df,
    use_container_width=True,
    height=330,
    hide_index=True
)

st.caption(
    f"Displaying **{filtered_cnt}** of **{total_cnt}** student records. Click any column header to sort ascending or descending."
)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PERFORMANCE DISTRIBUTION CHART (FILTERED SUBSET)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="edum-section-heading">
        <span>📊</span>
        <span>Filtered Cohort Target Category Distribution</span>
    </div>
    """,
    unsafe_allow_html=True
)

with st.container(border=True):
    fig_dist = render_filtered_distribution_chart(df_filtered)
    if fig_dist:
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# COMPARATIVE ANALYTICS SECTION
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="edum-section-heading">
        <span>📈</span>
        <span>Comparative Visual Analytics (Filtered Cohort)</span>
    </div>
    <p style="color: #64748B; font-size: 0.88rem; margin-top: -10px; margin-bottom: 16px;">
        Interactive cross-tabulations dynamically evaluated on the filtered student cohort.
        <em>Note: Visualized trends reflect observed group distributions and do not imply direct mechanical causation.</em>
    </p>
    """,
    unsafe_allow_html=True
)

charts = render_comparative_charts(df_filtered)

if charts:
    comp_row1_c1, comp_row1_c2 = st.columns(2)
    with comp_row1_c1:
        with st.container(border=True):
            st.plotly_chart(charts["school"], use_container_width=True, config={"displayModeBar": False})
    with comp_row1_c2:
        with st.container(border=True):
            st.plotly_chart(charts["studytime"], use_container_width=True, config={"displayModeBar": False})

    comp_row2_c1, comp_row2_c2 = st.columns(2)
    with comp_row2_c1:
        with st.container(border=True):
            st.plotly_chart(charts["gender"], use_container_width=True, config={"displayModeBar": False})
    with comp_row2_c2:
        with st.container(border=True):
            st.plotly_chart(charts["failures"], use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        st.plotly_chart(charts["absences"], use_container_width=True, config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
render_page_footer()
