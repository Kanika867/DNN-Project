"""
EduMetric — Centralized UI Design System & Theme Engine
Provides unified typography, design tokens, responsive CSS, application shell,
sidebar navigation, and standardized Plotly visualization theming.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# BRAND & DESIGN TOKENS
# =============================================================================
BRAND_NAME = "EduMetric"
BRAND_DESCRIPTOR = "Academic Performance Prediction & Intelligence Platform"
BRAND_VERSION = "v2.4 Enterprise"

# Core Color Palette
COLOR_NAVY_DARK = "#090D1A"       # Ultra-deep midnight navy (sidebar background)
COLOR_NAVY_SURFACE = "#0F172A"    # Deep slate surface
COLOR_NAVY_CARD = "#1E293B"       # Elevated slate container
COLOR_NAVY_BORDER = "#334155"     # Slate border token

COLOR_BG_PAGE = "#F8FAFC"         # Clean off-white page background
COLOR_SURFACE_WHITE = "#FFFFFF"   # Pure white container card
COLOR_BORDER_LIGHT = "#E2E8F0"    # Subtle border stroke
COLOR_BORDER_STRONG = "#CBD5E1"   # Defined border stroke

COLOR_TEXT_PRIMARY = "#0F172A"    # High-contrast slate heading
COLOR_TEXT_SECONDARY = "#475569"  # Balanced body text
COLOR_TEXT_MUTED = "#64748B"      # Subtitle and secondary metadata
COLOR_TEXT_SUBTLE = "#94A3B8"     # Micro labels and hints

# Primary Brand Accents
COLOR_ACCENT_PRIMARY = "#4F46E5"  # Electric Indigo
COLOR_ACCENT_HOVER = "#4338CA"    # Deep Indigo
COLOR_ACCENT_SOFT = "#EEF2FF"     # Subtle Indigo tint
COLOR_ACCENT_BORDER = "#C7D2FE"   # Indigo border tint

COLOR_COBALT = "#2563EB"          # Institutional Royal Blue
COLOR_CYAN = "#0EA5E9"            # Metric / Insight Cyan

# Calibrated Semantic Performance Indicators (4 Target Classes)
SEMANTIC_COLORS = {
    0: {
        "name": "Poor",
        "range": "0–9",
        "primary": "#E11D48",    # Crimson / Rose
        "soft_bg": "#FFF1F2",
        "border": "#FECDD3",
        "text": "#9F1239",
        "icon": "⚠️",
        "badge": "Academic Support Required"
    },
    1: {
        "name": "Average",
        "range": "10–12",
        "primary": "#D97706",    # Warm Amber
        "soft_bg": "#FFFBEB",
        "border": "#FDE68A",
        "text": "#92400E",
        "icon": "📋",
        "badge": "Core Baseline Standing"
    },
    2: {
        "name": "Good",
        "range": "13–15",
        "primary": "#2563EB",    # Royal Cobalt
        "soft_bg": "#EFF6FF",
        "border": "#BFDBFE",
        "text": "#1E40AF",
        "icon": "📘",
        "badge": "Proficient Academic Mastery"
    },
    3: {
        "name": "Excellent",
        "range": "16–20",
        "primary": "#059669",    # Emerald Jade
        "soft_bg": "#ECFDF5",
        "border": "#A7F3D0",
        "text": "#065F46",
        "icon": "🏆",
        "badge": "Outstanding Honors Level"
    }
}

CLASS_HEX_LIST = [
    SEMANTIC_COLORS[0]["primary"],
    SEMANTIC_COLORS[1]["primary"],
    SEMANTIC_COLORS[2]["primary"],
    SEMANTIC_COLORS[3]["primary"]
]

# =============================================================================
# GLOBAL CSS INJECTION
# =============================================================================
def inject_global_theme():
    """
    Injects the complete modern SaaS design system into the Streamlit app.
    Imports Inter and Plus Jakarta Sans Google Fonts, styles the entire shell,
    unifies cards, buttons, tables, forms, sliders, and navigation.
    """
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

            /* Global Font and Canvas Setup */
            html, body, [class*="css"], [class*="st-"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                letter-spacing: -0.011em;
            }

            .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
                background-color: #F8FAFC !important;
                color: #0F172A !important;
            }

            .block-container {
                padding-top: 1.8rem !important;
                padding-bottom: 3.5rem !important;
                max-width: 1320px !important;
                background-color: #F8FAFC !important;
            }

            /* Force light theme text hierarchy regardless of OS dark mode */
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
                color: #0F172A !important;
            }

            .stApp p, .stApp span:not([class*="badge"]), .stApp label {
                color: #334155 !important;
            }

            /* Hide default Streamlit multipage auto-nav to use our premium custom sidebar */
            [data-testid="stSidebarNav"] {
                display: none !important;
            }

            /* Sidebar Styling */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #090D1A 0%, #0F172A 100%) !important;
                border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
                box-shadow: 4px 0 24px rgba(0, 0, 0, 0.15) !important;
            }

            [data-testid="stSidebar"] hr {
                border-color: rgba(255, 255, 255, 0.1) !important;
                margin: 1.2rem 0 !important;
            }

            /* Typography */
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
                letter-spacing: -0.025em !important;
                font-weight: 700 !important;
                color: #0F172A;
            }

            /* Metric Display Numbers (Tabular figures for alignment) */
            div[data-testid="stMetricValue"] {
                font-family: 'Plus Jakarta Sans', sans-serif !important;
                font-weight: 800 !important;
                font-feature-settings: "tnum" 1 !important;
                color: #0F172A !important;
            }

            div[data-testid="stMetricLabel"] {
                font-size: 0.76rem !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.06em !important;
                color: #64748B !important;
            }

            /* Reusable Card Components */
            .edum-card {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 14px;
                padding: 22px 24px;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(15, 23, 42, 0.02);
                transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
                margin-bottom: 16px;
            }

            .edum-card:hover {
                border-color: #CBD5E1;
                box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.08), 0 2px 4px -2px rgba(15, 23, 42, 0.05);
            }

            .edum-card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                border-bottom: 1px solid #F1F5F9;
                padding-bottom: 14px;
                margin-bottom: 16px;
            }

            .edum-card-title {
                font-family: 'Plus Jakarta Sans', sans-serif;
                font-size: 1.05rem;
                font-weight: 700;
                color: #0F172A;
                margin: 0;
            }

            .edum-card-subtitle {
                font-size: 0.82rem;
                color: #64748B;
                margin-top: 3px;
                line-height: 1.4;
            }

            /* Hero Banner */
            .edum-hero {
                background: linear-gradient(135deg, #090D1A 0%, #0F172A 40%, #1E1B4B 80%, #312E81 100%);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 16px;
                padding: 32px 36px;
                color: #FFFFFF;
                margin-bottom: 24px;
                position: relative;
                overflow: hidden;
                box-shadow: 0 8px 30px rgba(15, 23, 42, 0.12);
            }

            .edum-hero::after {
                content: "";
                position: absolute;
                top: -50%;
                right: -20%;
                width: 400px;
                height: 400px;
                background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, rgba(99, 102, 241, 0) 70%);
                pointer-events: none;
            }

            .edum-badge-pill {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 9999px;
                padding: 4px 12px;
                font-size: 0.75rem;
                font-weight: 600;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                color: #E2E8F0;
                margin-bottom: 12px;
            }

            /* Custom Nav Item Styling in Sidebar */
            .nav-item {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 10px 14px;
                border-radius: 10px;
                font-size: 0.88rem;
                font-weight: 500;
                color: #94A3B8;
                text-decoration: none !important;
                margin-bottom: 4px;
                transition: all 0.15s ease;
                border: 1px solid transparent;
            }

            .nav-item:hover {
                background: rgba(255, 255, 255, 0.05);
                color: #FFFFFF;
                border-color: rgba(255, 255, 255, 0.08);
            }

            .nav-item.active {
                background: linear-gradient(90deg, rgba(79, 70, 229, 0.22) 0%, rgba(79, 70, 229, 0.08) 100%);
                color: #FFFFFF;
                font-weight: 600;
                border-left: 3px solid #6366F1;
                border-top: 1px solid rgba(99, 102, 241, 0.2);
                border-right: 1px solid rgba(99, 102, 241, 0.15);
                border-bottom: 1px solid rgba(99, 102, 241, 0.15);
            }

            /* Form Polish */
            div[data-testid="stForm"] {
                border: 1px solid #E2E8F0 !important;
                border-radius: 16px !important;
                background-color: #FFFFFF !important;
                padding: 24px 28px !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
            }

            /* Streamlit Input Labels */
            label[data-testid="stWidgetLabel"] p {
                font-size: 0.82rem !important;
                font-weight: 600 !important;
                color: #1E293B !important;
                margin-bottom: 4px !important;
            }

            /* Buttons */
            button[kind="primary"] {
                background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
                color: #FFFFFF !important;
                border: none !important;
                border-radius: 10px !important;
                font-weight: 600 !important;
                font-size: 0.9rem !important;
                padding: 8px 18px !important;
                box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
                transition: all 0.15s ease !important;
            }

            button[kind="primary"]:hover {
                background: linear-gradient(135deg, #4338CA 0%, #3730A3 100%) !important;
                box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35) !important;
                transform: translateY(-1px) !important;
            }

            button[kind="secondary"] {
                background-color: #FFFFFF !important;
                color: #1E293B !important;
                border: 1px solid #CBD5E1 !important;
                border-radius: 10px !important;
                font-weight: 600 !important;
                font-size: 0.85rem !important;
                transition: all 0.15s ease !important;
            }

            button[kind="secondary"]:hover {
                border-color: #94A3B8 !important;
                background-color: #F8FAFC !important;
                transform: translateY(-1px) !important;
            }

            /* Section Sub-Dividers */
            .edum-section-heading {
                display: flex;
                align-items: center;
                gap: 10px;
                font-family: 'Plus Jakarta Sans', sans-serif;
                font-size: 1.15rem;
                font-weight: 700;
                color: #0F172A;
                margin-top: 26px;
                margin-bottom: 16px;
                padding-bottom: 8px;
                border-bottom: 1.5px solid #E2E8F0;
            }

            /* Custom Dataframe Container */
            div[data-testid="stDataFrame"] {
                border: 1px solid #E2E8F0 !important;
                border-radius: 12px !important;
                overflow: hidden !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
            }

            /* Native Streamlit Bordered Container Polish */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background-color: #FFFFFF !important;
                border: 1px solid #E2E8F0 !important;
                border-radius: 14px !important;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
                margin-bottom: 14px !important;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] > div {
                background-color: #FFFFFF !important;
            }

            /* Expander Polish */
            details[data-testid="stExpander"] {
                border: 1px solid #E2E8F0 !important;
                border-radius: 12px !important;
                background-color: #FFFFFF !important;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02) !important;
            }

            details[data-testid="stExpander"] summary {
                font-family: 'Plus Jakarta Sans', sans-serif !important;
                font-weight: 600 !important;
                color: #0F172A !important;
                font-size: 0.9rem !important;
            }

            /* Footer Styling */
            .edum-footer {
                text-align: center;
                color: #94A3B8;
                font-size: 0.8rem;
                padding: 28px 0 12px 0;
                margin-top: 40px;
                border-top: 1px solid #E2E8F0;
            }

            /* Mobile & Tablet Responsiveness */
            @media (max-width: 768px) {
                .block-container {
                    padding-top: 1rem !important;
                    padding-left: 1rem !important;
                    padding-right: 1rem !important;
                }
                .edum-hero {
                    padding: 24px 20px !important;
                }
                .edum-hero h1 {
                    font-size: 1.6rem !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# UNIFIED APPLICATION SHELL COMPONENTS
# =============================================================================

def render_unified_sidebar(current_page="overview"):
    """
    Renders the consistent, polished EduMetric institutional sidebar shell.
    Provides clear brand identity, active page states, system status indicators,
    and institutional metadata.
    """
    with st.sidebar:
        # 1. Brand Logo & Header
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 12px; padding: 6px 4px 16px 4px;">
                <div style="width: 40px; height: 40px; border-radius: 10px; background: linear-gradient(135deg, #4F46E5 0%, #312E81 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35); border: 1px solid rgba(255, 255, 255, 0.2);">
                    <span style="font-size: 1.35rem;">📊</span>
                </div>
                <div>
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.3rem; font-weight: 800; letter-spacing: -0.03em; color: #FFFFFF; line-height: 1.1;">
                        {BRAND_NAME}
                    </div>
                    <div style="font-size: 0.68rem; font-weight: 600; color: #818CF8; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px;">
                        Academic Intelligence
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2. System Status Badge
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8;">System Status</span>
                    <span style="display: inline-flex; align-items: center; gap: 5px; font-size: 0.72rem; font-weight: 600; color: #34D399;">
                        <span style="height: 7px; width: 7px; background-color: #10B981; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #10B981;"></span>
                        Online
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: #E2E8F0; margin-top: 6px; font-weight: 500;">
                    Pipeline: <strong style="color: #FFFFFF;">Gradient Boosting</strong><br>
                    Cohort: <strong style="color: #FFFFFF;">649 Students Active</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 3. Navigation Menu Links
        pages = [
            ("overview", "🏠", "Overview Dashboard", "/"),
            ("prediction", "🔮", "Student Prediction", "/Student_Prediction"),
            ("explorer", "🔎", "Student Explorer", "/Student_Explorer"),
            ("analytics", "📊", "Performance Analytics", "/Performance_Analytics"),
            ("lab", "🧪", "Model Laboratory", "/Model_Laboratory"),
        ]

        st.markdown(
            """
            <div style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B; margin-bottom: 8px; padding-left: 6px;">
                Platform Modules
            </div>
            """,
            unsafe_allow_html=True
        )

        for key, icon, label, url in pages:
            active_cls = "active" if current_page == key else ""
            st.markdown(
                f"""
                <a href="{url}" target="_self" class="nav-item {active_cls}">
                    <span style="font-size: 1.15rem;">{icon}</span>
                    <span>{label}</span>
                </a>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # 4. Institutional Metadata Footer
        st.markdown(
            f"""
            <div style="font-size: 0.72rem; color: #64748B; padding: 4px 6px; line-height: 1.5;">
                <div style="color: #94A3B8; font-weight: 600;">{BRAND_NAME} {BRAND_VERSION}</div>
                <div>Calibrated on UCI Portuguese Secondary Dataset</div>
                <div style="margin-top: 4px; color: #475569;">Secondary Education Cohort</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_page_header(title, subtitle, badge_text="Production Intelligence Suite", icon="🎓"):
    """
    Renders an impressive, consistent SaaS hero banner for every page.
    """
    st.markdown(
        f"""
        <div class="edum-hero">
            <div class="edum-badge-pill">
                <span>{icon}</span>
                <span>{badge_text}</span>
            </div>
            <h1 style="font-size: 2.15rem; font-weight: 800; margin: 0; line-height: 1.2; color: #FFFFFF;">
                {title}
            </h1>
            <p style="font-size: 1rem; color: #CBD5E1; margin-top: 10px; margin-bottom: 0; max-width: 860px; line-height: 1.55;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_page_footer():
    """
    Renders the consistent institutional footer across all pages.
    """
    st.markdown(
        f"""
        <div class="edum-footer">
            <div style="font-weight: 600; color: #64748B;">{BRAND_NAME} — {BRAND_DESCRIPTOR}</div>
            <div style="margin-top: 4px; font-size: 0.75rem; color: #94A3B8;">
                Engineered for Academic Research, Institutional Advisors & Faculty Leadership · Calibrated on UCI Secondary Education Data
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# PLOTLY THEME ENGINE
# =============================================================================

def apply_plotly_theme(fig, title=None, height=320, show_legend=False):
    """
    Standardizes Plotly figures with modern typography, crisp slate axes,
    subtle gridlines, and clean padding matching the EduMetric design tokens.
    """
    layout_update = dict(
        font=dict(family="Inter, sans-serif", color=COLOR_TEXT_SECONDARY, size=11),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=35, r=25, t=48 if title else 25, b=35),
        height=height,
        xaxis=dict(
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            linecolor="#CBD5E1",
            tickfont=dict(size=11, color=COLOR_TEXT_MUTED)
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            linecolor="#CBD5E1",
            tickfont=dict(size=11, color=COLOR_TEXT_MUTED)
        )
    )

    if title:
        layout_update["title"] = dict(
            text=f"<b>{title}</b>",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=13.5, color=COLOR_TEXT_PRIMARY),
            x=0.01,
            xanchor="left"
        )

    if show_legend:
        layout_update["legend"] = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=COLOR_TEXT_MUTED),
            bgcolor="rgba(255,255,255,0.8)"
        )
    else:
        layout_update["showlegend"] = False

    fig.update_layout(**layout_update)
    return fig


def render_safe_dataframe(df, **kwargs):
    """
    Renders dataframe safely via Streamlit dataframe with automatic fallback to st.table
    in case of any serialization or environment IPC errors.
    """
    try:
        st.dataframe(df, **kwargs)
    except Exception:
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ["use_container_width", "height", "hide_index"]}
        st.table(df, **clean_kwargs)

