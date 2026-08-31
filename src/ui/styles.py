"""Design-system tokens for the Customer Intelligence dashboard.

A single source of truth for colours, typography, and spacing. Values are
deliberately restrained: a small blue primary plus selective semantic accents,
high-contrast text, and clean surfaces. Personas keep stable identity colours
so segments remain recognisable across every chart and card.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Theme detection
# ---------------------------------------------------------------------------

def _detect_theme() -> str:
    """Return ``"dark"`` or ``"light"`` based on the active Streamlit theme."""
    try:
        theme = st.get_option("theme.base")
        if theme in ("light", "dark"):
            return theme
    except Exception:
        pass
    return "light"


# ---------------------------------------------------------------------------
# Colour palette (light-mode defaults; overridden for dark mode in CSS)
# ---------------------------------------------------------------------------
# Primary accent
PRIMARY: str = "#2563eb"

# Semantic accents (used selectively, not for decoration alone)
SUCCESS: str = "#16a34a"
WARNING: str = "#ea580c"
ERROR: str = "#dc2626"
PURPLE: str = "#7c3aed"
TEAL: str = "#0891b2"
AMBER: str = "#ca8a04"

# Neutral text / surface
INK: str = "#0f172a"
MUTED: str = "#64748b"
TEXT_SECONDARY: str = "#94a3b8"
BORDER: str = "#e2e8f0"
SURFACE: str = "#f8fafc"
PAPER: str = "#ffffff"

# Chart palette — restrained, colourblind-friendly and distinct at a glance.
PALETTE: list[str] = [
    "#2563eb", "#16a34a", "#dc2626", "#7c3aed",
    "#ea580c", "#0891b2", "#ca8a04", "#db2777",
]

# Stable persona identity colours (keyed by persona.key from src/personas.py).
PERSONA_COLORS: dict[str, str] = {
    "vip": "#16a34a",
    "saver": "#2563eb",
    "impulsive": "#dc2626",
    "budget": "#7c3aed",
    "mainstream": "#ea580c",
}

NEUTRAL_COLORS: list[str] = ["#94a3b8", "#cbd5e1", "#e2e8f0"]

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_FAMILY: str = (
    "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', "
    "Roboto, Helvetica, Arial, sans-serif"
)
CHART_FONT: str = FONT_FAMILY

TEXT_TITLE = "1.7rem"
TEXT_SECTION = "1.25rem"
TEXT_SUBHEAD = "1.0rem"
TEXT_BODY = "0.95rem"
TEXT_CAPTION = "0.8rem"
TEXT_METRIC_LARGE = "2rem"
TEXT_METRIC_MED = "1.5rem"
TEXT_METRIC_SMALL = "1.1rem"

# ---------------------------------------------------------------------------
# Spacing
# ---------------------------------------------------------------------------
SPACE_PAGE: str = "1.5rem"
SPACE_SECTION: str = "1.5rem"
SPACE_CARD: str = "1rem"
SPACE_GAP: str = "0.75rem"
SPACE_INNER: str = "0.5rem"

# ---------------------------------------------------------------------------
# Surface / elevation
# ---------------------------------------------------------------------------
CARD_RADIUS: str = "10px"
CARD_SHADOW: str = "0 1px 2px rgba(15, 23, 42, 0.04)"

COLORWAY = PALETTE


# ---------------------------------------------------------------------------
# Theme-aware helpers
# ---------------------------------------------------------------------------

def cluster_color_map(personas: dict | None, n_clusters: int) -> list[str]:
    """Return a colour for each cluster id, falling back to neutrals."""
    colors: list[str] = []
    for cid in range(n_clusters):
        p = personas.get(cid) if personas else None
        if p is not None:
            colors.append(PERSONA_COLORS.get(p.key, "#94a3b8"))
        else:
            colors.append(NEUTRAL_COLORS[cid % len(NEUTRAL_COLORS)])
    return colors


def _theme_colors() -> dict[str, str]:
    """Return a dict of theme-aware color tokens for the active theme."""
    if _detect_theme() == "dark":
        return {
            "bg": "#0f172a",
            "ink": "#f1f5f9",
            "muted": "#94a3b8",
            "border": "#334155",
            "paper": "#1e293b",
            "surface": "#1e293b",
            "chart_bg": "#1e293b",
            "chart_grid": "#334155",
            "chart_text": "#94a3b8",
            "sidebar_bg": "#0f172a",
            "sidebar_text": "#f1f5f9",
            "input_bg": "#1e293b",
            "shadow": "0 1px 2px rgba(0, 0, 0, 0.3)",
            "text_secondary": "#cbd5e1",
        }
    return {
        "bg": "#ffffff",
        "ink": "#0f172a",
        "muted": "#64748b",
        "border": "#e2e8f0",
        "paper": "#ffffff",
        "surface": "#f8fafc",
        "chart_bg": "#ffffff",
        "chart_grid": "#eef2f7",
        "chart_text": "#64748b",
        "sidebar_bg": "#f8fafc",
        "sidebar_text": "#0f172a",
        "input_bg": "#ffffff",
        "shadow": "0 1px 2px rgba(15, 23, 42, 0.04)",
        "text_secondary": "#94a3b8",
    }


def get_theme() -> dict[str, str]:
    """Return semantic theme tokens for the active Streamlit theme."""
    c = _theme_colors()
    return {
        "name": _detect_theme(),
        "background": c["bg"],
        "surface": c["surface"],
        "surface_elevated": c["paper"],
        "border": c["border"],
        "text_primary": c["ink"],
        "text_secondary": c.get("text_secondary", c["muted"]),
        "text_muted": c["muted"],
        "accent": PRIMARY,
        "success": SUCCESS,
        "warning": WARNING,
        "danger": ERROR,
        "info": TEAL,
    }


def inject_global_styles() -> None:
    """Inject the dashboard's global CSS (fonts, spacing, typography, cards)."""
    c = _theme_colors()
    css = f"""
    <style>
    /* ── Reset / base ─────────────────────────────────────────── */
    html, body, .stApp {{
        font-family: {FONT_FAMILY};
        color: {c['ink']};
        background-color: {c['bg']};
    }}
    .block-container {{
        padding-top: {SPACE_PAGE};
        padding-bottom: {SPACE_PAGE};
        max-width: 1280px;
    }}

    /* ── Typography hierarchy ─────────────────────────────────── */
    h1 {{
        font-size: {TEXT_TITLE}; font-weight: 700; color: {c['ink']};
        margin-bottom: 0.25rem;
    }}
    h2 {{
        font-size: {TEXT_SECTION}; font-weight: 600; color: {c['ink']};
        margin-top: 0;
    }}
    h3 {{
        font-size: {TEXT_SUBHEAD}; font-weight: 600; color: {c['ink']};
    }}
    .stMarkdown {{
        font-size: {TEXT_BODY}; line-height: 1.5; color: {c['ink']};
    }}
    .stCaption {{
        color: {c['muted']}; font-size: {TEXT_CAPTION}; font-weight: 500;
    }}
    .stExpander {{
        border: 1px solid {c['border']}; border-radius: {CARD_RADIUS};
    }}

    /* ── Sidebar ──────────────────────────────────────────────── */
    .stSidebar {{
        background-color: {c['sidebar_bg']} !important;
        color: {c['sidebar_text']} !important;
    }}
    .stSidebar .stMarkdown, .stSidebar .stCaption {{
        color: {c['sidebar_text']} !important;
    }}
    .stSidebar .stRadio [role="radio"] {{
        padding: 8px 10px; border-radius: 6px;
    }}
    .stSidebar .stRadio [role="radio"][aria-checked="true"] {{
        background-color: {c['surface']}; color: {PRIMARY};
    }}
    .stSidebar .stSlider [data-baseweb="slider"] {{
        color: {PRIMARY};
    }}
    .stSidebar .stSelectbox > div > div {{
        background-color: {c['input_bg']};
    }}
    .sidebar-section {{
        margin-bottom: 1.25rem;
    }}
    .sidebar-section-title {{
        font-size: {TEXT_CAPTION}; font-weight: 700; color: {c['muted']};
        text-transform: uppercase; letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }}

    /* ── Card system (CSS-class based, no inline layout styles) ── */
    .kpi-card {{
        background: {c['paper']}; border: 1px solid {c['border']};
        border-left: 4px solid var(--accent, {PRIMARY});
        border-radius: {CARD_RADIUS}; padding: 16px 18px;
        box-shadow: {c['shadow']}; height: 100%;
    }}
    .kpi-label {{
        font-size: {TEXT_CAPTION}; color: {c['muted']}; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .kpi-value {{
        font-size: {TEXT_METRIC_LARGE}; color: {c['ink']}; font-weight: 700;
        margin-top: 8px; line-height: 1.1;
    }}
    .kpi-sub {{
        font-size: {TEXT_CAPTION}; color: {c['muted']}; margin-top: 6px;
    }}

    .metric-card {{
        background: {c['paper']}; border: 1px solid {c['border']};
        border-left: 4px solid var(--accent, {PRIMARY});
        border-radius: {CARD_RADIUS}; padding: 14px 16px;
        box-shadow: {c['shadow']}; height: 100%;
    }}
    .metric-label {{
        font-size: {TEXT_CAPTION}; color: {c['muted']}; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .metric-value {{
        font-size: {TEXT_METRIC_MED}; color: {c['ink']}; font-weight: 700;
        margin-top: 6px;
    }}
    .metric-direction {{
        font-size: {TEXT_CAPTION}; color: {c['muted']}; margin-top: 4px;
    }}
    .metric-note {{
        font-size: {TEXT_CAPTION}; color: {c['muted']}; margin-top: 2px;
    }}

    .persona-card {{
        background: {c['paper']}; border: 1px solid {c['border']};
        border-left: 4px solid var(--accent, {BORDER});
        border-radius: {CARD_RADIUS}; padding: 16px;
        box-shadow: {c['shadow']}; height: 100%;
        cursor: pointer; transition: box-shadow 0.15s ease, border-color 0.15s ease;
    }}
    .persona-card.selected {{
        border-left-color: var(--accent, {PRIMARY});
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }}
    .persona-name {{
        font-size: 1.05rem; font-weight: 700; color: {c['ink']}; margin-top: 4px;
    }}
    .persona-stat {{
        font-size: {TEXT_METRIC_LARGE}; font-weight: 700; color: {c['ink']};
    }}
    .persona-stat-accent {{
        font-size: {TEXT_METRIC_LARGE}; font-weight: 700;
        color: var(--accent, {PRIMARY});
    }}
    .persona-caption {{
        font-size: {TEXT_CAPTION}; color: {c['muted']};
    }}
    .persona-row {{
        display: flex; justify-content: space-between; align-items: flex-start;
    }}
    .persona-dot {{
        width: 12px; height: 12px; border-radius: 50%;
        background: var(--accent, {PRIMARY});
    }}
    .persona-stats {{
        display: flex; gap: 16px; margin-top: 12px;
    }}

    .insight-card {{
        background: {c['paper']}; border: 1px solid {c['border']};
        border-left: 4px solid var(--accent, {PRIMARY});
        border-radius: {CARD_RADIUS}; padding: 16px 18px;
        box-shadow: {c['shadow']}; height: 100%;
    }}
    .insight-title {{
        font-size: {TEXT_CAPTION}; font-weight: 700; color: var(--accent, {PRIMARY});
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .insight-body {{
        font-size: {TEXT_BODY}; color: {c['ink']}; margin-top: 8px;
        line-height: 1.45;
    }}

    .config-card {{
        background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: {CARD_RADIUS}; padding: 14px 16px;
        box-shadow: {c['shadow']};
    }}
    .config-line {{
        font-size: {TEXT_BODY}; color: {c['ink']}; margin: 2px 0;
    }}

    .recommendation-box {{
        background: {c['surface']}; border: 1px solid {c['border']};
        border-left: 4px solid var(--accent, {PRIMARY});
        border-radius: {CARD_RADIUS}; padding: 16px 18px;
        box-shadow: {c['shadow']}; margin: 12px 0;
    }}
    .recommendation-title {{
        font-size: {TEXT_CAPTION}; font-weight: 700; color: var(--accent, {PRIMARY});
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .recommendation-body {{
        font-size: {TEXT_BODY}; color: {c['ink']}; margin-top: 8px;
        line-height: 1.5;
    }}
    .recommendation-list {{
        margin: 8px 0 0 18px; padding: 0;
    }}
    .recommendation-item {{
        margin-bottom: 4px; font-size: {TEXT_BODY}; color: {c['ink']};
    }}

    .surface-box {{
        background: {c['paper']}; border: 1px solid {c['border']};
        border-radius: {CARD_RADIUS}; padding: 16px 18px;
        box-shadow: {c['shadow']}; height: 100%;
    }}
    .surface-box-bar {{
        border-left: 4px solid var(--accent, {PRIMARY});
        padding: 4px 0 4px 14px; margin-bottom: 10px;
    }}
    .surface-box-bar-title {{
        font-size: 1.25rem; font-weight: 700; color: {c['ink']};
    }}
    .surface-box-tag {{
        font-size: {TEXT_CAPTION}; font-weight: 700; color: {c['muted']};
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .surface-box-plain {{
        font-size: {TEXT_SUBHEAD}; font-weight: 700; color: {c['ink']};
        margin-bottom: 6px;
    }}
    .surface-box-body {{
        font-size: {TEXT_BODY}; color: {c['ink']}; line-height: 1.5;
    }}
    .persona-header-title {{
        font-size: 1.4rem; font-weight: 700; color: {c['ink']};
    }}
    .persona-header-sub {{
        font-size: 0.85rem; color: {c['muted']};
    }}
    .result-title {{
        font-size: 1.6rem; font-weight: 700; color: {c['ink']}; margin: 6px 0;
    }}
    .result-subtitle {{
        font-size: 1.2rem; font-weight: 600; color: var(--accent, {PRIMARY});
        margin-bottom: 10px;
    }}

    /* ── App header ───────────────────────────────────────────── */
    .app-header {{
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; margin-bottom: 1rem; gap: 8px;
    }}
    .app-header-brand {{
        display: flex; flex-direction: column; gap: 2px;
    }}
    .app-header-appname {{
        font-size: 0.75rem; font-weight: 700; color: {c['muted']};
        text-transform: uppercase; letter-spacing: 0.08em;
    }}
    .app-header-tagline {{
        font-size: 1rem; font-weight: 600; color: {c['ink']};
    }}
    .app-header-meta {{
        display: flex; gap: 14px; flex-wrap: wrap; align-items: center;
    }}
    .app-header-meta span {{
        font-size: {TEXT_CAPTION}; color: {c['muted']};
    }}
    .app-header-meta b {{
        color: {c['ink']};
    }}

    /* ── Hero / headline box ──────────────────────────────────── */
    .hero-box {{
        background: {c['surface']}; border: 1px solid {c['border']};
        border-left: 4px solid {PRIMARY}; border-radius: {CARD_RADIUS};
        padding: 18px 20px; margin-bottom: 1.25rem;
    }}
    .hero-box-tag {{
        font-size: 0.8rem; font-weight: 700; color: {c['muted']};
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .hero-box-text {{
        font-size: 0.95rem; color: {c['ink']}; margin-top: 8px;
        line-height: 1.5;
    }}

    /* ── Section spacing ──────────────────────────────────────── */
    .section-gap {{
        margin-top: 1.5rem;
    }}

    /* ── Responsive helpers ───────────────────────────────────── */
    @media (max-width: 1100px) {{
        .kpi-grid-4 .kpi-card {{
            font-size: 0.9em;
        }}
    }}
    @media (max-width: 768px) {{
        .app-header {{
            flex-direction: column; align-items: flex-start;
        }}
        .kpi-value {{
            font-size: 1.5rem;
        }}
        .metric-value {{
            font-size: 1.25rem;
        }}
    }}

    /* ── Plotly chart overrides ───────────────────────────────── */
    .js-plotly-plot .plotly .main-svg {{
        background: {c['chart_bg']} !important;
    }}
    .js-plotly-plot .plotly .bg {{
        fill: {c['chart_bg']} !important;
    }}

    /* ── Component helpers ─────────────────────────────────────── */
    .status-dot {{
        display: inline-block; width: 8px; height: 8px;
        border-radius: 50%; margin-right: 6px;
    }}
    .status-label {{
        font-size: {TEXT_CAPTION}; color: {c['muted']};
        display: inline-flex; align-items: center; gap: 6px;
    }}
    .dataset-meta {{
        font-size: {TEXT_BODY}; font-weight: 600; color: {c['ink']};
        margin-left: 16px;
    }}
    .dataset-status {{
        font-size: {TEXT_CAPTION}; color: {c['muted']}; margin-left: 16px;
        margin-top: 2px;
    }}
    .empty-state {{
        text-align: center; padding: 24px 0; color: {c['muted']};
        font-size: {TEXT_BODY};
    }}
    .empty-state-icon {{
        font-size: 1.5rem; margin-bottom: 8px;
    }}
    .empty-state-message {{
        margin-top: 4px;
    }}
    .empty-state-hint {{
        margin-top: 6px; font-size: {TEXT_CAPTION};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def fmt_income(value: float | None) -> str:
    """Format an income value (in thousands) as a dollar string."""
    if value is None:
        return "n/a"
    return f"${float(value):,.1f}k"


def fmt_metric(value: float | int | None, digits: int = 3) -> str:
    """Format a numeric metric, returning a dash for missing / NaN values."""
    import numpy as np
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value):,.{digits}f}"
