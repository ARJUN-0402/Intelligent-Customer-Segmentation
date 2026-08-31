"""Design-system tokens for the Customer Intelligence dashboard.

A single source of truth for colours, typography, and spacing so that every
component renders consistently. Values are deliberately restrained: a small blue
primary plus selective semantic accents, high-contrast ink/muted text, and a
white paper surface. Personas keep stable identity colours so segments remain
recognisable across every chart and card.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Colour palette
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
# These are deliberately reused across charts, cards and borders so a segment is
# always the same colour regardless of where it appears.
PERSONA_COLORS: dict[str, str] = {
    "vip": "#16a34a",       # High-Value Customers
    "saver": "#2563eb",     # Premium Savers
    "impulsive": "#dc2626", # Budget-Conscious Spenders
    "budget": "#7c3aed",    # Low-Engagement Customers
    "mainstream": "#ea580c",# Growth Opportunity Customers
}

# Neutral fallback palette for clusters without a persona mapping.
NEUTRAL_COLORS: list[str] = ["#94a3b8", "#cbd5e1", "#e2e8f0"]

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_FAMILY: str = (
    "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', "
    "Roboto, Helvetica, Arial, sans-serif"
)

# Font family for Plotly chart text (mirrors the app body font).
CHART_FONT: str = FONT_FAMILY


# Relative size ladder (avoids excessive font-size variation)
TEXT_TITLE = "1.7rem"          # app title
TEXT_SECTION = "1.25rem"       # h2 section headings
TEXT_SUBHEAD = "1.0rem"        # h3 subheadings
TEXT_BODY = "0.95rem"          # body copy
TEXT_CAPTION = "0.8rem"        # captions / muted
TEXT_METRIC_LARGE = "2rem"     # large metric value in KPI card
TEXT_METRIC_MED = "1.5rem"     # medium metric value
TEXT_METRIC_SMALL = "1.1rem"   # small metric value

# ---------------------------------------------------------------------------
# Spacing (rem-based, consistent across layout)
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


def cluster_color_map(personas: dict | None, n_clusters: int) -> list[str]:
    """Return a colour for each cluster id, falling back to neutrals.

    Parameters
    ----------
    personas:
        Mapping of cluster_id -> Persona (with a ``.key`` attribute).
    n_clusters:
        Total number of clusters to cover.
    """
    colors: list[str] = []
    for cid in range(n_clusters):
        p = personas.get(cid) if personas else None
        if p is not None:
            colors.append(PERSONA_COLORS.get(p.key, "#94a3b8"))
        else:
            colors.append(NEUTRAL_COLORS[cid % len(NEUTRAL_COLORS)])
    return colors


def inject_global_styles() -> None:
    """Inject the dashboard's global CSS (fonts, spacing, typography)."""
    css = f"""
    <style>
      html, body, .stApp {{
        font-family: {FONT_FAMILY};
        color: {INK};
        background-color: {PAPER};
      }}
      .block-container {{
        padding-top: {SPACE_PAGE};
        padding-bottom: {SPACE_PAGE};
        max-width: 1280px;
      }}
      h1 {{ font-size: {TEXT_TITLE}; font-weight: 700; color: {INK}; margin-bottom: 0.25rem; }}
      h2 {{ font-size: {TEXT_SECTION}; font-weight: 600; color: {INK}; margin-top: 0; }}
      h3 {{ font-size: {TEXT_SUBHEAD}; font-weight: 600; color: {INK}; }}
      .stMarkdown {{ font-size: {TEXT_BODY}; line-height: 1.5; }}
      .stCaption {{ color: {MUTED}; font-size: {TEXT_CAPTION}; font-weight: 500; }}
      .stExpander {{ border: 1px solid {BORDER}; border-radius: {CARD_RADIUS}; }}
      /* Sidebar navigation radio: clean, no extra gutters */
      .stSidebar .stRadio [role="radio"] {{
        padding: 6px 8px;
      }}
      .stSidebar .stRadio [role="radio"][aria-checked="true"] {{
        color: {PRIMARY};
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
