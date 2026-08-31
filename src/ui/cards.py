"""Reusable card components: KPI, metric, persona and insight cards.

Each card renders through a small amount of structural HTML combined with
CSS classes defined in ``styles.py``.  This avoids fragile inline-style
strings and keeps the visual design system in one place.
"""

from __future__ import annotations

import streamlit as st

from .styles import PRIMARY, TEXT_METRIC_LARGE


def _esc(text: str) -> str:
    """Minimal HTML-escape for user-provided strings."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def kpi_card(
    label: str,
    value: str,
    sub: str = "",
    color: str = PRIMARY,
    value_size: str = TEXT_METRIC_LARGE,
) -> None:
    """Render a single KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent:{color};">
            <div class="kpi-label">{_esc(label)}</div>
            <div class="kpi-value" style="font-size:{value_size};">{_esc(value)}</div>
            {f'<div class="kpi-sub">{_esc(sub)}</div>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    direction: str = "",
    note: str = "",
    color: str = PRIMARY,
) -> None:
    """A compact metric card used in the Model Evaluation grid."""
    st.markdown(
        f"""
        <div class="metric-card" style="--accent:{color};">
            <div class="metric-label">{_esc(label)}</div>
            <div class="metric-value">{_esc(value)}</div>
            {f'<div class="metric-direction">{_esc(direction)}</div>' if direction else ''}
            {f'<div class="metric-note">{_esc(note)}</div>' if note else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def persona_card(
    persona_name: str,
    persona_key: str,
    segment_id: int,
    customer_count: int,
    percentage: float,
    color: str,
    is_selected: bool = False,
    description: str = "",
) -> None:
    """Render a persona overview card used in the Segment Explorer."""
    selected_cls = "selected" if is_selected else ""
    weight = "700" if is_selected else "600"
    st.markdown(
        f"""
        <div class="persona-card {selected_cls}" style="--accent:{color};"
             data-persona="{_esc(persona_key)}">
            <div class="persona-row">
                <div>
                    <div class="persona-caption" style="font-weight:700;">
                        Segment {segment_id}
                    </div>
                    <div class="persona-name" style="font-weight:{weight};">
                        {_esc(persona_name)}
                    </div>
                </div>
                <div class="persona-dot"></div>
            </div>
            <div class="persona-stats">
                <div>
                    <div class="persona-stat">{customer_count:,}</div>
                    <div class="persona-caption">customers</div>
                </div>
                <div>
                    <div class="persona-stat-accent">{percentage:.1f}%</div>
                    <div class="persona-caption">of base</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(title: str, detail: str, color: str = PRIMARY) -> None:
    """A data-driven insight card with a coloured accent bar."""
    st.markdown(
        f"""
        <div class="insight-card" style="--accent:{color};">
            <div class="insight-title">{_esc(title)}</div>
            <div class="insight-body">{_esc(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def styled_metric_table(df) -> None:
    """Render a dataframe with a light, borderless presentation."""
    st.dataframe(df, width="stretch", hide_index=True)


def surface_box(
    title: str,
    body: str = "",
    accent: str = PRIMARY,
    title_style: str = "tag",
) -> None:
    """A generic bordered surface with an optional title bar."""
    if title_style == "bar":
        title_html = f"""
        <div class="surface-box-bar">
            <div class="surface-box-bar-title">{_esc(title)}</div>
        </div>
        """
    elif title_style == "tag":
        title_html = f'<div class="surface-box-tag">{_esc(title)}</div>'
    else:
        title_html = f'<div class="surface-box-plain">{_esc(title)}</div>'

    body_html = f'<div class="surface-box-body">{_esc(body)}</div>' if body else ""
    st.markdown(
        f"""
        <div class="surface-box" style="--accent:{accent};">
            {title_html}
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
