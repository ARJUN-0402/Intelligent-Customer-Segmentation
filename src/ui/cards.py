"""Reusable card components: KPI, metric, persona and insight cards.

Each card is a self-contained HTML snippet with a consistent look-and-feel so
that ``app.py`` never re-defines card styling. Cards respect the persona colour
system and never rely on colour alone to convey meaning (labels are textual).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .styles import (
    BORDER, CARD_RADIUS, CARD_SHADOW, INK, MUTED, PAPER,
    PRIMARY, TEXT_METRIC_LARGE, TEXT_METRIC_MED,
    TEXT_BODY, TEXT_CAPTION, TEXT_SUBHEAD,
)

def _container(
    html: str,
    border_color: str = BORDER,
    bg: str = PAPER,
    pad: str = "16px 18px",
) -> str:
    """Wrap *html* in a standard surface container."""
    return f"""
    <div style="background:{bg};border:1px solid {BORDER};
        border-left:4px solid {border_color};border-radius:{CARD_RADIUS};
        padding:{pad};height:100%;box-shadow:{CARD_SHADOW};">
        {html}
    </div>
    """


def kpi_card(
    label: str,
    value: str,
    sub: str = "",
    color: str = PRIMARY,
    value_size: str = TEXT_METRIC_LARGE,
) -> None:
    """Render a single KPI card.

    Parameters
    ----------
    label:
        Short, uppercase metric name shown above the value.
    value:
        The headline metric (already formatted as a string).
    sub:
        Optional supporting context line beneath the value.
    color:
        Accent colour for the left border.
    value_size:
        CSS font-size for the value.
    """
    html = f"""
        <div style="font-size:{TEXT_CAPTION};color:{MUTED};font-weight:600;
            text-transform:uppercase;letter-spacing:0.05em;">
            {label}</div>
        <div style="font-size:{value_size};color:{INK};font-weight:700;
            margin-top:8px;line-height:1.1;">{value}</div>
        <div style="font-size:{TEXT_CAPTION};color:{MUTED};margin-top:6px;">
            {sub}</div>
    """
    st.markdown(_container(html, border_color=color), unsafe_allow_html=True)


def metric_card(
    label: str,
    value: str,
    direction: str = "",
    note: str = "",
    color: str = PRIMARY,
) -> None:
    """A compact metric card used in the Model Evaluation grid.

    Parameters
    ----------
    label:
        Metric name (e.g. "Silhouette Score").
    value:
        Formatted value string.
    direction:
        "Higher is better" / "Lower is better" — shown as a subtitle tag.
    note:
        Optional note shown beneath the value (e.g. "K-Means only").
    color:
        Accent border colour.
    """
    direction_html = ""
    if direction:
        direction_html = (
            f'<div style="font-size:{TEXT_CAPTION};color:{MUTED};'
            f'margin-top:4px;">{direction}</div>'
        )
    note_html = ""
    if note:
        note_html = (
            f'<div style="font-size:{TEXT_CAPTION};color:{MUTED};'
            f'margin-top:2px;">{note}</div>'
        )
    html = f"""
        <div style="font-size:{TEXT_CAPTION};color:{MUTED};font-weight:600;
            text-transform:uppercase;letter-spacing:0.05em;">{label}</div>
        <div style="font-size:{TEXT_METRIC_MED};color:{INK};font-weight:700;
            margin-top:6px;">{value}</div>
        {direction_html}
        {note_html}
    """
    st.markdown(_container(html, border_color=color), unsafe_allow_html=True)


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
    """Render a persona overview card used in the Segment Explorer.

    Parameters
    ----------
    persona_name:
        Display name of the persona.
    persona_key:
        Machine key (used to add a data attribute for testing).
    segment_id:
        Numeric cluster id.
    color:
        Persona identity colour.
    is_selected:
        Whether this card represents the currently expanded segment.
    """
    border = color if is_selected else BORDER
    weight = "700" if is_selected else "600"
    shadow = (
        "0 4px 12px rgba(15, 23, 42, 0.12)" if is_selected
        else "0 1px 2px rgba(15, 23, 42, 0.04)"
    )
    html = f"""
        <div data-persona="{persona_key}" style="border-left:4px solid {border};
            border-radius:{CARD_RADIUS};padding:16px;height:100%;
            background:{PAPER};box-shadow:{shadow};cursor:pointer;
            transition:box-shadow 0.15s ease,border-color 0.15s ease;">
            <div style="display:flex;justify-content:space-between;
                align-items:flex-start;">
                <div>
                    <div style="font-size:{TEXT_CAPTION};font-weight:700;
                        color:{MUTED};text-transform:uppercase;
                        letter-spacing:0.05em;">Segment {segment_id}</div>
                    <div style="font-size:1.05rem;font-weight:{weight};
                        color:{INK};margin-top:4px;">{persona_name}</div>
                </div>
                <div style="width:12px;height:12px;border-radius:50%;
                    background:{color};"></div>
            </div>
            <div style="display:flex;gap:16px;margin-top:12px;">
                <div>
                    <div style="font-size:{TEXT_METRIC_LARGE};font-weight:700;
                        color:{INK};">{customer_count:,}</div>
                    <div style="font-size:{TEXT_CAPTION};color:{MUTED};">
                        customers</div>
                </div>
                <div>
                    <div style="font-size:{TEXT_METRIC_LARGE};font-weight:700;
                        color:{color};">{percentage:.1f}%</div>
                    <div style="font-size:{TEXT_CAPTION};color:{MUTED};">
                        of base</div>
                </div>
            </div>
        </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def insight_card(title: str, detail: str, color: str = PRIMARY) -> None:
    """A data-driven insight card with a coloured accent bar."""
    html = f"""
        <div style="background:{PAPER};border:1px solid {BORDER};
            border-left:4px solid {color};border-radius:{CARD_RADIUS};
            padding:16px 18px;box-shadow:{CARD_SHADOW};height:100%;">
            <div style="font-size:{TEXT_CAPTION};font-weight:600;
                color:{color};text-transform:uppercase;
                letter-spacing:0.05em;">{title}</div>
            <div style="font-size:{TEXT_BODY};color:{INK};margin-top:8px;
                line-height:1.45;">{detail}</div>
        </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def styled_metric_table(df: pd.DataFrame) -> None:
    """Render a dataframe with a light, borderless presentation."""
    st.dataframe(df, width="stretch", hide_index=True)


def surface_box(
    title: str,
    body: str = "",
    accent: str = PRIMARY,
    title_style: str = "tag",
) -> None:
    """A generic bordered surface with an optional title bar.

    ``title_style`` controls the title rendering:
    - ``"tag"`` — small uppercase accent tag above the body.
    - ``"bar"``   — a coloured bar to the left of a bold heading.
    - ``"plain"`` — plain bold heading, no accent.
    """
    if title_style == "bar":
        html = f"""
        <div style="border-left:4px solid {accent};padding:4px 0 4px 14px;
            margin-bottom:10px;">
            <div style="font-size:1.25rem;font-weight:700;color:{INK};">
                {title}</div>
        </div>
        """
    elif title_style == "tag":
        html = f"""
        <div style="font-size:{TEXT_CAPTION};font-weight:700;color:{MUTED};
            text-transform:uppercase;letter-spacing:0.05em;">{title}</div>
        """
    else:
        html = f"""
        <div style="font-size:{TEXT_SUBHEAD};font-weight:700;color:{INK};
            margin-bottom:6px;">{title}</div>
        """
    if body:
        html += f'<div style="font-size:{TEXT_BODY};color:{INK};line-height:1.5;">{body}</div>'
    st.markdown(_container(html, border_color=accent), unsafe_allow_html=True)
