"""Layout and state components: headers, status indicators, recommendation boxes
and empty / error / loading states.

These keep the presentation of structural elements consistent across tabs.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from .styles import (
    BORDER, CARD_RADIUS, CARD_SHADOW, ERROR, INK, MUTED, PRIMARY,
    SURFACE, SUCCESS, TEXT_CAPTION, TEXT_BODY,
)


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------
def section_header(title: str, caption: str = "") -> None:
    """Render a consistent section title with an optional supporting caption."""
    st.markdown(
        f"<h2 style='color:{INK};margin-bottom:2px;'>{title}</h2>",
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def app_header(
    app_name: str,
    tagline: str,
    algorithm: str,
    n_segments: int,
    n_customers: int,
    active_label: str = "",
) -> None:
    """Render a compact, premium top header.

    Communicates the product name, tagline, and the currently active model /
    segment / customer summary in a single condensed bar.
    """
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;
            align-items:center;flex-wrap:wrap;margin-bottom:0.5rem;">
            <div>
                <div style="font-size:0.75rem;font-weight:700;
                    color:{MUTED};text-transform:uppercase;
                    letter-spacing:0.08em;">{app_name}</div>
                <div style="font-size:1rem;font-weight:600;color:{INK};">
                    {tagline}</div>
            </div>
            <div style="display:flex;gap:14px;flex-wrap:wrap;">
                <span style="font-size:{TEXT_CAPTION};color:{MUTED};">
                    Model: <b style="color:{INK}">{algorithm}</b>
                </span>
                <span style="font-size:{TEXT_CAPTION};color:{MUTED};">
                    Segments: <b style="color:{INK}">{n_segments}</b>
                </span>
                <span style="font-size:{TEXT_CAPTION};color:{MUTED};">
                    Customers: <b style="color:{INK}">{n_customers}</b>
                </span>
                {f'<span style="font-size:{TEXT_CAPTION};color:{MUTED}">Configuration: <b style="color:{INK}">{active_label}</b></span>' if active_label else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar building blocks
# ---------------------------------------------------------------------------
def sidebar_nav(
    options: list[str],
    labels: list[str],
    index: int = 0,
    help_text: str = "",
) -> str:
    """Render a numbered navigation radio in the sidebar and return the choice."""
    display = {o: lbl for o, lbl in zip(options, labels)}
    if help_text:
        st.sidebar.caption(help_text)
    return st.sidebar.radio(
        "Navigate",
        options=options,
        format_func=lambda o: display[o],
        index=index,
        label_visibility="collapsed",
    )


def dataset_status(raw_rows: int, raw_cols: int, ok: bool, version: str = "") -> None:
    """Render a subtle dataset-status summary in the sidebar."""
    status = "Ready" if ok else "Unavailable"
    dot = SUCCESS if ok else ERROR
    st.sidebar.markdown(
        f"""
        <div style="font-size:{TEXT_CAPTION};color:{MUTED};margin-top:8px;">
            <span style="display:inline-block;width:8px;height:8px;
                border-radius:50%;background:{dot};margin-right:6px;"></span>
            Dataset
        </div>
        <div style="font-size:{TEXT_BODY};font-weight:600;color:{INK};
            margin-left:16px;">{raw_rows} customers · {raw_cols} attributes</div>
        <div style="font-size:{TEXT_CAPTION};color:{MUTED};margin-left:16px;
            margin-top:2px;">{status}{' · ' + version if version else ''}</div>
        """,
        unsafe_allow_html=True,
    )


def status_dot(label: str, ok: bool = True, color: Optional[str] = None) -> None:
    """Render a small coloured status dot with a label."""
    bg = color or (SUCCESS if ok else ERROR)
    st.markdown(
        f"""<span style="display:inline-flex;align-items:center;gap:6px;
            font-size:{TEXT_CAPTION};color:{MUTED}">
            <span style="width:10px;height:10px;border-radius:50%;
                background:{bg};display:inline-block;"></span>
            {label}</span>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Recommendation / information surfaces
# ---------------------------------------------------------------------------
def recommendation_box(
    title: str,
    body: str,
    items: Optional[list[str]] = None,
    accent: str = PRIMARY,
) -> None:
    """A prominent but restrained recommendation/insight box."""
    item_html = ""
    if items:
        bullets = "".join(
            f'<li style="margin-bottom:4px;font-size:{TEXT_BODY};'
            f'color:{INK};">{it}</li>' for it in items
        )
        item_html = f'<ul style="margin:8px 0 0 18px;padding:0;">{bullets}</ul>'
    st.markdown(
        f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};
            border-left:4px solid {accent};border-radius:{CARD_RADIUS};
            padding:16px 18px;box-shadow:{CARD_SHADOW};margin:12px 0;">
            <div style="font-size:{TEXT_CAPTION};font-weight:700;
                color:{accent};text-transform:uppercase;
                letter-spacing:0.05em;">{title}</div>
            <div style="font-size:{TEXT_BODY};color:{INK};
                line-height:1.5;margin-top:8px;">{body}</div>
            {item_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_box(text: str, kind: str = "info") -> None:
    """A soft, themed information callout.

    ``kind`` is one of ``"info"`` (blue), ``"success"`` (green), ``"warning"``
    (orange). Uses Streamlit-native styling for consistency.
    """
    if kind == "success":
        st.success(text)
    elif kind == "warning":
        st.warning(text)
    else:
        st.info(text)


# ---------------------------------------------------------------------------
# Empty / error / loading states
# ---------------------------------------------------------------------------
def error_state(message: str, detail: str = "") -> None:
    """Show a clean error without exposing tracebacks."""
    st.error(message)
    if detail:
        st.caption(detail)


def empty_state(
    message: str,
    hint: str = "",
    action: Optional[tuple[str, str]] = None,
) -> None:
    """Render a centred empty state with optional action button.

    Parameters
    ----------
    message:
        Short explanation of why the view is empty.
    hint:
        Optional secondary line.
    action:
        Optional ``(label, emoji)`` tuple for a contextual call-to-action.
    """
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.markdown(
            f"""
            <div style="text-align:center;padding:24px 0;color:{MUTED};
                font-size:{TEXT_BODY};">
                <div style="font-size:1.5rem;margin-bottom:8px;">○</div>
                <div>{message}</div>
                {f'<div style="margin-top:6px;font-size:{TEXT_CAPTION};">{hint}</div>' if hint else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if action:
            st.button(action[0])
