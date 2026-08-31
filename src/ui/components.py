"""Layout and state components: headers, status indicators, recommendation boxes
and empty / error / loading states.

These keep the presentation of structural elements consistent across tabs.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from .styles import ERROR, PRIMARY, SUCCESS


def _esc(text: str) -> str:
    """Minimal HTML-escape for user-provided strings."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

def section_header(title: str, caption: str = "") -> None:
    """Render a consistent section title with an optional supporting caption."""
    st.markdown(f"<h2>{_esc(title)}</h2>", unsafe_allow_html=True)
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
    """Render a compact, premium top header."""
    cfg_block = (
        f'<span>Configuration: <b>{_esc(active_label)}</b></span>'
        if active_label else ""
    )
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-brand">
                <div class="app-header-appname">{_esc(app_name)}</div>
                <div class="app-header-tagline">{_esc(tagline)}</div>
            </div>
            <div class="app-header-meta">
                <span>Model: <b>{_esc(algorithm)}</b></span>
                <span>Segments: <b>{n_segments}</b></span>
                <span>Customers: <b>{n_customers:,}</b></span>
                {cfg_block}
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
        <div class="sidebar-section">
            <div class="sidebar-section-title">
                <span class="status-dot" style="background:{dot};"></span>
                Dataset
            </div>
            <div class="dataset-meta">{raw_rows:,} customers · {raw_cols} attributes</div>
            <div class="dataset-status">{status}{' · ' + _esc(version) if version else ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_dot(label: str, ok: bool = True, color: Optional[str] = None) -> None:
    """Render a small coloured status dot with a label."""
    bg = color or (SUCCESS if ok else ERROR)
    st.markdown(
        f"""<span class="status-dot" style="background:{bg};"></span>
           <span class="status-label">{_esc(label)}</span>""",
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
            f'<li class="recommendation-item">{_esc(it)}</li>' for it in items
        )
        item_html = f'<ul class="recommendation-list">{bullets}</ul>'
    st.markdown(
        f"""
        <div class="recommendation-box" style="--accent:{accent};">
            <div class="recommendation-title">{_esc(title)}</div>
            <div class="recommendation-body">{_esc(body)}</div>
            {item_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_box(text: str, kind: str = "info") -> None:
    """A soft, themed information callout."""
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
    """Render a centred empty state with optional action button."""
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.markdown(
            f"""
            <div class="empty-state">
                <div class="empty-state-icon">○</div>
                <div class="empty-state-message">{_esc(message)}</div>
                {f'<div class="empty-state-hint">{_esc(hint)}</div>' if hint else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if action:
            st.button(action[0])
