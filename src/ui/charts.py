"""Reusable Plotly chart helpers with the dashboard's visual theme.

All interactive charts share a restrained look with consistent fonts,
subtle gridlines, horizontal legends, and meaningful hover information. Colour
is never the only channel of communication — legend and hover labels always name
the category explicitly.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .styles import (
    CHART_FONT,
    chart_palette,
    get_theme,
)


def _to_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Theme application
# ---------------------------------------------------------------------------

def get_plotly_theme(theme: Optional[dict] = None) -> dict:
    """Return a Plotly ``layout`` update dict driven by the app's theme tokens.

    Charts consume the *same* semantic tokens as the rest of the UI so the
    plot background, text, grid, legend and hover labels stay coherent with
    the rest of the dashboard in both light and dark mode.

    Every chart in the dashboard is styled through this single function so
    background, text, grid, axis, legend, hover and the categorical
    colourway are always consistent. Light and dark themes both use the
    ``plotly`` template (which is theme-agnostic) and rely on the explicit
    tokens below for colour; this avoids any template-driven defaults that
    would mismatch the surrounding UI.
    """
    t = theme if isinstance(theme, dict) else get_theme(theme)
    is_dark = t["name"] == "dark"
    axis = dict(
        gridcolor=t["chart_grid"],
        zerolinecolor=t["chart_axis"],
        linecolor=t["chart_axis"],
        mirror=False,
        title_font=dict(size=12, color=t["text_primary"]),
        tickfont=dict(size=11, color=t["text_primary"]),
        showline=True,
        ticks="outside",
        tickcolor=t["chart_axis"],
        ticklen=4,
    )
    return {
        "template": "plotly",
        "font": dict(family=CHART_FONT, size=13, color=t["text_primary"]),
        "paper_bgcolor": t["chart_paper"],
        "plot_bgcolor": t["chart_plot"],
        "colorway": chart_palette(t["name"]),
        "hoverlabel": dict(
            font_size=12,
            font_family=CHART_FONT,
            bgcolor=t["surface_elevated"],
            font_color=t["text_primary"],
            bordercolor=t["border_strong"],
        ),
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=t["text_secondary"], family=CHART_FONT, size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        "xaxis": dict(axis),
        "yaxis": dict(axis),
        # `modebar` (zoom/pan toolbar) — dark mode keeps the icons light.
        "modebar": dict(
            bgcolor="rgba(0,0,0,0)",
            color=t["text_secondary"] if is_dark else t["text_muted"],
        ),
    }


def style_chart(
    fig: go.Figure,
    title: Optional[str] = None,
    height: int = 360,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> go.Figure:
    """Apply the shared analytics theme to a Plotly figure."""
    t = get_theme()
    fig.update_layout(**get_plotly_theme(t))
    fig.update_layout(
        height=height,
        margin=dict(t=54, b=44, l=54, r=24),
    )
    if title is not None:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left"))
    if xlabel is not None:
        fig.update_xaxes(title=xlabel)
    if ylabel is not None:
        fig.update_yaxes(title=ylabel)
    return fig


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def scatter_segments(
    df: "object",
    x: str,
    y: str,
    color: str,
    title: str,
    color_map: Optional[dict] = None,
    hover_data: Optional[list] = None,
    hover_labels: Optional[list] = None,
    hovertemplate: Optional[str] = None,
    centers_x: Optional["object"] = None,
    centers_y: Optional["object"] = None,
    center_label: str = "Segment centre",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    height: int = 420,
) -> go.Figure:
    """Interactive scatter of customers coloured by segment/persona."""
    # px.scatter treats an integer ``color`` column as a *numeric* variable
    # and ignores ``color_discrete_map`` — that collapses every segment into
    # a single continuous colorscale instead of producing one trace per
    # category. Coerce the column to string so px emits one trace per
    # distinct category and ``color_discrete_map`` is honoured.
    df = df.copy()
    color_series = df[color]
    if not pd.api.types.is_string_dtype(color_series):
        df[color] = color_series.astype(str)
    fig = px.scatter(
        df, x=x, y=y, color=color,
        hover_data=hover_data or [],
        title=title,
        color_discrete_map=color_map,
    )
    if color_map:
        # px.scatter with `color=` sets `marker.color` to integer category
        # indices into the layout colorway. To guarantee that each point is
        # filled with the colour the caller actually asked for (and so the
        # chart stays readable regardless of which theme is active), resolve
        # each trace's colour directly from the discrete map. px.scatter
        # emits one trace per category, so the per-trace index in
        # ``fig.data`` matches the order of first appearance in the column.
        palette = chart_palette()
        cats: list[str] = []
        for v in df[color].astype(str).tolist():
            if v not in cats:
                cats.append(v)
        for i, trace in enumerate(fig.data):
            key = cats[i] if i < len(cats) else None
            seg_color = (
                (color_map.get(key) if key is not None else None)
                or (color_map.get(_to_int(key)) if key is not None else None)
                or palette[i % len(palette)]
            )
            trace.update(marker=dict(color=seg_color))
    if hovertemplate is not None:
        fig.update_traces(hovertemplate=hovertemplate)
    elif hover_labels:
        parts = [f"{lab}: %{{customdata[{i}]}}" for i, lab in enumerate(hover_labels)]
        extra = "<br>".join(parts)
        fig.update_traces(
            hovertemplate=(
                f"{x}: %{{x}}<br>{y}: %{{y}}<br>{extra}<extra></extra>"
            ),
        )
    else:
        fig.update_traces(hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>")
    t = get_theme()
    fig.update_traces(
        marker=dict(
            size=9,
            opacity=0.9 if t["name"] == "dark" else 0.82,
            line=dict(width=0.5, color=t["chart_marker_outline"]),
        ),
    )
    if centers_x is not None and centers_y is not None:
        fig.add_trace(
            go.Scatter(
                x=centers_x, y=centers_y, mode="markers",
                marker=dict(
                    symbol="diamond", size=16, color=t["text_primary"],
                    line=dict(width=2, color=t["text_primary"]),
                ),
                name=center_label,
                hovertemplate=(
                    f"{center_label}<br>Income: %{{x}} k$<br>"
                    f"Spending: %{{y}}<extra></extra>"
                ),
            )
        )
    return style_chart(fig, title=title, height=height, xlabel=xlabel, ylabel=ylabel)


def bar_segments(
    counts_df: "object",
    x: str,
    y: str,
    color_map: Optional[dict] = None,
    title: str = "",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    height: int = 320,
) -> go.Figure:
    """Vertical bar chart of segment sizes."""
    fig = px.bar(
        counts_df, x=x, y=y, title=title,
        color=x, color_discrete_map=color_map,
        color_discrete_sequence=chart_palette(),
    )
    t = get_theme()
    fig.update_traces(
        hovertemplate=f"{x}: %{{x}}<br>Customers: %{{y}}<extra></extra>",
        marker_line_color=t["chart_marker_outline"], marker_line_width=1,
    )
    # px.bar with `color=` produces a categorical colorway that defaults to
    # integer indices when the discrete_map keys are integers (Plotly matches
    # map keys as strings). Override with explicit per-trace colours so every
    # bar is filled with the intended hex value regardless of theme.
    palette = chart_palette()
    if color_map:
        for i, trace in enumerate(fig.data):
            seg = trace.x[0] if len(trace.x) else None
            seg_color = (
                (color_map.get(str(seg)) if seg is not None else None)
                or (color_map.get(_to_int(seg)) if seg is not None else None)
                or palette[i % len(palette)]
            )
            trace.update(marker=dict(color=seg_color))
    else:
        # No explicit map — fill each per-segment trace with the theme palette.
        for i, trace in enumerate(fig.data):
            trace.update(marker=dict(color=palette[i % len(palette)]))
    fig.update_layout(showlegend=False)
    return style_chart(fig, title=title, height=height, xlabel=xlabel, ylabel=ylabel)


def histogram_chart(
    df: "object",
    x: str,
    title: str,
    color: Optional[str] = None,
    nbins: int = 20,
    xlabel: Optional[str] = None,
    ylabel: str = "Customers",
    height: int = 320,
) -> go.Figure:
    """A single-distribution histogram."""
    fig = px.histogram(
        df, x=x, nbins=nbins,
        title=title, color=color,
        color_discrete_sequence=chart_palette() if color is None else None,
    )
    fig.update_traces(hovertemplate=f"{x}: %{{x}}<br>Customers: %{{y}}<extra></extra>")
    return style_chart(fig, title=title, height=height, xlabel=xlabel, ylabel=ylabel)


def elbow_silhouette_chart(
    k_values: list[int],
    inertias: list[float],
    silhouettes: list[float],
    optimal_k: int,
    title: str = "Elbow & Silhouette vs k",
    height: int = 420,
) -> go.Figure:
    """Dual-axis K selection chart: inertia (left) + silhouette (right)."""
    palette = chart_palette()
    t = get_theme()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k_values, y=inertias, mode="lines+markers",
            name="Inertia (WCSS)",
            line=dict(color=palette[0], width=2.5),
            marker=dict(size=8, color=palette[0], line=dict(width=0)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=k_values, y=silhouettes, mode="lines+markers",
            name="Silhouette Score", yaxis="y2",
            line=dict(color=palette[1], width=2.5),
            marker=dict(size=8, color=palette[1], line=dict(width=0)),
        )
    )
    fig.add_vline(
        x=optimal_k, line_dash="dot",
        line_color=t["text_secondary"],
        annotation_text=f"optimal k = {optimal_k}",
        annotation_position="top",
        annotation_font_color=t["text_primary"],
    )
    fig = style_chart(fig, title=title, height=height)
    fig.update_layout(
        xaxis=dict(
            title="Number of segments (k)",
            tickmode="array", tickvals=k_values,
        ),
        yaxis=dict(title="Inertia"),
        yaxis2=dict(
            title="Silhouette", overlaying="y", side="right", showgrid=False,
        ),
    )
    return fig


def metric_bar(
    df: "object",
    metric: str,
    name: str,
    ascending: bool = False,
    title: str = "",
    height: int = 360,
) -> go.Figure:
    """Horizontal bar chart comparing a metric across model configurations."""
    palette = chart_palette()
    t = get_theme()
    fig = go.Figure()
    order = sorted(range(len(df)), key=lambda i: df[metric].iloc[i], reverse=not ascending)
    muted = t["text_secondary"] if t["name"] == "dark" else "#cbd5e1"
    fig.add_trace(
        go.Bar(
            x=[df[metric].iloc[i] for i in order],
            y=[df[name].iloc[i] for i in order],
            orientation="h",
            marker_color=[
                palette[0] if i == order[0] else muted for i in order
            ],
            text=[f"{df[metric].iloc[i]:.3f}" for i in order],
            textposition="outside",
            textfont=dict(color=t["text_primary"]),
            hovertemplate=f"%{{y}}<br>{metric}: %{{x}}<extra></extra>",
        )
    )
    fig = style_chart(fig, title=title, height=height, xlabel=metric, ylabel=name)
    return fig


def correlation_heatmap(
    corr: "object",
    labels: Optional[list[str]] = None,
    title: str = "Feature Correlation",
    height: int = 380,
) -> go.Figure:
    """A restrained correlation matrix heatmap."""
    t = get_theme()
    # Use a colour scale whose midpoint matches the chart text colour so the
    # heatmap text stays readable, with strong ends that work in both themes.
    fig = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r", title=title,
        labels=labels or {}, zmin=-1, zmax=1,
    )
    fig.update_traces(
        hovertemplate="%{y} vs %{x}<br>Correlation: %{z}<extra></extra>",
        textfont=dict(color=t["text_primary"], size=12),
    )
    fig.update_coloraxes(colorbar_title="Correlation")
    fig.update_layout(
        coloraxis_colorbar=dict(
            tickfont=dict(color=t["text_primary"]),
            title=dict(font=dict(color=t["text_primary"])),
        ),
    )
    return style_chart(fig, title=title, height=height)
