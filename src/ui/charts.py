"""Reusable Plotly chart helpers with the dashboard's visual theme.

All interactive charts share a restrained look with consistent fonts,
subtle gridlines, horizontal legends, and meaningful hover information. Colour
is never the only channel of communication — legend and hover labels always name
the category explicitly.
"""

from __future__ import annotations

from typing import Optional

import plotly.express as px
import plotly.graph_objects as go

from .styles import (
    CHART_FONT,
    COLORWAY,
    PALETTE,
    get_theme,
)


# ---------------------------------------------------------------------------
# Theme application
# ---------------------------------------------------------------------------

def get_plotly_theme(theme: Optional[dict] = None) -> dict:
    """Return a Plotly ``layout`` update dict driven by the app's theme tokens.

    Charts consume the *same* semantic tokens as the rest of the UI so the
    plot background, text, grid, legend and hover labels stay coherent with the
    rest of the dashboard in both light and dark mode.
    """
    t = theme if isinstance(theme, dict) else get_theme(theme)
    axis = dict(
        gridcolor=t["border"],
        zerolinecolor=t["border"],
        linecolor=t["border"],
        mirror=False,
        title_font=dict(size=12, color=t["text_primary"]),
        tickfont=dict(size=11, color=t["text_primary"]),
    )
    return {
        "template": "plotly_white",
        "font": dict(family=CHART_FONT, size=13, color=t["text_primary"]),
        "paper_bgcolor": t["background"],
        "plot_bgcolor": t["surface"],
        "colorway": COLORWAY,
        "hoverlabel": dict(
            font_size=12,
            font_family=CHART_FONT,
            bgcolor=t["surface_elevated"],
            font_color=t["text_primary"],
        ),
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=t["text_secondary"], family=CHART_FONT, size=12),
        ),
        "xaxis": dict(axis),
        "yaxis": dict(axis),
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
    fig = px.scatter(
        df, x=x, y=y, color=color,
        hover_data=hover_data or [],
        title=title,
        color_discrete_map=color_map,
    )
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
    fig.update_traces(
        marker=dict(size=9, opacity=0.82, line=dict(width=0.5, color="white")),
    )
    if centers_x is not None and centers_y is not None:
        t = get_theme()
        fig.add_trace(
            go.Scatter(
                x=centers_x, y=centers_y, mode="markers",
                marker=dict(
                    symbol="diamond", size=16, color=t["text_primary"],
                    line=dict(width=2, color="white"),
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
    )
    fig.update_traces(
        hovertemplate=f"{x}: %{{x}}<br>Customers: %{{y}}<extra></extra>",
        marker_line_color="white", marker_line_width=1,
    )
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
        color_discrete_sequence=PALETTE if color is None else None,
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
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k_values, y=inertias, mode="lines+markers",
            name="Inertia (WCSS)", line=dict(color="#2563eb"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=k_values, y=silhouettes, mode="lines+markers",
            name="Silhouette Score", yaxis="y2", line=dict(color="#16a34a"),
        )
    )
    fig.add_vline(
        x=optimal_k, line_dash="dot",
        line_color="#94a3b8",
        annotation_text=f"optimal k = {optimal_k}",
        annotation_position="top",
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
    fig = go.Figure()
    order = sorted(range(len(df)), key=lambda i: df[metric].iloc[i], reverse=not ascending)
    fig.add_trace(
        go.Bar(
            x=[df[metric].iloc[i] for i in order],
            y=[df[name].iloc[i] for i in order],
            orientation="h",
            marker_color=["#2563eb" if i == order[0] else "#cbd5e1" for i in order],
            text=[f"{df[metric].iloc[i]:.3f}" for i in order],
            textposition="outside",
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
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", title=title,
        labels=labels or {},
    )
    fig.update_traces(
        hovertemplate="%{y} vs %{x}<br>Correlation: %{z}<extra></extra>"
    )
    fig.update_coloraxes(colorbar_title="Correlation")
    return style_chart(fig, title=title, height=height)
