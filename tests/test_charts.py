"""Visibility regression tests for every chart in the dashboard.

These tests do NOT only assert that a chart function returns a Plotly
``Figure`` — they inspect the figure contents (trace data, marker opacity,
bar fill, layout tokens) in both light and dark mode so a regression that
hides the data against the plot background is caught immediately.

The customer counts and cluster assignments come from the real dataset
configuration so the tests reflect what users see, not synthetic stand-ins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.clustering import run_clustering
from src.preprocessing import CustomerDataPreprocessor
from src.ui import charts
from src.ui.styles import (
    PALETTE_DARK,
    PALETTE_LIGHT,
    chart_palette,
    get_theme,
    persona_colors,
)
from src.utils import (
    AGE,
    ANNUAL_INCOME,
    FEATURE_COLUMNS,
    GENRE,
    SPENDING_SCORE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scaled_data(df: pd.DataFrame):
    pre = CustomerDataPreprocessor(
        numeric_features=FEATURE_COLUMNS, categorical_features=[]
    )
    X, _cleaned = pre.fit_transform(df)
    return X, pre, df


@pytest.fixture(scope="module")
def segmentation(scaled_data):
    X, _pre, df = scaled_data
    result = run_clustering(X, algorithm="kmeans", n_clusters=5, random_state=42)
    plot_df = df[[AGE, ANNUAL_INCOME, SPENDING_SCORE, GENRE]].copy()
    plot_df["Segment"] = result.labels
    counts = pd.Series(result.labels).value_counts().sort_index()
    size_df = pd.DataFrame(
        {"Segment": counts.index.astype(int), "Customers": counts.values.astype(int)}
    )
    return plot_df, size_df


def _marker_dict(marker) -> dict:
    """Return a plain dict view of a Plotly Marker (which is not a dict)."""
    if marker is None:
        return {}
    if isinstance(marker, dict):
        return marker
    # plotly.graph_objs.Marker exposes its properties as attributes.
    return {k: getattr(marker, k) for k in ("size", "opacity", "color", "line")
            if getattr(marker, k, None) is not None}


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance for a single sRGB hex colour."""
    r, g, b = _hex_to_rgb01(hex_color)
    def _adj(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * _adj(r) + 0.7152 * _adj(g) + 0.0722 * _adj(b)


def _contrast_ratio(c1: str, c2: str) -> float:
    l1 = _relative_luminance(c1)
    l2 = _relative_luminance(c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


# ---------------------------------------------------------------------------
# Theme contract — every chart palette colour must be visible against its
# theme's plot background.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_chart_palette_is_theme_aware(theme_name: str):
    """The chart palette must switch between light and dark themes."""
    palette = chart_palette(theme_name)
    assert isinstance(palette, list) and len(palette) >= 6
    expected = PALETTE_DARK if theme_name == "dark" else PALETTE_LIGHT
    assert palette == expected


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_chart_palette_contrast(theme_name: str):
    """Every palette colour must have WCAG contrast >= 3.0 vs plot_bg."""
    t = get_theme(theme_name)
    plot_bg = t["chart_plot"]
    for color in chart_palette(theme_name):
        assert _contrast_ratio(color, plot_bg) >= 3.0, (
            f"{theme_name} palette colour {color} has insufficient contrast "
            f"vs {plot_bg}"
        )


def test_dark_persona_colors_contrast_against_dark_plot():
    """Dark-mode persona identities must be readable against #111827."""
    plot_bg = get_theme("dark")["chart_plot"]
    for key, color in persona_colors("dark").items():
        assert _contrast_ratio(color, plot_bg) >= 3.0, (
            f"dark persona {key}={color} too dark vs {plot_bg}"
        )


# ---------------------------------------------------------------------------
# Plotly layout tokens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_get_plotly_theme_layout_tokens(theme_name: str):
    layout = charts.get_plotly_theme(theme_name)
    t = get_theme(theme_name)
    assert layout["paper_bgcolor"] == t["chart_paper"]
    assert layout["plot_bgcolor"] == t["chart_plot"]
    assert layout["font"]["color"] == t["text_primary"]
    assert layout["colorway"] == chart_palette(theme_name)
    # Axes must use the chart_grid / chart_axis tokens, NOT the page border.
    for axis in ("xaxis", "yaxis"):
        assert layout[axis]["gridcolor"] == t["chart_grid"]
        assert layout[axis]["linecolor"] == t["chart_axis"]
        assert layout[axis]["tickfont"]["color"] == t["text_primary"]
    # Hover label must be readable on the elevated surface.
    assert layout["hoverlabel"]["font_color"] == t["text_primary"]
    assert layout["hoverlabel"]["bgcolor"] == t["surface_elevated"]


# ---------------------------------------------------------------------------
# Segment Distribution chart
# ---------------------------------------------------------------------------

def _build_segment_distribution(size_df: pd.DataFrame, theme_name: str) -> go.Figure:
    palette = chart_palette(theme_name)
    color_map = {int(cid): palette[int(cid) % len(palette)] for cid in size_df["Segment"]}
    fig = charts.bar_segments(
        size_df, "Segment", "Customers",
        color_map=color_map,
        xlabel="Segment", ylabel="Customers", height=380,
    )
    # Force the theme — style_chart reads the active Streamlit theme, which
    # during unit tests is unknown.
    fig.update_layout(**charts.get_plotly_theme(theme_name))
    return fig


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_segment_distribution_has_data_and_visible_bars(
    segmentation, theme_name: str
):
    plot_df, size_df = segmentation
    assert not size_df.empty
    assert (size_df["Customers"] > 0).all()

    fig = _build_segment_distribution(size_df, theme_name)

    # At least one bar trace.
    bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
    assert bar_traces, "Segment Distribution has no Bar trace"

    # Real customer counts (not zero, not synthetic).
    total = sum(int(sum(t.y)) for t in bar_traces)
    assert total == int(size_df["Customers"].sum())

    # Visible fill — Plotly defaults opacity to 1.0; ensure it was not
    # accidentally lowered. opacity may be on each trace or in marker config.
    for t in bar_traces:
        marker = _marker_dict(t.marker)
        opacity = marker.get("opacity", 1.0)
        assert opacity is None or opacity >= 0.5, f"bar opacity too low: {opacity}"

        # Bar colours are defined (a real palette colour, not transparent).
        colors = marker.get("color")
        assert colors is not None, "bars have no colour"
        if isinstance(colors, list):
            for c in colors:
                if isinstance(c, str) and c.startswith("rgba"):
                    assert not c.endswith(",0)") and not c.endswith(", 0)"), (
                        f"bar colour is transparent: {c}"
                    )
        elif isinstance(colors, str) and colors.startswith("rgba"):
            assert not colors.endswith(",0)") and not colors.endswith(", 0)"), (
                f"bar colour is transparent: {colors}"
            )

    # Layout uses the theme tokens.
    t = get_theme(theme_name)
    assert fig.layout.plot_bgcolor == t["chart_plot"]
    assert fig.layout.paper_bgcolor == t["chart_paper"]
    # Every bar colour has enough contrast against the plot background.
    plot_bg = t["chart_plot"]
    for trace in bar_traces:
        for c in (
            trace.marker.color
            if isinstance(trace.marker.color, list)
            else [trace.marker.color]
        ):
            if isinstance(c, str) and c.startswith("#"):
                assert _contrast_ratio(c, plot_bg) >= 3.0, (
                    f"bar {c} has insufficient contrast vs {plot_bg}"
                )


def test_segment_distribution_bar_colors_are_hex_strings(segmentation):
    """Regression: Plotly px.bar with integer x + color= previously emitted
    integer indices instead of hex colours, leaving bars in the default
    categorical template (effectively invisible in dark mode)."""
    _plot_df, size_df = segmentation
    # Use distinct hex per segment so we can verify each bar gets its own.
    cmap = {0: "#60a5fa", 1: "#34d399", 2: "#fb7185", 3: "#a78bfa", 4: "#fb923c"}
    size_df = size_df.copy()
    size_df["Segment"] = size_df["Segment"].astype(str)
    str_cmap = {str(k): v for k, v in cmap.items()}
    fig = charts.bar_segments(
        size_df, "Segment", "Customers", color_map=str_cmap,
        xlabel="Segment", ylabel="Customers", height=380,
    )
    # px.bar with `color=` emits one trace per category — collect every bar
    # colour across all traces and assert they are all hex strings AND match
    # the discrete map.
    colours: list = []
    for trace in fig.data:
        c = _marker_dict(trace.marker).get("color")
        if isinstance(c, list):
            colours.extend(c)
        else:
            colours.append(c)
    assert len(colours) == len(size_df), (
        f"expected {len(size_df)} bar colours, got {len(colours)}"
    )
    for c in colours:
        # Must be a hex string, NOT an integer index like 0/1/2/3/4.
        assert isinstance(c, str) and c.startswith("#"), (
            f"bar colour is not a hex string: {c!r} (regression of "
            f"Plotly integer-index bug)"
        )
    assert set(colours) == set(cmap.values()), (
        f"bar colours {colours} do not match discrete map {list(cmap.values())}"
    )


def test_income_vs_spending_produces_per_segment_traces(segmentation):
    """Regression: px.scatter with an integer color column previously
    collapsed every segment into one continuous-colorscale trace, leaving
    the chart effectively monochrome. Each segment must now be its own
    Scatter trace with its own hex colour from the color_map."""
    plot_df, _ = segmentation
    cmap = {0: "#60a5fa", 1: "#34d399", 2: "#fb7185", 3: "#a78bfa", 4: "#fb923c"}
    fig = _build_income_vs_spending(plot_df, "dark")
    cmap_int_to_str = {str(k): v for k, v in cmap.items()}
    n_segs = plot_df["Segment"].nunique()
    assert len(fig.data) == n_segs, (
        f"expected {n_segs} per-segment traces, got {len(fig.data)}"
    )
    for trace in fig.data:
        color = trace.marker.color
        assert isinstance(color, str) and color.startswith("#"), (
            f"scatter trace colour not hex: {color!r}"
        )
        assert color in cmap_int_to_str.values(), (
            f"scatter trace colour {color!r} not from color_map"
        )


# ---------------------------------------------------------------------------
# Income vs Spending scatter
# ---------------------------------------------------------------------------

def _build_income_vs_spending(plot_df: pd.DataFrame, theme_name: str) -> go.Figure:
    cmap = {int(cid): chart_palette(theme_name)[i % len(chart_palette(theme_name))]
            for i, cid in enumerate(sorted(plot_df["Segment"].unique()))}
    fig = charts.scatter_segments(
        plot_df, ANNUAL_INCOME, SPENDING_SCORE, "Segment",
        "Income vs Spending",
        color_map=cmap,
        hover_data=[AGE, GENRE],
        hover_labels=["Age", "Gender"],
        xlabel="Annual Income (k$)", ylabel="Spending Score (1-100)",
        height=440,
    )
    fig.update_layout(**charts.get_plotly_theme(theme_name))
    return fig


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_income_vs_spending_has_visible_points(segmentation, theme_name: str):
    plot_df, _size_df = segmentation
    fig = _build_income_vs_spending(plot_df, theme_name)

    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert scatter_traces, "Income vs Spending has no Scatter trace"

    # All customer points must be present.
    total_points = sum(
        len(np.asarray(t.x)) for t in scatter_traces if t.x is not None
    )
    assert total_points == len(plot_df), (
        f"expected {len(plot_df)} points, got {total_points}"
    )

    # Markers must be visible: size > 0, opacity > 0.
    for t in scatter_traces:
        m = _marker_dict(t.marker)
        size = m.get("size", 6)
        opacity = m.get("opacity", 1.0)
        assert size is None or size > 0
        assert opacity is None or opacity > 0

    # Layout tokens applied.
    layout_t = get_theme(theme_name)
    assert fig.layout.plot_bgcolor == layout_t["chart_plot"]
    assert fig.layout.paper_bgcolor == layout_t["chart_paper"]
    assert fig.layout.xaxis.gridcolor == layout_t["chart_grid"]
    assert fig.layout.yaxis.gridcolor == layout_t["chart_grid"]
    assert fig.layout.xaxis.tickfont.color == layout_t["text_primary"]
    assert fig.layout.yaxis.tickfont.color == layout_t["text_primary"]


# ---------------------------------------------------------------------------
# Light vs dark parity — the same data must be present in both themes.
# ---------------------------------------------------------------------------

def test_segment_distribution_light_dark_parity(segmentation):
    _plot_df, size_df = segmentation
    light = _build_segment_distribution(size_df, "light")
    dark = _build_segment_distribution(size_df, "dark")
    def _total_y(fig):
        return int(sum(int(sum(t.y)) for t in fig.data))
    light_y = _total_y(light)
    dark_y = _total_y(dark)
    assert light_y == dark_y == int(size_df["Customers"].sum())


def test_income_vs_spending_light_dark_parity(segmentation):
    plot_df, _ = segmentation
    light = _build_income_vs_spending(plot_df, "light")
    dark = _build_income_vs_spending(plot_df, "dark")
    def _point_count(fig):
        return sum(len(np.asarray(t.x)) for t in fig.data if t.x is not None)
    assert _point_count(light) == _point_count(dark) == len(plot_df)
    # Backgrounds must differ between themes so dark gets the dark tokens.
    assert light.layout.plot_bgcolor != dark.layout.plot_bgcolor
    assert light.layout.colorway != dark.layout.colorway