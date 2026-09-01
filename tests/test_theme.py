"""Tests for the canonical theme token contract (src/ui/styles.py)."""
from __future__ import annotations

from src.ui import styles


DARK = styles.THEME_TOKENS["dark"]
LIGHT = styles.THEME_TOKENS["light"]
REQUIRED = styles.REQUIRED_TOKENS

DARK_SPEC = {
    "background": "#0b0f14",
    "surface": "#111827",
    "surface_elevated": "#172033",
    "surface_hover": "#1f2937",
    "border": "#263244",
    "border_strong": "#334155",
    "text_primary": "#f8fafc",
    "text_secondary": "#cbd5e1",
    "text_muted": "#94a3b8",
    "accent": "#2563eb",
    "accent_hover": "#1d4ed8",
    "success": "#16a34a",
    "warning": "#ea580c",
    "danger": "#dc2626",
    "info": "#0891b2",
    "input_background": "#172033",
    "input_border": "#334155",
    "sidebar_background": "#0b0f14",
    "chart_paper": "#0b0f14",
    "chart_plot": "#111827",
    "chart_grid": "#334155",
    "chart_axis": "#64748b",
    "chart_marker_outline": "#0b0f14",
}

LIGHT_SPEC = {
    "background": "#ffffff",
    "surface": "#ffffff",
    "surface_elevated": "#ffffff",
    "surface_hover": "#f1f5f9",
    "border": "#e2e8f0",
    "border_strong": "#cbd5e1",
    "text_primary": "#0f172a",
    "text_secondary": "#475569",
    "text_muted": "#94a3b8",
    "accent": "#2563eb",
    "accent_hover": "#1d4ed8",
    "success": "#16a34a",
    "warning": "#ea580c",
    "danger": "#dc2626",
    "info": "#0891b2",
    "input_background": "#ffffff",
    "input_border": "#cbd5e1",
    "sidebar_background": "#f8fafc",
    "chart_paper": "#ffffff",
    "chart_plot": "#ffffff",
    "chart_grid": "#e2e8f0",
    "chart_axis": "#94a3b8",
    "chart_marker_outline": "#ffffff",
}


def test_light_theme_tokens():
    t = styles.get_theme("light")
    assert t["name"] == "light"
    for key, expected in LIGHT_SPEC.items():
        assert t[key] == expected, f"light {key} mismatch: {t[key]} != {expected}"


def test_dark_theme_tokens():
    t = styles.get_theme("dark")
    assert t["name"] == "dark"
    for key, expected in DARK_SPEC.items():
        assert t[key] == expected, f"dark {key} mismatch: {t[key]} != {expected}"


def test_theme_token_completeness():
    for theme_name in styles.THEME_TOKENS:
        tokens = styles.THEME_TOKENS[theme_name]
        for key in REQUIRED:
            assert key in tokens, f"{theme_name} missing token {key}"
        # every token except the theme marker must be a hex colour
        for key in REQUIRED:
            if key == "name":
                continue
            assert isinstance(tokens[key], str) and tokens[key].startswith("#"), (
                f"{theme_name}.{key} is not a hex colour: {tokens[key]}"
            )
    dark = styles.THEME_TOKENS["dark"]
    light = styles.THEME_TOKENS["light"]
    assert dark["background"] != light["background"]
    assert dark["text_primary"] != light["text_primary"]
    assert dark["surface"] != light["surface"]
    assert dark["border"] != light["border"]


def test_get_theme_returns_independent_copies():
    a = styles.get_theme("dark")
    a["background"] = "#000000"
    b = styles.get_theme("dark")
    assert b["background"] == DARK["background"]


def test_theme_contract_is_theme_name():
    """get_theme with no explicit name follows the active Streamlit theme."""
    active = styles.get_theme()
    assert active["name"] in ("light", "dark")
    assert set(active) >= set(REQUIRED)
