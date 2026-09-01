"""AppTest smoke suite for the Streamlit dashboard.

Runs every algorithm across every dashboard section through Streamlit's
``AppTest`` runtime and asserts that no section produces an exception or a
``st.warning`` alert.  This guards the UI/theme integration end-to-end.

Performance note: each algorithm commits its configuration once and the
resulting ``AppTest`` instance is then reused to navigate through all seven
sections.  This loads the (cached) dataset and clusters it once per algorithm
instead of once per parametrised case, keeping the suite fast while still
exercising every ``algorithm x section`` combination.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

from app import ALGORITHMS

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

SECTION_HEADINGS: dict[str, str] = {
    "overview": "Customer Intelligence Overview",
    "analytics": "Customer Analytics",
    "lab": "Segmentation Lab",
    "evaluation": "Model Evaluation",
    "explorer": "Segment Explorer",
    "prediction": "New Customer Prediction",
    "export": "Export",
}

_SIDEBAR_ALGO = "Algorithm"
_SIDEBAR_NAV = "Navigate to a section"
_SIDEBAR_APPLY = "Apply Configuration"
_SIDEBAR_RELOAD = "Reload data"

_APP_CACHE: dict[str, AppTest] = {}


def _new_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    return at


def _assert_clean(at: AppTest, ctx: str) -> None:
    assert not at.exception, f"[{ctx}] exceptions: {[str(e) for e in at.exception]}"
    assert not at.warning, f"[{ctx}] warnings: {[str(w) for w in at.warning]}"


def _find(widgets, label: str):
    """Locate a sidebar widget by its rendered ``label``.

    Streamlit 1.62's ``WidgetList`` only supports integer/slice indexing for
    ``__getitem__`` and key-based lookup for ``__call__``; several controls
    here (the algorithm selectbox, the nav radio, the buttons) have no stable
    widget key, so widgets are always resolved by ``.label``.  This also keeps
    selection stable when a second selectbox (Linkage) is rendered for the
    agglomerative algorithm.
    """
    for w in widgets:
        if w.label == label:
            return w
    have = [w.label for w in widgets]
    raise AssertionError(f"widget {label!r} not found; available labels: {have}")


def _committed_app(algo_key: str) -> AppTest:
    """Return a cached ``AppTest`` with *algo_key* selected and applied.

    The configuration is committed once per algorithm (select -> Apply, which
    triggers an internal ``st.rerun``); the same instance is reused by every
    section test for that algorithm.
    """
    at = _APP_CACHE.get(algo_key)
    if at is not None:
        return at
    at = _new_app()
    _assert_clean(at, f"init {algo_key}")

    _find(at.sidebar.selectbox, _SIDEBAR_ALGO).select(algo_key)
    at.run()
    _assert_clean(at, f"select {algo_key}")

    _find(at.sidebar.button, _SIDEBAR_APPLY).click()
    at.run()
    _assert_clean(at, f"apply {algo_key}")
    assert str(_find(at.sidebar.selectbox, _SIDEBAR_ALGO).value) == algo_key

    _APP_CACHE[algo_key] = at
    return at


@pytest.mark.parametrize("algo_key", list(ALGORITHMS.keys()))
@pytest.mark.parametrize("section_key", list(SECTION_HEADINGS.keys()))
def test_algorithm_across_sections(algo_key: str, section_key: str):
    """Each of the 4 algorithms must render every section without errors."""
    at = _committed_app(algo_key)

    _find(at.sidebar.radio, _SIDEBAR_NAV).set_value(section_key)
    at.run()
    _assert_clean(at, f"nav {algo_key}/{section_key}")
    assert _find(at.sidebar.radio, _SIDEBAR_NAV).value == section_key

    heading = SECTION_HEADINGS[section_key]
    assert any(
        heading in (m.value or "") for m in at.markdown
    ), f"section heading not rendered: {heading}"


def test_reload_data_button_is_clean():
    at = _new_app()
    _assert_clean(at, "reload base")
    _find(at.sidebar.button, _SIDEBAR_RELOAD).click()
    at.run()
    _assert_clean(at, "reload after click")
