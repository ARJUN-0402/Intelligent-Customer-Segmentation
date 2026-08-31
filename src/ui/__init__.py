"""Reusable presentation-layer utilities for the Customer Intelligence dashboard.

This package centralises the visual design system (colors, typography, spacing)
and provides reusable components (cards, charts, layout helpers) so that
``app.py`` stays a thin, readable presentation layer on top of the ``src`` ML
package. No ML / clustering logic lives here.
"""

from . import cards, charts, components, styles

__all__ = ["styles", "cards", "components", "charts"]
