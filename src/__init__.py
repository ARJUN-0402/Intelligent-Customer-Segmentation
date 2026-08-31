"""Intelligent Customer Segmentation - modular ML pipeline package.

This package contains the reusable building blocks for loading, validating,
preprocessing, clustering, evaluating, and interpreting customer-segmentation
data. Import the individual modules from ``app.py`` or a notebook.
"""

from __future__ import annotations

__all__: list[str] = [
    "utils",
    "data_loader",
    "preprocessing",
    "clustering",
    "evaluation",
    "personas",
    "business_insights",
    "analytics",
]

__version__: str = "0.2.0"
