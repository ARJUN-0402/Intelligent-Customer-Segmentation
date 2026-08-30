"""Shared constants, path resolution, and helper utilities.

Centralising paths and column names here keeps every module portable: no
absolute, machine-specific paths are used anywhere in the codebase.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure as mpl_figure  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset column names
# ---------------------------------------------------------------------------
CUSTOMER_ID: str = "CustomerID"
GENRE: str = "Genre"
AGE: str = "Age"
ANNUAL_INCOME: str = "Annual Income (k$)"
SPENDING_SCORE: str = "Spending Score (1-100)"

#: Columns treated as numeric features.
NUMERIC_COLUMNS: list[str] = [AGE, ANNUAL_INCOME, SPENDING_SCORE]

#: Columns treated as categorical features.
CATEGORICAL_COLUMNS: list[str] = [GENRE]

#: Identifier columns excluded from modelling.
ID_COLUMNS: list[str] = [CUSTOMER_ID]

#: Default feature pair used for clustering (matches original analysis).
FEATURE_COLUMNS: list[str] = [ANNUAL_INCOME, SPENDING_SCORE]

# ---------------------------------------------------------------------------
# Portable path resolution (no absolute machine paths)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
MODELS_DIR: Path = PROJECT_ROOT / "models"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"

DEFAULT_DATA_PATH: Path = DATA_DIR / "Mall_Customers.csv"

# ---------------------------------------------------------------------------
# Modelling defaults
# ---------------------------------------------------------------------------
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_N_CLUSTERS: int = 5
MIN_K: int = 2
MAX_K: int = 10
N_INIT: int = 10

#: Column-name pairs for plotting labels.
LABEL_MAP: dict[str, str] = {
    ANNUAL_INCOME: "Annual Income (k$)",
    SPENDING_SCORE: "Spending Score (1-100)",
}


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a project-level logger.

    Uses a idempotent handler so repeated calls do not duplicate log lines.
    """
    logger = logging.getLogger("customer_segmentation")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_directory(path: Path) -> Path:
    """Create *path* (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(
    fig: mpl_figure.Figure,
    filename: str,
    directory: Optional[Path] = None,
) -> Path:
    """Persist *fig* as a PNG and return the destination path."""
    target_dir = directory if directory is not None else FIGURES_DIR
    ensure_directory(target_dir)
    dest = target_dir / filename
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    return dest
