"""Presenting the features"""

import sys
from pathlib import Path
from enum import Enum
import pandas as pd

import streamlit as st

try:
    from .feature_data_overview import data_overview  # type: ignore
    from .feature_visualizing import feature_visulisation  # type: ignore
    from .presenter import presenter
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from feature_data_overview import data_overview  # type: ignore
    from feature_visualizing import feature_visulisation  # type: ignore
    from presenter import presenter  # type: ignore


def _set_title() -> None:
    st.title("Feature Engineering")


def _intro() -> None:

    base = Path(__file__).resolve().parents[0]
    slides_dir = base / "slides"

    slide_files = sorted(list(slides_dir.glob("feature*.png")))

    presenter(slides=slide_files, ran_seed=23, length=21)


class FeaturePages(Enum):
    """Main pages in the Feature section"""
    OVERVIEW = 'Data Overview'
    FEATURE_VIZ = 'Feature Visualization'
    NONE = None


def featuers(train_df: pd.DataFrame | None, test_df: pd.DataFrame | None
             ) -> None:
    """set the feature page up"""
    _set_title()
    _intro()

    pages = [page.value for page in FeaturePages]
    page = st.radio('Feature Engineering & Visualization',
                    pages,
                    horizontal=True,
                    index=None)

    page_handlers = {
        FeaturePages.OVERVIEW.value: lambda: data_overview(train_df, test_df),
        FeaturePages.FEATURE_VIZ.value:
            lambda: feature_visulisation(train_df, test_df),
    }

    if page is not None and page in pages:
        page_handlers[page]()  # types: ignore


if __name__ == '__main__':
    featuers(None, None)
