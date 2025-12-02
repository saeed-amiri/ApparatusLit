"""Presenting the features"""

import sys
from pathlib import Path
from enum import Enum
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


import streamlit as st

try:
    from .feature_data_loading import load_data  # type: ignore
    from .feature_data_overview import data_overview  # type: ignore
    from .feature_visualizing import feature_visulisation
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from feature_data_loading import load_data
    from feature_data_overview import data_overview
    from feature_visualizing import feature_visulisation


BASE_PATH = Path(__file__).resolve().parents[2]
DATA_BASE_PATH = BASE_PATH / Path("data/final_dataset")


def _set_title() -> None:
    st.title("🔧 Feature Engineering")
    st.markdown("""
    This application visualizes the **Hybrid Anomaly Detection
     System** for CAN Bus security.
    It detects **DoS Attacks** and **Fuzzy Attacks** using a
     combination of:
    1.  **Isolation Forest** (Global Outlier Detection)
    2.  **Local Outlier Factor (LOF)** (Local Outlier Detection)
    3.  **Heuristic Priority Logic** (New ID & Frequency Checks)
    """)


class FeaturePages(Enum):
    """Main pages in the Feature section"""
    OVERVIEW = '📊 Data Overview'
    FEATURE_VIZ = '📈 Feature Visualization'
    NONE = None


def featuers() -> None:
    """set the feature page up"""
    _set_title()
    test_df: pd.DataFrame | None
    train_df: pd.DataFrame | None
    train_df, test_df = load_data(DATA_BASE_PATH)
    pages = [page.value for page in FeaturePages]
    page = st.radio('Feature Engineering & Visualization',
                    pages,
                    horizontal=True,
                    index=None)

    if page == FeaturePages.OVERVIEW.value:
        st.subheader(page)
        data_overview(train_df, test_df)

    elif page == FeaturePages.FEATURE_VIZ.value:
        st.subheader(page)
        st.write("Displaying Feature Visualization.")
        feature_visulisation(train_df, test_df)

    else:
        pass


if __name__ == '__main__':
    featuers()
