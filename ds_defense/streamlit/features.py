"""Presenting the features"""

import sys
from pathlib import Path
from enum import Enum
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

import streamlit as st

try:
    from .feature_data_loading import load_data  # type: ignore
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from feature_data_loading import load_data


BASE_PATH = Path(__file__).resolve().parents[2]
DATA_BASE_PATH = BASE_PATH / Path("data/final_dataset")


class FeaturePages(Enum):
    """Main pages in the Feature section"""
    OVERVIEW = '📊 Data Overview'
    VIZ = '📈 Feature Visualization'
    RAW_VIZ = '📊 Raw Data Visualization'
    NONE = None


class DataPages(Enum):
    """Main pages in the Feature section"""
    ANATOMY = 'Anatomy of Messages'
    OVERVIEW = 'Data Overview'
    ATTACK_VIZ = 'Attack Ditribution'
    NONE = None


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


def _show_anatomy_pic() -> None:
    base_path = Path(__file__).parent / "figs" / "features"
    pic_path = base_path / "can_message.png"
    if pic_path.exists():
        st.image(str(pic_path), width="content")
    else:
        st.warning("The picture cannot be shown right now!")


def _show_data_tabels(
        train_df: pd.DataFrame | None, test_df: pd.DataFrame | None) -> None:
    st.header("Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Data (Normal)")
        if train_df is not None:
            st.write(train_df.head())
            st.write(f"Shape: {train_df.shape}")
            st.write("Label Distribution:")
            st.write(train_df['label'].value_counts())
        else:
            st.error("It does seemes the you trained nothing!")

    with col2:
        st.subheader("Testing Data (Attacks)")
        if test_df is not None:
            st.write(test_df.head())
            st.write(f"Shape: {test_df.shape}")
            st.write("Label Distribution (0=Normal, 1=DoS, 2=Fuzzy):")
            st.write(test_df['label'].value_counts())
        else:
            st.error("It does seemes the you tested nothing!")


def _show_attack_dist(test_df: pd.DataFrame | None) -> None:
    st.subheader("Attack Distribution")
    if test_df is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x='label', data=test_df, ax=ax, color='royalblue')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Normal', 'DoS', 'Fuzzy'])
        st.pyplot(fig)
    else:
        st.error("It does seemes there is nothing to present!")


def _data_overview(train_df: pd.DataFrame | None, test_df: pd.DataFrame | None
                   ) -> None:
    pages = [page.value for page in DataPages]
    page = st.radio('Anatomy and Overview',
                    pages,
                    horizontal=True,
                    index=None)

    if page == DataPages.ANATOMY.value:
        _show_anatomy_pic()

    elif page == DataPages.OVERVIEW.value:
        _show_data_tabels(train_df, test_df)

    elif page == DataPages.ATTACK_VIZ.value:
        _show_attack_dist(test_df)

    if page == DataPages.NONE.value:
        pass


def featuers() -> None:
    """set the feature page up"""
    _set_title()
    train_df, test_df = load_data(DATA_BASE_PATH)
    pages = [page.value for page in FeaturePages]
    page = st.radio('Feature Engineering & Visualization',
                    pages,
                    horizontal=True,
                    index=None)

    if page == FeaturePages.OVERVIEW.value:
        st.subheader(page)
        st.write("Displaying Data Overview.")
        _data_overview(train_df, test_df)

    elif page == FeaturePages.VIZ.value:
        st.subheader(page)
        st.write("Displaying Feature Visualization.")

    elif page == FeaturePages.RAW_VIZ.value:
        st.subheader(page)
        st.write("Displaying Raw Data Visualization.")
    else:
        pass


if __name__ == '__main__':
    featuers()
