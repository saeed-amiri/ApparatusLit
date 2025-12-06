"""
feature_data_overview
DataOverview section in the feature analyis
"""

from pathlib import Path
from enum import Enum
from typing import Callable
import functools

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

import streamlit as st


def check_df_input(func: Callable) -> Callable | None:
    """Check if the injected df is not None"""
    @functools.wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        if df is None:
            st.error("No data available to display. Please load a file first.")
            return
        return func(df, *args, **kwargs)
    return wrapper


class DataPages(Enum):
    """Main pages in the Feature section"""
    ANATOMY = 'Anatomy of Messages'
    OVERVIEW = 'Data Overview'
    DIST_VIZ = 'Ditributions & Hitograms'
    NONE = None


class DistrPages(Enum):
    """Distributions and Histograms"""
    ATTACK = 'Attack Distribution'
    TOP_CAN_ID = 'Top CAN IDs Distribution'
    IAT_LOG = 'IAT Distribution (Log Scale)'
    DATA_BYTE_HIST = 'Data Byte Histograms'
    DATA_BYTE_VIOLIN = 'Data Byte Violin Plots'
    DATA_BYTE_BOX = 'Data Byte Boxplots'
    NONE = None


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


@st.cache_data
def _get_top_frequent_can_id(df_i: pd.DataFrame, max_id: int
                             ) -> tuple[pd.DataFrame, list]:
    viz_prep_df = df_i.copy()

    label_mapper = {0: 'Normal', 1: 'DoS', 2: 'Fuzzy'}

    if pd.api.types.is_integer_dtype(viz_prep_df['label']):
        viz_prep_df['Label Name'] = viz_prep_df['label'].map(label_mapper)
    else:
        viz_prep_df['Label Name'] = viz_prep_df['label']

    unique_labels = viz_prep_df['Label Name'].unique()

    top_ids_sets = []

    for lab in unique_labels:
        top_id = (viz_prep_df[viz_prep_df['Label Name'] == lab]['can_id']
                  .value_counts()
                  .head(max_id)
                  .index
                  .tolist())
        top_ids_sets.extend(top_id)

    mixed_top_ids = sorted(list(set(top_ids_sets)))

    if len(viz_prep_df) > 100000:
        viz_df = (viz_prep_df[viz_prep_df['can_id'].isin(mixed_top_ids)]
                  .sample(n=100000, random_state=42))
    else:
        viz_df = viz_prep_df[viz_prep_df['can_id'].isin(mixed_top_ids)]
    return viz_df, mixed_top_ids


@check_df_input
def _show_attack_distribution(df_i: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x='label', data=df_i, ax=ax, color='royalblue')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Normal', 'DoS', 'Fuzzy'])
    st.pyplot(fig)


@check_df_input
def _show_top_can_id(df_i: pd.DataFrame, max_id: int = 20) -> None:
    st.markdown(f"Distribution of the most {max_id} frequent CAN IDs "
                "across different labels (Log Scale).")

    viz_df, mixed_top_ids = _get_top_frequent_can_id(df_i, max_id)
    fig4, ax4 = plt.subplots(figsize=(18, 8))
    sns.countplot(data=viz_df,
                  x='can_id',
                  hue='Label Name',
                  order=mixed_top_ids,
                  palette='tab10',
                  ax=ax4)

    plt.title(f"Combined Top {max_id} CAN IDs from Each Label")
    plt.xlabel("CAN ID")
    plt.ylabel("Frequency (Log Scale)")
    plt.xticks(rotation=90)
    ax4.set_yscale('log')
    plt.legend(title='Label')
    st.pyplot(fig4)


@check_df_input
def _show_iat_ditribution(df_i: pd.DataFrame) -> None:

    iat_col: str | None = 'iat'
    if iat_col not in df_i.columns:
        if 'iat_rolling_mean_20' in df_i.columns:
            iat_col = 'iat_rolling_mean_20'
        else:
            iat_col = None

    if iat_col:
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df_i,
                     x=iat_col,
                     hue='label',
                     log_scale=True,
                     bins=100,
                     alpha=0.6,
                     palette='crest',
                     ax=ax5)
        plt.title("IAT Distribution by Label (Log Scale)")
        st.pyplot(fig5)
    else:
        st.warning("IAT feature not found.")


@check_df_input
def _show_data_byte_hist(df_i: pd.DataFrame, data_cols: list[str]) -> None:
    if data_cols:
        rows = (len(data_cols) + 3) // 4
        fig6, axes6 = plt.subplots(rows, 4, figsize=(20, 5*rows))
        axes6 = axes6.flatten()

        i = 0
        for i, col in enumerate(data_cols):
            sns.histplot(data=df_i, x=col, hue='Label Name',
                         bins=20,
                         palette='Paired',
                         ax=axes6[i],
                         alpha=0.6)

            axes6[i].set_title(f"{col} by Label")
            axes6[i].set_yscale('log')

        for j in range(i+1, len(axes6)):
            axes6[j].axis('off')

        plt.tight_layout()
        st.pyplot(fig6)


@check_df_input
def _show_data_byte_violin(df_i: pd.DataFrame, data_cols: list[str]
                           ) -> None:
    if data_cols:
        rows = (len(data_cols) + 3) // 4
        fig7, axes7 = plt.subplots(rows, 4, figsize=(20, 5*rows))
        axes7 = axes7.flatten()

        i = 0
        for i, col in enumerate(data_cols):
            sns.violinplot(data=df_i,
                           x='Label Name',
                           y=col,
                           ax=axes7[i],
                           cut=0,
                           palette='muted')

            axes7[i].set_title(f"Violin Plot: {col}")

        for j in range(i+1, len(axes7)):
            axes7[j].axis('off')

        plt.tight_layout()
        st.pyplot(fig7)


@check_df_input
def _show_data_byte_box(df_i: pd.DataFrame, data_cols: list[str]
                        ) -> None:
    if data_cols:
        rows = (len(data_cols) + 3) // 4
        fig8, axes8 = plt.subplots(rows, 4, figsize=(20, 5*rows))
        axes8 = axes8.flatten()
        i = 0
        for i, col in enumerate(data_cols):
            sns.boxplot(data=df_i,
                        x='Label Name',
                        y=col,
                        ax=axes8[i],
                        palette='muted')
            axes8[i].set_title(f"Boxplot: {col}")

        for j in range(i+1, len(axes8)):
            axes8[j].axis('off')

        plt.tight_layout()
        st.pyplot(fig8)


def _show_distributions(test_df: pd.DataFrame | None) -> None:
    data_cols = [f'data{i}' for i in range(8)]
    if test_df is not None:
        valid_col = [c for c in data_cols if c in test_df.columns]
        sample_size = st.sidebar.slider(
            "Sample Size for Plots", 1000, 50000, 5000, key="data_viz_sample")
        plot_df = (
            test_df
            .groupby('label')
            .apply(
                lambda x: x.sample(n=min(len(x), sample_size), random_state=42)
                )
            .reset_index(drop=True))
        plot_df['Label Name'] = \
            plot_df['label'].map({0: 'Normal', 1: 'DoS', 2: 'Fuzzy'})
    else:
        return

    pages = [page.value for page in DistrPages]
    page = st.radio('Distributions and Histograms',
                    pages,
                    horizontal=True,
                    index=None)

    page_handlers = {
        DistrPages.ATTACK.value:
            lambda: _show_attack_distribution(plot_df),  # type: ignore
        DistrPages.TOP_CAN_ID.value:
            lambda: _show_top_can_id(plot_df),  # type: ignore
        DistrPages.IAT_LOG.value:
            lambda: _show_iat_ditribution(plot_df),  # type: ignore
        DistrPages.DATA_BYTE_HIST.value:
            lambda: _show_data_byte_hist(plot_df, valid_col),  # type: ignore
        DistrPages.DATA_BYTE_VIOLIN.value:
            lambda: _show_data_byte_violin(plot_df, valid_col),  # type: ignore
        DistrPages.DATA_BYTE_BOX.value:
            lambda: _show_data_byte_box(plot_df, valid_col),  # type: ignore
    }

    if page in page_handlers:
        page_handlers[page]()


def data_overview(train_df: pd.DataFrame | None, test_df: pd.DataFrame | None
                  ) -> None:
    """
    self explanatory
    """
    pages = [page.value for page in DataPages]
    page = st.radio('Anatomy and Overview',
                    pages,
                    horizontal=True,
                    index=None)

    if page == DataPages.ANATOMY.value:
        _show_anatomy_pic()

    if page == DataPages.OVERVIEW.value:
        _show_data_tabels(train_df, test_df)

    if page == DataPages.DIST_VIZ.value:
        _show_distributions(test_df)

    if page == DataPages.NONE.value:
        pass
