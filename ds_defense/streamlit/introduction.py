"""Data_Introduction.py"""
from pathlib import Path
from io import StringIO
import pandas as pd

import streamlit as st

from intro_visualisations import mk_files_df, load_data

def _configure_page():
    st.set_page_config(page_title="Data Introduction", page_icon=None)


def _display_title():
    st.title("Data Introduction")


def _display_dataset_overview():
    st.header("Dataset Overview")
    st.markdown("""
    The data for this project consists of real-world CAN bus
     logs, collected from a vehicle under various conditions.
""")
    st.markdown("""
- **Source:** Real vehicle CAN bus communication logs.
- **Format:** Time-series data, stored in efficient `parquet` files.
- **Structure:** Organized hierarchically into training and test sets,
    with further subdivisions for different attack scenarios.
""")


def _display_challenges_and_limitations():
    st.header("Key Challenges and Limitations")

    st.error("""
        **1. Extreme Class Imbalance**
        The dataset is highly imbalanced, with normal traffic
         (`attack=0`) vastly outnumbering attack traffic
         (`attack=1`). This poses a significant challenge
         for model training.
        """)

    st.error("""
        **2. Temporal Dependencies**
        CAN bus data is a time-series. The order and timing
         of messages are critical. Anomalies are often defined
         by a *change* in the temporal pattern, not just a
         single bad message.
        """)

    st.error("""
        **3. Subtle Attack Patterns**
        Some attacks, like `rpm` or `force-neutral`, are very
         subtle and designed to mimic normal behavior, making
         them difficult to distinguish from legitimate vehicle
         operations.
        """)

    st.info("""
        **Note:** A fundamental challenge for this project
         was that the initial test set only contained attack
         samples, making it impossible to measure false
         positives. A proper evaluation requires a test set
         with both normal and attack data.
        """)


def _file_selection() -> pd.DataFrame:
    """Plot samples of the data"""
    files_df: pd.DataFrame = mk_files_df()

    st.header("File Selection")

    cols = st.columns(4)

    with cols[0]:
        all_sets = sorted(files_df['set'].unique())
        selected_set = st.selectbox(
            "Select Set",
            all_sets,
            index=None,
            placeholder="Select contact method...")

    with cols[1]:
        filtered_set = files_df[files_df['set'] == selected_set]
        available_categories = sorted(filtered_set['category'].unique())
        selected_category = st.selectbox(
            "Select Category",
            available_categories,
            index=None,
            placeholder="Select contact method...",)

    with cols[2]:
        filtered_category = \
            filtered_set[filtered_set['category'] == selected_category]
        selected_file: str | None = st.selectbox(
            "Select File",
            filtered_category['name'],
            index=None,
            placeholder="Select contact method...",)

    with cols[3]:
        row = filtered_category[filtered_category['name'] == selected_file]
        attack_value = row['attack'].values[0] if not row.empty else None
        st.text_input(
            "Attack",
            value=attack_value if attack_value is not None else "",
            disabled=True,
            placeholder="Select a file first...")

    return row

@st.cache_data
def _get_df_info_string(df_i: pd.DataFrame) -> str:
    buffer = StringIO()
    df_i.info(buf=buffer)
    return buffer.getvalue()


@st.cache_data
def _get_df_head(df_i: pd.DataFrame) -> str:
    return df_i.head().to_markdown()


def _df_query(df_i: pd.DataFrame) -> None:
    """Show a summary of the dataframe"""
    page_names = ['Info', 'Head']
    page = st.radio('Quick Query', page_names, key='query_page_selector')

    if page == 'Info':
        st.text(_get_df_info_string(df_i))
    else:
        st.markdown(_get_df_head(df_i))


def _data_query() -> None:
    """Look into data"""
    st.markdown("---")
    st.title('Data Query')
    st.markdown("---")

    selected_row: pd.DataFrame = _file_selection()
    if not selected_row.empty:
        st.markdown(f"Number of rwos: {selected_row['n_rows']}")
        file_path_str = selected_row['path'].iloc[0]
        df = load_data(Path(file_path_str))
        _df_query(df)


def introduction() -> None:
    """combine"""
    _configure_page()
    _display_title()
    _display_dataset_overview()
    _display_challenges_and_limitations()
    _data_query()

if __name__ == "__main__":
    introduction()
