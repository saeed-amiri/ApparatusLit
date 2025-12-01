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


def _get_df_info_interactive(df_i: pd.DataFrame) -> None:
    """Displays an interactive summary of the DataFrame."""

    st.subheader("Column Summary")

    # Prepare data for the editor
    col_info = []
    for col_name in df_i.columns:
        col_info.append({
            "Column Name": col_name,
            "Data Type": str(df_i[col_name].dtype),
            "Non-Null Count": df_i[col_name].count(),
            "Null Count": df_i[col_name].isnull().sum(),
            "# Unique": df_i[col_name].nunique(),
            "Sample Value": str(
                df_i[col_name].dropna().unique()[0]
                if not df_i[col_name].dropna().empty else 'N/A')
        })

    info_df = pd.DataFrame(col_info)

    st.data_editor(
        info_df,
        column_config={
            "Column Name": st.column_config.TextColumn(disabled=True),
            "Data Type": st.column_config.TextColumn(disabled=True),
            "Non-Null Count": st.column_config.NumberColumn(disabled=True),
            "Null Count": st.column_config.NumberColumn(disabled=True),
            "# Unique": st.column_config.NumberColumn(disabled=True),
            "Sample Value": st.column_config.TextColumn(disabled=True)
        },
        hide_index=True,
        width='stretch',
        num_rows="dynamic"
    )


@st.cache_data
def _get_df_head(df_i: pd.DataFrame) -> str:
    st.subheader("First rows")
    return df_i.head().to_markdown()


@st.cache_data
def _get_number_attacks(df_i: pd.DataFrame) -> str:
    st.subheader("Value counts of attacks")
    n_attacks = df_i['attack'].value_counts()
    return n_attacks.to_frame().to_markdown()


def _df_query(df_i: pd.DataFrame) -> None:
    """Show a summary of the dataframe"""
    page_names = ['Info', 'Head', 'Attacks', None]
    page = st.radio('**Quick Query**',
                    page_names,
                    key='query_page_selector',
                    horizontal=True,
                    index=None,
                    )

    if page == 'Info':
        _get_df_info_interactive(df_i)
    elif page == 'Head':
        st.markdown(_get_df_head(df_i))
    elif page == 'Attacks':
        st.markdown(_get_number_attacks(df_i))
    else:
        st.text('None!')



def _data_query() -> None:
    """Look into data"""
    st.markdown("---")
    st.title('Data Query')
    st.markdown("---")

    selected_row: pd.DataFrame = _file_selection()
    if not selected_row.empty:
        st.markdown(f"Number of rwos: {selected_row['n_rows'].iloc[0]}")
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
