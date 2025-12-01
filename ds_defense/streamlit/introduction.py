"""Data_Introduction.py"""
from pathlib import Path
import pandas as pd

import streamlit as st

from intro_overview_chalenges import display_challenges_and_limitations, \
    display_dataset_overview
from intro_file_selection import file_selection
from intro_query import df_query


@st.cache_data
def _load_data(file_path: Path) -> pd.DataFrame:
    """Loads a single parquet file into a DataFrame."""
    try:
        df = pd.read_parquet(file_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    except (FileNotFoundError, IOError, KeyError, ValueError) as e:
        st.error(f"Error loading file {file_path}: {e}")
        return pd.DataFrame()


def _configure_page():
    st.set_page_config(page_title="Data Introduction", page_icon=None)


def _display_title():
    st.title("Data Introduction")


def _data_query() -> None:
    """Look into data"""

    selected_row: pd.DataFrame = file_selection()
    if not selected_row.empty:
        file_path_str = selected_row['path'].iloc[0]
        df = _load_data(Path(file_path_str))
        df_query(selected_row, df)


def introduction() -> None:
    """combine"""
    _configure_page()
    _display_title()
    display_dataset_overview()
    display_challenges_and_limitations()
    _data_query()


if __name__ == "__main__":
    introduction()
