"""Data_Introduction.py"""
from pathlib import Path
import pandas as pd

import streamlit as st

from intro_overview_chalenges import display_challenges_and_limitations, \
    display_dataset_overview
from intro_file_selection import file_selection, load_data
from intro_query import df_query


def _configure_page():
    st.set_page_config(page_title="Data Introduction", page_icon=None)


def _display_title():
    st.title("Data Introduction")


def _data_query() -> None:
    """Look into data"""
    st.markdown("---")
    st.title('Data Query')
    st.markdown("---")

    selected_row: pd.DataFrame = file_selection()
    if not selected_row.empty:
        st.markdown(f"Number of rwos: {selected_row['n_rows'].iloc[0]}")
        file_path_str = selected_row['path'].iloc[0]
        df = load_data(Path(file_path_str))
        df_query(df)


def introduction() -> None:
    """combine"""
    _configure_page()
    _display_title()
    display_dataset_overview()
    display_challenges_and_limitations()
    _data_query()


if __name__ == "__main__":
    introduction()
