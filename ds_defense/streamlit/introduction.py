"""introduction.py"""
from pathlib import Path
import pandas as pd

from intro_file_selection import file_selection  # type: ignore
from intro_query import df_query  # type: ignore
from intro_visualization import visualization  # type: ignore
from presenter import presenter  # type: ignore

import streamlit as st


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


def _configure_page() -> None:
    st.set_page_config(page_title="CAN Protocol", page_icon=None)


def _display_title() -> None:
    st.title("CAN Protocol")


def _intro() -> None:

    base = Path(__file__).resolve().parents[0]
    slides_dir = base / "slides"

    slide_files = sorted(list(slides_dir.glob("canbus*.png")))

    presenter(slides=slide_files, ran_seed=19, length=19)


def _data_query() -> None:
    """Look into data"""

    selected_row: pd.DataFrame = file_selection()
    if not selected_row.empty:
        file_path_str = selected_row['path'].iloc[0]
        df = _load_data(Path(file_path_str))
        df_query(selected_row, df)
        visualization(selected_row, df)


def introduction() -> None:
    """combine"""
    _configure_page()
    _display_title()
    _intro()
    _data_query()


if __name__ == "__main__":
    introduction()
