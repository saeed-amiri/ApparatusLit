"""
Streamlit app for the final presentation
"""
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from welcome import welcome
from project_intro import project_intro
from introduction import introduction
from conclusion import conclusion
from features import featuers
from results import results
from models import models

from feature_data_loading import load_data


BASE_PATH = Path(__file__).resolve().parents[2]
DATA_BASE_PATH = BASE_PATH / Path("data/final_dataset")


def _configure_app() -> None:
    st.set_page_config(
        page_title="CAN Bus Anomaly Detection",
        layout="wide",
        initial_sidebar_state="expanded")


def mk_pages(pages: list[str]) -> str:
    """Make three pages"""
    return st.sidebar.radio("Table of Contents", pages)


def main() -> None:
    """Self explanatory"""
    test_df: pd.DataFrame | None
    train_df: pd.DataFrame | None
    train_df, test_df = load_data(DATA_BASE_PATH)

    _configure_app()

    pages: dict[str, Callable] = {
        "Welcome!": welcome,
        "Project Intro": project_intro,
        "CanBus Data": introduction,
        "Features": lambda: featuers(train_df, test_df),
        "Modeling": lambda: models(train_df, test_df),
        "Results": results,
        "Conclusions": conclusion
    }

    page: str = mk_pages(list(pages.keys()))

    st.markdown("---")

    if page in pages:
        pages[page]()


if __name__ == '__main__':
    main()
