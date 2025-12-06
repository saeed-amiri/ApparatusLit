"""
Streamlit app for the final presentation
"""
from enum import Enum
from pathlib import Path
from typing import Callable

import pandas as pd

from welcome import welcome  # type: ignore
from project_intro import project_intro  # type: ignore
from introduction import introduction  # type: ignore
from conclusion import conclusion  # type: ignore
from feature_data_loading import load_data  # type: ignore
from features import featuers  # type: ignore
from models import modeller  # type: ignore

import streamlit as st


BASE_PATH = Path(__file__).resolve().parents[2]
DATA_BASE_PATH = BASE_PATH / Path("data/final_dataset")


class MainPages(Enum):
    """Main pages of the app"""
    WELCOME = "Welcome!"
    PROJECT = "Project Intro"
    FEATURES = "Features"
    MODELS = "Modeling & Results"
    CONCLUSION = "Conclusions"
    CANBUS = "Appendix (CAN-Bus Data)"


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

    pages = [page.value for page in MainPages]
    page = mk_pages(pages=pages)

    page_handlers: dict[str, Callable] = {
        MainPages.WELCOME.value: welcome,
        MainPages.PROJECT.value: project_intro,
        MainPages.FEATURES.value: lambda: featuers(train_df, test_df),
        MainPages.MODELS.value: lambda: modeller(train_df, test_df),
        MainPages.CONCLUSION.value: conclusion,
        MainPages.CANBUS.value: introduction
    }

    if page in page_handlers:
        page_handlers[page]()

    st.markdown("---")


if __name__ == '__main__':
    main()
