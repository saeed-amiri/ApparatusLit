"""Presenting the models"""

from pathlib import Path

from enum import Enum
import pandas as pd

from models_supervised import supervised_models  # type: ignore
from models_unsupervised import unsupervised_models  # type: ignore
from presenter import presenter  # type: ignore

import streamlit as st


class MLTypes(Enum):
    """ML types"""
    SUPERVIZ = 'Supervised & Semi-Supervised Approaches'
    UNSUPERVIZ = 'Unsupervised &  Rule-Based Approaches'
    NONE = None


def _configure_page() -> None:
    st.set_page_config(page_title="Models", page_icon=None)


def _display_title() -> None:
    st.title("Machine Learning Model Approaches")


def _intro() -> None:

    base = Path(__file__).resolve().parents[0]
    slides_dir = base / "slides"

    slide_files = sorted(list(slides_dir.glob("ml-models*.png")))

    presenter(slides=slide_files)


def _model_selection(
        train_df: pd.DataFrame | None, test_df: pd.DataFrame | None) -> None:
    """The selections for the models"""
    pages = [page.value for page in MLTypes]
    page = st.radio('**ML type**',
                    pages,
                    key='model_page_selector',
                    horizontal=True,
                    label_visibility='collapsed',
                    index=None,
                    )

    page_handlers = {
        MLTypes.SUPERVIZ.value: supervised_models,
        MLTypes.UNSUPERVIZ.value:
            lambda: unsupervised_models(train_df, test_df) if
            (train_df is not None) and (test_df is not None) else None,
    }

    if page in page_handlers:
        page_handlers[page]()


def modeller(train_df: pd.DataFrame | None, test_df: pd.DataFrame | None
             ) -> None:
    """Model selections"""
    _configure_page()
    _intro()
    st.markdown("---")
    _model_selection(train_df, test_df)


if __name__ == '__main__':
    modeller(None, None)
