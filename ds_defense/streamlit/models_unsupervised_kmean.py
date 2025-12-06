"""
streamlit.models_unsupervised_knn

KNN model in unsuervised model
"""
from pathlib import Path
from enum import Enum

from presenter import presenter  # type: ignore

import streamlit as st


class MainPages(Enum):
    """Pages in hybrid section"""
    INTRO = 'Introduction'
    FEATURE = 'Features'
    PARAM = 'Parameters'
    TRAIN = 'Training models on a subset and running detection logic'
    SHAP = "Value of Contribution (SHAP)"
    NONE = None


FEATURES = ['can_id_dec', 'dlc', 'is_new_id', 'rolling_volatility',
            'zero_ratio', 'hamming_dist', 'log_iat', 'frequency_hz',
            'iat_rolling_mean_20']


def _train_model() -> None:
    pass


def _intro() -> None:

    base = Path(__file__).resolve().parents[0]
    slides_dir = base / "slides"

    slide_files = sorted(list(slides_dir.glob("unsupervised-knn*.png")))

    presenter(slides=slide_files, ran_seed=42, length=21)


def _feature() -> None:
    for feat in FEATURES:
        st.markdown('- ' + f"**{feat}**")


def _param() -> None:
    pass


def kmean_model() -> None:
    """Implementing KNN model in unsupervised ML"""
    pages = [page.value for page in MainPages]
    page = st.radio('Select Page',
                    pages,
                    horizontal=True,
                    index=None,
                    label_visibility='collapsed',
                    )

    page_handlers = {
        MainPages.TRAIN.value: _train_model,
        MainPages.INTRO.value: _intro,
        MainPages.FEATURE.value: _feature,
        MainPages.PARAM.value: _param,
    }
    if page in page_handlers:
        page_handlers[page]()
