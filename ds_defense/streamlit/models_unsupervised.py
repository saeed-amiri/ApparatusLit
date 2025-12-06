"""
streamlit.models_unsupervised
Unsupervised models
"""

from enum import Enum
import pandas as pd

from models_unsupervised_hybrid import hybrid_model
from models_unsupervised_kmean import kmean_model

import streamlit as st


class UnsupervisedModels(Enum):
    """Studied models"""
    HYBRID = 'Hybrid Anomaly Detection System'
    KMEAN = 'K-Means Clustering'
    NONE = None


def _hybrid_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    hybrid_model(train_df, test_df)


def _kmean_model() -> None:
    kmean_model()


def unsupervised_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Unsupervised models"""
    pages = [page.value for page in UnsupervisedModels]
    page = st.radio('Unsupervised Models',
                    pages,
                    horizontal=True,
                    index=None,
                    label_visibility='collapsed',
                    )

    page_handlers = {
        UnsupervisedModels.HYBRID.value:
            lambda: _hybrid_model(train_df, test_df),
        UnsupervisedModels.KMEAN.value: _kmean_model,
    }

    if page in page_handlers:
        page_handlers[page]()
