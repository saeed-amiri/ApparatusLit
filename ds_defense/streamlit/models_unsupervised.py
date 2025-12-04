"""
streamlit.models_unsupervised
Unsupervised models
"""

from enum import Enum
import pandas as pd
import streamlit as st

from models_unsupervised_hybrid import hybrid_model

class UnsupervisedModels(Enum):
    """Studied models"""
    KNN = 'KNN'
    HYBRID = 'Hybrid Anomaly Detection System'
    NONE: None


def _knn_model() -> None:
    ...


def _hybrid_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    hybrid_model(train_df, test_df)



def unsupervised_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Unsupervised models"""
    pages = [page.value for page in UnsupervisedModels]
    page = st.radio('Unsupervised Models',
                    pages,
                    horizontal=True,
                    index=None
                    )

    page_handlers = {
        UnsupervisedModels.KNN.value: _knn_model,
        UnsupervisedModels.HYBRID.value: \
            lambda: _hybrid_model(train_df, test_df),
    }

    if page in page_handlers:
        page_handlers[page]()
