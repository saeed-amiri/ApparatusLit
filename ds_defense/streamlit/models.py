"""Presenting the models"""

import pandas as pd
import streamlit as st

from models_overview_chalenges import display_models_overview, \
    display_models_challenges_and_limitations
from models_supervised import supervised_models
from models_unsupervised import unsupervised_models


def _configure_page() -> None:
    st.set_page_config(page_title="Models", page_icon=None)


def _display_title() -> None:
    st.title("Model")

def _model_selection(
        train_df: pd.DataFrame | None, test_df: pd.DataFrame | None) -> None:
    """The selections for the models"""
    page_names = ['Supervised ML', 'Unsupervised ML', None]
    page = st.radio('**ML type**',
                    page_names,
                    key='model_page_selector',
                    horizontal=True,
                    index=None,
                    )
    if page == 'Supervised ML':
        st.text('Supervised models')
        supervised_models()
    elif page == 'Unsupervised ML':
        st.text('unsupervised models')
        if (train_df is not None) and (test_df is not None):
            unsupervised_models(train_df, test_df)
        else:
            st.error('It seems the dataframes are empty!')
    else:
        pass


def models(train_df: pd.DataFrame | None, test_df: pd.DataFrame | None
           ) -> None:
    """Model selections"""
    _configure_page()
    _display_title()
    display_models_overview()
    display_models_challenges_and_limitations()
    st.markdown("---")
    _model_selection(train_df, test_df)


if __name__ == '__main__':
    models(None, None)
