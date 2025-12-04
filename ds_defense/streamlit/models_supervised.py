"""
The superivsed models

"""
from dataclasses import dataclass

import streamlit as st


@dataclass
class Features:
  can_id: str
  can_id_prev1_3: str
  dlc: int
  data0_7: int
  delta_t: float
  can_id_freq: float
  iat_std: float


def _dt_background() -> None:
    st.header("Background")


def _dt_features() -> None:
    st.header('Selected Features')

def _decision_tree() -> None:
    """The DT sections"""
    dt_check_backg = st.checkbox('Background')
    if dt_check_backg:
        _dt_background()
    dt_feat_backg = st.checkbox("Features Selection")
    if dt_feat_backg:
        _dt_features()



def supervised_models() -> None:
    """supervised models used in the training"""
    page_names = ['Decision Tree', 'XGBoost', None]
    page = st.radio("**Applied Supervised Models**",
                    page_names,
                    key='uupervised_model_page',
                    horizontal=True,
                    )
    if page == 'Decision Tree':
        st.header('Decision Tree')
        _decision_tree()
    elif page == 'XGBoost':
        st.header('XGBoost')


if __name__ == '__main__':
    pass