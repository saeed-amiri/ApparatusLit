"""
The superivsed models

"""
from dataclasses import dataclass
from enum import Enum

import streamlit as st


class SupervisedModels(Enum):
    """Studied models"""
    DT = 'Decision Tree'
    XGB = 'XGBoost'
    SVM = 'Support Vector Machines'
    NONE = None


class SubPages(Enum):
    """sub pages for eavery main model"""
    INTRO = 'Introduction'
    FEATURE = 'Features'
    PARAM = 'Parameters'
    TRAIN = 'Training models on a subset and running detection logic'
    SHAP = "Value of Contribution (SHAP)"
    NONE = None


@dataclass
class Features:
  can_id: str
  can_id_prev1_3: str
  dlc: int
  data0_7: int
  delta_t: float
  can_id_freq: float
  iat_std: float


def _intro() -> None:
    st.header("Background")


def _dt_features() -> None:
    st.header('Selected Features')


def _decision_tree() -> None:
    """The DT sections"""
    pages = [page.value for page in SubPages]
    page = st.radio("**Decision Tree**",
                    pages,
                    key='dt_model_page',
                    horizontal=True,
                    label_visibility='collapsed',
                    )
    page_handlers = {
        SubPages.INTRO.value: _intro,
        SubPages.FEATURE.value: _dt_features,
    }


def _xgboost() -> None:
    pages = [page.value for page in SubPages]
    page = st.radio("**XGBoost**",
                    pages,
                    key='xgb_model_page',
                    horizontal=True,
                    label_visibility='collapsed',
                    )
    page_handlers = {
        SubPages.INTRO.value: _intro,
        SubPages.FEATURE.value: _dt_features,
    }


def _svm() -> None:
    pages = [page.value for page in SubPages]
    page = st.radio("**SVM**",
                    pages,
                    key='svm_model_page',
                    horizontal=True,
                    label_visibility='collapsed',
                    )
    page_handlers = {
        SubPages.INTRO.value: _intro,
        SubPages.FEATURE.value: _dt_features,
    }


def supervised_models() -> None:
    """supervised models used in the training"""
    pages = [page.value for page in SupervisedModels]
    page = st.radio("**Applied Supervised Models**",
                    pages,
                    key='uupervised_model_page',
                    horizontal=True,
                    label_visibility='collapsed',
                    )

    page_handlers = {
        SupervisedModels.DT.value: _decision_tree,
        SupervisedModels.XGB.value: _xgboost,
        SupervisedModels.SVM.value: _svm,
    }
    if page in page_handlers:
        page_handlers[page]()
