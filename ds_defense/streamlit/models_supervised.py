"""
The superivsed models

"""
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


def _intro() -> None:
    st.header("Background")


def _dt_features() -> None:
    st.header('Selected Features')
    st.markdown("""
    - **CAN ID**: `can_id`, `can_id_prev1`, `can_id_prev2`, `can_id_prev3`
    - **Time**: `timestamp`, `delta_t`, `iat_std`, `iat_mean`
    - **Payload**: `data0` through `data7`
    - **Frequency**: `can_id_freq`
    - **Metadata**:
        `set`, `scenario_category`, `vehicle_type`,
        `attack_scenario`, `data_label`, `file_attack`
    """)


def _dt_params() -> None:
    st.header('Parameters')
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Criterion", ['gini', 'entropy'], key='dt_criterion')
        st.slider("Max Depth", 1, 20, 5, key='dt_max_depth')
        st.slider("Min Samples Split", 2, 20, 2, key='dt_min_samples_split')
    with col2:
        st.slider("Min Samples Leaf", 1, 20, 1, key='dt_min_samples_leaf')
        st.number_input("Random State", value=42, key='dt_random_state')


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
        SubPages.PARAM.value: _dt_params,
    }

    if page in page_handlers:
        page_handlers[page]()


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
        SubPages.PARAM.value: _dt_params,
    }

    if page in page_handlers:
        page_handlers[page]()


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
        SubPages.PARAM.value: _dt_params,
    }

    if page in page_handlers:
        page_handlers[page]()


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
