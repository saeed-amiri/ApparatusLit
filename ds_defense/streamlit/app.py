"""
Streamlit app for the final presentation
"""

from typing import Callable

import streamlit as st

from welcome import welcome
from project_intro import project_intro
from introduction import introduction
from conclusion import conclusion
from features import featuers
from results import results
from models import models


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
    _configure_app()

    pages: dict[str, Callable] = {
        "Welcome!": welcome,
        "Project Intro": project_intro,
        "CanBus Data": introduction,
        "Features": featuers,
        "Modeling": models,
        "Results": results,
        "Conclusions": conclusion
    }

    page: str = mk_pages(list(pages.keys()))

    st.markdown("---")

    if page in pages:
        pages[page]()


if __name__ == '__main__':
    main()
