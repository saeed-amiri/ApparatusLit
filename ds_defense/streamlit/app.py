
"""
Streamlit app for the final presentation

"""

import streamlit as st
from welcome import welcome
from project_intro import project_intro
from introduction import introduction
from features import featuers
from models import models
from results import results
from conclusion import conclusion


def mk_pages(pages: list[str]) -> str:
    """Make three pages"""
    page = st.sidebar.radio("Go to", pages)
    return page


def main() -> None:
    """Self explanatory"""

    pages = {
        "Welcome!": welcome,
        "Project Intro": project_intro,
        "CanBus Data": introduction,
        "Features": featuers,
        "Modeling": models,
        "Results": results,
        "Conclusions": conclusion
    }

    page = mk_pages(list(pages.keys()))
    st.markdown("---")

    if page in pages:
        pages[page]()


if __name__ == '__main__':
    main()
