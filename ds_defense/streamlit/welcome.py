
"""
Streamlit app for the final presentation

"""

from pathlib import Path
from typing import Any
import streamlit as st


def _display_can_bus_title() -> None:
    title = "CAN Bus Anomaly Detection Project"
    sub_title = "A Machine Learning Approach for Vehicle Security"
    st.markdown(
        f"<h1 style='text-align:center;'>{title}</h1>",
        unsafe_allow_html=True,
        )
    st.markdown(
        f"<h3 style='text-align:center;'>{sub_title} </h3>",
        unsafe_allow_html=True,)
    st.markdown("---")


def _git_link() -> str:
    source = "https://github.com/DataScientest-Studio"
    repo_name = "sep25_bds_can-bus-anomaly-detection"
    link = f"{source}/{repo_name}"
    return link


def _presenter(presenter: dict[str, Any], mentor: str) -> None:

    st.subheader("Presenters:")
    cols = st.columns(3)
    cols[0].markdown(f"**{presenter['sravani']}**")
    cols[1].markdown(f"**{presenter['anas']}**")
    cols[2].markdown(f"**{presenter['saeed']}**")

    st.markdown("---")

    st.subheader("Project Mentor:")
    st.markdown(f"**{mentor}**")
    st.markdown("---")

    st.markdown(f"**GitHub Repository:** {_git_link()}")


def _mk_header():
    """
    Display images from figs/front_page/ in left and right
    corners"""
    # Create 3 columns: left image — title — right image
    col_left, col_mid, col_right = st.columns([1, 2, 1])
    base_path = Path(__file__).parent / "figs" / "front_page"
    logo_path = base_path / "agentur-fur-arbeit-logo.png"
    datascience_path = base_path / "datascience.png"

    with col_right:
        if logo_path.exists():
            st.image(str(logo_path), width="stretch")
        else:
            st.warning("Logo image not found")

    with col_mid:
        st.markdown(
            "<h1 style='text-align:center;'>DataScientest Project</h1>",
            unsafe_allow_html=True,
        )

    with col_left:
        if datascience_path.exists():
            st.image(str(datascience_path), width="stretch")
        else:
            st.warning("DataScience image not found")

    st.markdown("---")


def welcome() -> None:
    """Welcome page"""
    presenter: dict[str, str] = {
        "sravani": "Sravani Hukumathi Venkata",
        "anas": "Anas Haj Naeif",
        "saeed": "Saeed Amiri"
    }
    _mk_header()
    _display_can_bus_title()
    _presenter(presenter=presenter, mentor='Vincent Lalanne')


if __name__ == '__main__':
    welcome()
