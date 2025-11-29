
"""
Streamlit app for the final presentation

"""

import streamlit as st


def _configure_app() -> None:
    st.set_page_config(
        page_title="CAN Bus Anomaly Detection",
        layout="centered",
        initial_sidebar_state="expanded"
        )


def _display_can_bus_title() -> None:
    st.title("CAN Bus Anomaly Detection")
    st.markdown(
        "### A Machine Learning Approach for Vehicle Security")
    st.markdown("---")


def _when() -> None:
    st.markdown("#### What?")
    st.info("""
    Developing a system to detect anomalies and
    cyber-attacks in a vehicle's Controller Area Network
    (CAN) bus using machine learning.
    """)


def _why() -> None:
    st.markdown("#### Why?")
    st.warning("""
    Modern vehicles are increasingly connected, making
     their critical CAN bus systems vulnerable to
     sophisticated cyber-attacks that traditional
         security can't catch.
    """)


def _how() -> None:
    st.markdown("#### How?")
    st.success("""
    By engineering time-series features and training
     advanced models like XGBoost to distinguish between
     normal vehicle behavior and malicious attack patterns.
    """)


def _render_markdown_separator() -> None:

    st.markdown("---")
    st.header("Presentation Contents")

    st.markdown("""
    set the sidebar to navigate through the different sections
     of this project:
    1.  **Data Introduction:** Explore the source, structure,
          and challenges of the CAN bus dataset.
    2.  **Data Visualization:** Visualize data distributions,
          temporal patterns, and class imbalance.
    3.  **Features:** Understand the key features engineered
          for the model.
    4.  **Models:** Learn about the machine learning models
          implemented.
    5.  **Results:** Compare model performance and see the
          outcomes.
    6.  **Conclusions:** Review key findings and future
          research directions.
    """)

    st.markdown("---")


def project_intro() -> None:
    """Welcome page"""

    _configure_app()
    _display_can_bus_title()
    _why()
    _when()
    _how()
    _render_markdown_separator()


if __name__ == '__main__':
    project_intro()
