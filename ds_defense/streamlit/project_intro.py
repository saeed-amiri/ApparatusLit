
"""
Streamlit app for the final presentation

"""

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as componets


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


def show_presentation_advanced() -> None:
    """Presenting the slides"""
    if 'current_slide' not in st.session_state:
        st.session_state.current_slide = 0

    st.title("🎙️ Project Presentation")
    base = Path(__file__).resolve().parents[0]

    slides_dir = base / "slides"
    slide_files = sorted(list(slides_dir.glob("*.png")))

    total_slides = len(slide_files)

    col_prev, _, col_next = st.columns([1, 4, 1])

    with col_prev:
        if st.button(
            "⬅️ Previous", disabled=st.session_state.current_slide == 0):
            st.session_state.current_slide -= 1

    with col_next:
        if st.button(
            "Next ➡️",
            disabled=st.session_state.current_slide == total_slides - 1):
            if st.session_state.current_slide < total_slides - 1:
                st.session_state.current_slide += 1
            else:
                st.session_state.current_slide = 0

    st.markdown("<h3 style='text-align: center;'>Slide "
                f"{st.session_state.current_slide + 1} of "
                f"{total_slides}</h3>", unsafe_allow_html=True)

    current_slide_file = slide_files[st.session_state.current_slide]
    st.image(str(current_slide_file), width='content')

    st.session_state.current_slide = st.slider(
        "Or jump to slide:",
        0, total_slides - 1,
        st.session_state.current_slide
    )


def project_intro() -> None:
    """Welcome page"""

    _display_can_bus_title()
    # _why()
    # _when()
    # _how()
    # _render_markdown_separator()
    show_presentation_advanced()


if __name__ == '__main__':
    project_intro()
