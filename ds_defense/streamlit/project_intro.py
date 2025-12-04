
"""
Streamlit app for the final presentation

"""
import sys
from pathlib import Path
import streamlit as st

try:
    from .presenter import presenter  # type: ignore
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from presenter import presenter


def _display_can_bus_title() -> None:
    st.title("CAN Bus Anomaly Detection")
    st.markdown(
        "### A Machine Learning Approach for Vehicle Security")
    st.markdown("---")


def _show_presentation() -> None:
    """Presenting the slides"""
    if 'current_slide' not in st.session_state:
        st.session_state.current_slide = 0

    st.title("🎙️ Project Presentation")
    base = Path(__file__).resolve().parents[0]

    slides_dir = base / "slides"
    slide_files = sorted(list(slides_dir.glob("*slides*.png")))
    presenter(slides=slide_files)


def project_intro() -> None:
    """Welcome page"""

    _display_can_bus_title()
    _show_presentation()


if __name__ == '__main__':
    project_intro()
