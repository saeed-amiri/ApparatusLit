"""Picture presenter"""
from pathlib import Path
import streamlit as st


def presenter(slides: list[Path]) -> None:
    """Present pictures as slide"""
    n_slide = len(slides)

    col_prev, _, col_next = st.columns([1, 4, 1])

    with col_prev:
        if st.button("⬅️ Previous",
                     disabled=st.session_state.current_slide == 0):
            st.session_state.current_slide -= 1

    with col_next:
        if st.button("Next ➡️",
                     disabled=st.session_state.current_slide == n_slide - 1):
            if st.session_state.current_slide < n_slide - 1:
                st.session_state.current_slide += 1
            else:
                st.session_state.current_slide = 0

    st.markdown("<h3 style='text-align: center;'>Slide "
                f"{st.session_state.current_slide + 1} of "
                f"{n_slide}</h3>", unsafe_allow_html=True)

    current_slide_file = slides[st.session_state.current_slide]
    st.image(str(current_slide_file), width='content')

    st.session_state.current_slide = st.slider(
        "Or jump to slide:",
        0,
        n_slide - 1,
        st.session_state.current_slide
    )
