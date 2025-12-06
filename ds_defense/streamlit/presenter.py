"""Picture presenter"""
import random
import string
from pathlib import Path
import streamlit as st


def _gen_random_key(seed: int = 99, lenght: int = 12) -> str:

    random.seed(seed)
    charcters = string.ascii_letters

    random_key = random.choices(charcters, k=lenght)
    return ''.join(random_key)


def presenter(slides: list[Path], ran_seed: int = 99, length: int = 12
              ) -> None:
    """Present pictures as slide"""
    n_slide = len(slides)

    if n_slide == 0:
        st.warning('There is nothing to show!')
        return

    if n_slide == 1:
        st.image(str(slides[0]), width="stretch")
        return

    col_prev, _, col_next = st.columns([1, 4, 1])

    if 'current_slide' not in st.session_state:
        st.session_state.current_slide = 0

    with col_prev:
        if st.button("⬅️ Previous",
                     key=_gen_random_key(seed=ran_seed, lenght=length),
                     disabled=st.session_state.current_slide == 0):
            if st.session_state.current_slide > 0:
                st.session_state.current_slide -= 1
            else:
                st.session_state.current_slide = 0

    with col_next:
        if st.button("Next ➡️",
                     key=_gen_random_key(seed=ran_seed + 1, lenght=length),
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
