"""The introduction of the Streamlit"""

import streamlit as st

st.title('Test')
st.write('Introduction')

if st.checkbox('Display'):
    st.write('Streamlit works!')
