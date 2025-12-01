"""
Query of the data in the introduction page
"""

import pandas as pd

import streamlit as st


def _get_df_info_interactive(df_i: pd.DataFrame) -> None:
    """Displays an interactive summary of the DataFrame."""

    st.subheader("Column Summary")

    # Prepare data for the editor
    col_info = []
    for col_name in df_i.columns:
        col_info.append({
            "Column Name": col_name,
            "Data Type": str(df_i[col_name].dtype),
            "Non-Null Count": df_i[col_name].count(),
            "Null Count": df_i[col_name].isnull().sum(),
            "# Unique": df_i[col_name].nunique(),
            "Sample Value": str(
                df_i[col_name].dropna().unique()[0]
                if not df_i[col_name].dropna().empty else 'N/A')
        })

    info_df = pd.DataFrame(col_info)

    st.data_editor(
        info_df,
        column_config={
            "Column Name": st.column_config.TextColumn(disabled=True),
            "Data Type": st.column_config.TextColumn(disabled=True),
            "Non-Null Count": st.column_config.NumberColumn(disabled=True),
            "Null Count": st.column_config.NumberColumn(disabled=True),
            "# Unique": st.column_config.NumberColumn(disabled=True),
            "Sample Value": st.column_config.TextColumn(disabled=True)
        },
        hide_index=True,
        width='stretch',
        num_rows="dynamic"
    )


@st.cache_data
def _get_df_head(df_i: pd.DataFrame) -> str:
    st.subheader("First rows")
    return df_i.head().to_markdown()


@st.cache_data
def _get_number_attacks(df_i: pd.DataFrame) -> str:
    st.subheader("Value counts of attacks")
    n_attacks = df_i['attack'].value_counts()
    return n_attacks.to_frame().to_markdown()


def df_query(selected_row: pd.DataFrame, df_i: pd.DataFrame) -> None:
    """Show a summary of the dataframe"""
    st.markdown(f"Number of rwos: {selected_row['n_rows'].iloc[0]}")
    page_names = ['Info', 'Head', 'Attacks', None]
    page = st.radio('**Quick Query**',
                    page_names,
                    key='query_page_selector',
                    horizontal=True,
                    index=None,
                    )

    if page == 'Info':
        _get_df_info_interactive(df_i)
    elif page == 'Head':
        st.markdown(_get_df_head(df_i))
    elif page == 'Attacks':
        st.markdown(_get_number_attacks(df_i))
    else:
        st.text('None!')
