"""
feature_visualizing
Showing graphs for the selected features
"""

import sys
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

import streamlit as st

try:
    from .feature_engineering import \
        feature_engineering_pipeline  # type: ignore
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from feature_engineering import feature_engineering_pipeline


@st.cache_data
def _compute_features(train_df: pd.DataFrame, test_df: pd.DataFrame
                      ) -> pd.DataFrame:
    if 'frequency_hz' not in test_df.columns:
        with st.spinner("Calculating features..."):
            feature_df = feature_engineering_pipeline(test_df, train_df)
        return feature_df
    return test_df


def _sampler_slider(df_i: pd.DataFrame) -> pd.DataFrame:
    sample_size = st.slider("Sample Size for Plots", 1000, 50000, 5000)
    plot_df = (
        df_i
        .groupby('label', group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), sample_size), random_state=42))
        .reset_index(drop=True)
        )
    plot_df['Label Name'] = \
        plot_df['label'].map({0: 'Normal', 1: 'DoS', 2: 'Fuzzy'})
    return plot_df


def _freuency_sample_plot(df_i: pd.DataFrame) -> None:
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='Label Name',
                y='frequency_hz',
                data=df_i,
                ax=ax1,
                showfliers=False,
                color='royalblue'
                )
    plt.yscale('log')
    st.pyplot(fig1)
    st.caption("DoS attacks show massive frequency spikes (>50Hz).")


def _rolling_sample_plot(df_i: pd.DataFrame) -> None:
    if 'rolling_volatility' in df_i.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='Label Name',
                    y='rolling_volatility',
                    data=df_i,
                    ax=ax2,
                    showfliers=False
                    )
        st.pyplot(fig2)
        st.caption("Fuzzy attacks often have higher volatility due "
                   "to random injection.")
    else:
        st.error("Rolling Volatility feature not found in dataset.")


def _heuristic_sample_plot(df_i: pd.DataFrame) -> None:
    st.sidebar.subheader("3. New ID Check (Heuristic)")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Label Name', hue='is_new_id', data=df_i, ax=ax3)
    plt.legend(title='Is New ID')
    st.pyplot(fig3)
    st.caption(
        "Fuzzy attacks almost exclusively use New IDs. Normal and DoS do not.")


def _sample_plots(df_i: pd.DataFrame) -> None:
    frq_check = st.checkbox(
        'Frequency (DoS Detection)', key='frequency_dos')
    if frq_check:
        _freuency_sample_plot(df_i)
    roll_check = st.checkbox('Rolling Volatility (Fuzzy Detection)',
                             key='rolling_fuzzy')
    if roll_check:
        _rolling_sample_plot(df_i)
    heuristic_check = st.checkbox('New ID Check (Heuristic)',
                                  key='heuristic')
    if heuristic_check:
        _heuristic_sample_plot(df_i)


def feature_visulisation(train_df: pd.DataFrame | None,
                         test_df: pd.DataFrame | None) -> None:
    """Visualising features"""
    if test_df is not None and train_df is not None:
        st.markdown(
            "Analyzing key features that separate attacks from normal traffic."
            )
        feature_df: pd.DataFrame = _compute_features(train_df, test_df)
        plot_df = _sampler_slider(feature_df)
        _sample_plots(plot_df)
