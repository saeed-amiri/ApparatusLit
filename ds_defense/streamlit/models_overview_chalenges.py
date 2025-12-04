"""
Showing overview and chalenges of the CanBus Dat
"""

import streamlit as st


def display_models_overview():
    """Overview of the data"""
    st.header("Models Overview")
    model_view = st.checkbox("Display models")
    if model_view:
        st.markdown("""
        The data for this project consists of real-world CAN bus
         logs, collected from a vehicle under various conditions.
    """)
        st.markdown("""
    - **Source:** Real vehicle CAN bus communication logs.
    - **Format:** Time-series data, stored in efficient `parquet` files.
    - **Structure:** Organized hierarchically into training and test sets,
        with further subdivisions for different attack scenarios.
    """)


def display_models_challenges_and_limitations():
    """Some challenges and limitations of the daat"""
    st.header("Key Challenges and Limitations")
    challenges = st.checkbox("Display challenges")

    if challenges:
        st.error("""
            **1. Extreme Class Imbalance**
            The dataset is highly imbalanced, with normal traffic
             (`attack=0`) vastly outnumbering attack traffic
             (`attack=1`). This poses a significant challenge
             for model training.
            """)

        st.error("""
            **2. Temporal Dependencies**
            CAN bus data is a time-series. The order and timing
             of messages are critical. Anomalies are often defined
             by a *change* in the temporal pattern, not just a
             single bad message.
            """)

        st.error("""
            **3. Subtle Attack Patterns**
            Some attacks, like `rpm` or `force-neutral`, are very
             subtle and designed to mimic normal behavior, making
             them difficult to distinguish from legitimate vehicle
             operations.
            """)

        st.info("""
            **Note:** A fundamental challenge for this project
             was that the initial test set only contained attack
             samples, making it impossible to measure false
             positives. A proper evaluation requires a test set
             with both normal and attack data.
            """)
