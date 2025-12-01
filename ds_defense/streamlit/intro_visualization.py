"""
Visualization of the selected data
"""

import pandas as pd
import plotly.express as px  # type: ignore

import streamlit as st


@st.cache_data
def _plot_can_id_frequency(df_i: pd.DataFrame) -> None:
    """Plots the frequency of each CAN ID."""
    if 'arbitration_id' not in df_i.columns:
        st.warning("Column 'arbitration_id' not found.")
        return

    id_counts = df_i['arbitration_id'].value_counts().reset_index()

    fig = px.bar(
        id_counts.head(20),
        x='arbitration_id',
        y='count',
        title='Top 20 Most Frequent CAN IDs',
        labels={'arbitration_id': 'CAN ID (in Hex)',
                'count': 'Number of Messages'},
        color='count',
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')


@st.cache_data
def _plot_message_rate_over_time(df_i: pd.DataFrame) -> None:
    """Plots the number of messages per second over time."""
    if 'timestamp' not in df_i.columns:
        st.warning("Column 'timestamp' not found.")
        return

    df_resampled = (df_i
                    .set_index('timestamp')
                    .resample('1s')
                    .size()
                    .reset_index()
                    .rename(columns={0: 'message_count'}))

    fig = px.line(
        df_resampled,
        x='timestamp',
        y='message_count',
        title='Message Rate Over Time (Messages per Second)',
        labels={'timestamp': 'Time', 'message_count': 'Messages per Second'}
    )
    st.plotly_chart(fig, width='stretch')


@st.cache_data
def _plot_delta_t_distribution(df_i: pd.DataFrame) -> None:
    """Plots the distribution of time between messages (Delta T)."""

    if 'delta_t' not in df_i.columns:
        if 'timestamp' in df_i.columns:
            df_i['delta_t'] = df_i['timestamp'].diff().dt.total_seconds()
        else:
            st.error("Cannot calculate Delta T without 'timestamp' column.")
            return

    delta_t_filtered = df_i['delta_t'].dropna()
    delta_t_filtered = delta_t_filtered[
        delta_t_filtered < delta_t_filtered.quantile(0.99)]

    fig = px.histogram(
        delta_t_filtered,
        nbins=200,
        title='Distribution of Inter-Arrival Times (Delta T)',
        labels={'value': 'Delta T (seconds)', 'count': 'Frequency'},
        marginal='box'
    )
    st.plotly_chart(fig, width='stretch')


def _plot_payload_byte_heatmap(df_i: pd.DataFrame, max_rows: int = 299999
                               ) -> None:
    """
    Analyzes and visualizes the frequency of each byte in the
    data payload.
    """
    if 'data_field' not in df_i.columns:
        st.warning("Column 'data_field' not found.")
        return
    if len(df_i) > max_rows:
        st.warning("Oops! The data is too big for HeatMap!\n"
                   f"(The limit is {max_rows} rows)")
        return

    df_payload = df_i[['data_field']].copy()

    df_payload['data_field'] = df_payload['data_field'].astype(str)
    df_payload['data_field'] = \
        df_payload['data_field'].str.replace(r'[^0-9a-fA-F]', '', regex=True)

    max_len = 16
    df_payload['data_field'] = \
        df_payload['data_field'].str.pad(max_len, side='left', fillchar='0')

    for i in range(8):
        start = i * 2
        end = start + 2
        df_payload[f'data{i}'] = df_payload['data_field'].str.slice(start, end)

    byte_cols = [f'data{i}' for i in range(8)]
    df_melted = df_payload[byte_cols].melt(
        var_name='Byte Position', value_name='Hex Value')

    df_melted['Hex Value'] = \
        df_melted['Hex Value'].apply(lambda x: int(x, 16) if x else 0)

    fig = px.density_heatmap(
        df_melted,
        x='Byte Position',
        y='Hex Value',
        title='Heatmap of Payload Byte Frequencies',
        labels={'count': 'Frequency', 'Hex Value': 'Byte Value (0-255)'},
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        xaxis_title="Payload Byte Position (0-7)",
        yaxis_title="Hex Value (as Integer 0-255)",
        coloraxis_colorbar_title="Frequency"
    )

    st.plotly_chart(fig, width='stretch')

    st.caption("""
    **How to read this plot:**
    - **X-axis:** Shows the position of the byte in the
         8-byte payload (0 to 7).
    - **Y-axis:** Shows the possible value of that byte
         (from 0 to 255).
    - **Color:** Bright spots indicate that a particular byte
         at a particular position very frequently has that value.
    - **Attack Pattern:** An attack might cause a normally dark
         area to become very bright, or vice-versa.
    """)


def visualization(selected_row: pd.DataFrame, df: pd.DataFrame) -> None:
    """plot some graphs for the introductions"""
    st.markdown("---")
    st.title(f'Data Visualization for: {selected_row['name'].iloc[0]}')
    st.markdown("---")

    can_freq = st.checkbox('Can Id Frequency')
    if can_freq:
        _plot_can_id_frequency(df)
    msg_in_time = st.checkbox('Message Rate Over Time')
    if msg_in_time:
        _plot_message_rate_over_time(df)
    delta_t = st.checkbox(r'$\Delta$t Distribution')
    if delta_t:
        _plot_delta_t_distribution(df)
    delta_t = st.checkbox('Payload Hyte Heatmap')
    if delta_t:
        _plot_payload_byte_heatmap(df)


if __name__ == '__main__':
    pass
