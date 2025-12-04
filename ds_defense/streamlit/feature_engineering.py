"""
ds_defense.streamlit.feature_engineering
Feature selection and engineerings

"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE  # type: ignore


def _calculate_rolling_volatility(df_i: pd.DataFrame, payload_cols: list[str],
                                  window: int = 5) -> pd.Series:
    """
    Calculates the rolling standard deviation of payload bytes.
    Args:
        df_i (pd.DataFrame): DataFrame containing:
            'data0'...'data7' and 'can_id'.
        window (int): Rolling window size.
    Returns:
        pd.Series: Rolling volatility values.
    """
    volatility = pd.Series(0.0, index=df_i.index)

    existing_cols = [col for col in payload_cols if col in df_i.columns]

    if not existing_cols:
        return volatility

    for col in existing_cols:
        byte_std = (df_i
                    .groupby('can_id')[col]
                    .transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                        )
                    )
        volatility += byte_std.fillna(0)

    return volatility / len(existing_cols) if existing_cols else volatility


def _calculate_hamming_distance(df_i: pd.DataFrame, payload_cols: list[str]
                                ) -> pd.Series:
    """
    Calculates the Hamming distance (bit flips) between
    consecutive payloads of the same CAN ID.
    Args:
        df_i (pd.DataFrame): DataFrame containing
            'data0'...'data7' and 'can_id'.
    Returns:
        pd.Series: Hamming distance values.
    """
    existing_cols = [col for col in payload_cols if col in df_i.columns]

    if not existing_cols:
        return pd.Series(0, index=df_i.index)

    payload_df = df_i[existing_cols].fillna(0).astype(np.uint8)

    # Shift to get previous payload for the same CAN ID
    prev_payloads = (payload_df
                     .groupby(df_i['can_id'])
                     .shift(1)
                     .fillna(0)
                     .astype(np.uint8))

    current_payloads_np = payload_df.to_numpy()
    prev_payloads_np = prev_payloads.to_numpy()

    # XOR to find bit differences
    xor_results = np.bitwise_xor(current_payloads_np, prev_payloads_np)

    # Sum the bits (unpackbits expands uint8 to 8 bits)
    # axis=1 sums bits across all columns for each row
    return np.sum(np.unpackbits(xor_results, axis=1), axis=1)


def _calculate_zero_ratio(df_i: pd.DataFrame, payload_cols: list[str]
                          ) -> np.ndarray | pd.Series:
    """
    Calculates the ratio of zero bytes in the payload.
    """
    existing_cols = [col for col in payload_cols if col in df_i.columns]

    if not existing_cols:
        return pd.Series(0.0, index=df_i.index)

    payloads = df_i[existing_cols].fillna(0).to_numpy()
    zero_counts = (payloads == 0).sum(axis=1)
    return zero_counts / 8.0


def _calculate_frequency_features(df_i: pd.DataFrame, window: int = 20
                                  ) -> pd.DataFrame:
    """
    Calculates Rolling IAT Mean and Frequency (Hz).
    Requires 'iat' column or 'timestamp'.
    """
    epsilon = 1e-9

    # Calculate IAT if missing but timestamp exists
    if 'iat' not in df_i.columns and 'timestamp' in df_i.columns:
        df_i = df_i.sort_values('timestamp')
        df_i['iat'] = df_i.groupby('can_id')['timestamp'].diff().fillna(0)

    if 'iat' in df_i.columns:
        # Rolling Mean IAT
        df_i['iat_rolling_mean_20'] = (
            df_i
            .groupby('can_id')['iat']
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean()
                       )
            )
        # Frequency = 1 / IAT
        df_i['frequency_hz'] = 1.0 / (df_i['iat_rolling_mean_20'] + epsilon)
        # Log IAT
        df_i['log_iat'] = np.log1p(df_i['iat'] + epsilon)

    return df_i


def _add_new_id_feature(df_i: pd.DataFrame, train_df_ref: pd.DataFrame
                        ) -> pd.DataFrame:
    """
    Adds 'is_new_id' flag based on reference training data.
    """
    if 'label' in train_df_ref.columns:
        # Assume normal label is 0 or 'Normal' or 'attack-free'
        # Check unique values to be safe, or assume 'label' column exists
        # Based on notebook 00, label is 'attack-free' or 0
        # Based on parquet loading in app.py, it might be mapped
        # We'll assume train_df_ref contains the Normal data
        # we want to learn from.
        # Best to filter outside or check for common normal labels
        normal_labels = [0, 'Normal', 'attack-free']
        valid_ids = set(train_df_ref[train_df_ref['label']
                        .isin(normal_labels)]['can_id']
                        .unique())
    else:
        # If no label, assume all ref data is normal
        valid_ids = set(train_df_ref['can_id'].unique())

    df_i['is_new_id'] = \
        df_i['can_id'].apply(lambda x: 1 if x not in valid_ids else 0)
    return df_i


def _calculate_pca_features(
        df_i: pd.DataFrame, features=None, n_components: int = 2
        ) -> pd.DataFrame:
    """
    Calculates PCA components for the dataframe.
    """
    if features is None:
        features = [f'data{i}' for i in range(8) if f'data{i}' in df_i.columns]
        if not features:
            return df_i

    x = df_i[features].fillna(0).values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x)

    for i in range(n_components):
        df_i[f'pca_{i+1}'] = principal_components[:, i]

    return df_i


def _calculate_tsne_features(
        df_i: pd.DataFrame, features=None, n_components: int = 2,
        perplexity: int = 30, n_iter: int = 1000) -> pd.DataFrame:
    """
    Calculates t-SNE components. WARNING: Slow on large datasets.
    """
    if features is None:
        features = [f'data{i}' for i in range(8) if f'data{i}' in df_i.columns]
        if not features:
            return df_i

    x = df_i[features].fillna(0).values
    x = StandardScaler().fit_transform(x)

    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                max_iter=n_iter,
                random_state=42)
    tsne_results = tsne.fit_transform(x)

    for i in range(n_components):
        df_i[f'tsne_{i+1}'] = tsne_results[:, i]

    return df_i


def _convert_can_id_to_decimal(df_i: pd.DataFrame) -> pd.DataFrame:
    if 'can_id' in df_i.columns:
        df_i['can_id_dec'] = (
            df_i['can_id']
            .apply(lambda x: int(x, 16) if isinstance(x, str) else x))
    return df_i


def _convert_dlc_to_int(df_i: pd.DataFrame) -> pd.DataFrame:
    """
    Converts DLC column to integer type.
    """
    if 'dlc' in df_i.columns:
        df_i['dlc'] = df_i['dlc'].fillna(0).astype(int)
    return df_i


def feature_engineering_pipeline(
        df_i: pd.DataFrame, train_df_ref: pd.DataFrame | None = None
        ) -> pd.DataFrame:
    """
    Master function to apply all feature engineering.
    """
    # 1. Heuristics
    if train_df_ref is not None:
        df_i = _add_new_id_feature(df_i, train_df_ref)

    # 2. Frequency
    df_i = _calculate_frequency_features(df_i)

    # 3. Payload Features
    payload_cols: list[str] = [f'data{i}' for i in range(8)]
    df_i['rolling_volatility'] = \
        _calculate_rolling_volatility(df_i, payload_cols)
    df_i['hamming_dist'] = _calculate_hamming_distance(df_i, payload_cols)
    df_i['zero_ratio'] = _calculate_zero_ratio(df_i, payload_cols)

    # 4. Conversions
    df_i = _convert_can_id_to_decimal(df_i)

    df_i = _convert_dlc_to_int(df_i)

    df_i.fillna(0, inplace=True)

    return df_i
