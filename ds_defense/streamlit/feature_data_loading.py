"""
ds_defense.streamlit.feature_data_loading
Helper functions for loading and handling data for the feature
engineerig
"""

from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class FileNames:
    train: str = 'can_train_dataset_featured.parquet'
    test: str = 'can_test_dataset_featured.parquet'


@st.cache_data
def load_data(base_dir: Path, file_names: FileNames | None = None
              ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load data and retrun then"""

    if file_names is None:
        file_names = FileNames()
    train_path = base_dir / file_names.train
    test_path = base_dir / file_names.test

    try:
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        return train_df, test_df
    except (FileExistsError, FileNotFoundError) as err:
        st.error(f"Problem in loading {train_path} and/or {test_path}\n{err}")
        return None, None
