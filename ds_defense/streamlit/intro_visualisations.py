"""Visulaising the source data"""

import re
from pathlib import Path
import pandas as pd

import streamlit as st


BASE_PATH = Path(__file__).resolve().parents[2]
DATA_BASE_PATH = BASE_PATH / Path("data/processed/can-train-and-test")


def classify_from_path(paths: list[Path],
                       check_attack: bool = True
                       ) -> pd.DataFrame:
    """
    Classify the files manually based on the names of dirs
    the dirs are catagorized as:
    tree -d data/processed/can-train-and-test/
        data/processed/can-train-and-test/
        ├── set_01
        │   ├── test_01_known_vehicle_known_attack
        │   ├── test_02_unknown_vehicle_known_attack
        │   ├── test_03_known_vehicle_unknown_attack
        │   ├── test_04_unknown_vehicle_unknown_attack
        │   └── train_01
        ├── set_02
        │   ├── test_01_known_vehicle_known_attack
        │   ├── test_02_unknown_vehicle_known_attack
        │   ├── test_03_known_vehicle_unknown_attack
        │   ├── test_04_unknown_vehicle_unknown_attack
        │   └── train_01
        ├── set_03
        │   ├── test_01_known_vehicle_known_attack
        │   ├── test_02_unknown_vehicle_known_attack
        │   ├── test_03_known_vehicle_unknown_attack
        │   ├── test_04_unknown_vehicle_unknown_attack
        │   └── train_01
        └── set_04
            ├── test_01_known_vehicle_known_attack
            ├── test_02_unknown_vehicle_known_attack
            ├── test_03_known_vehicle_unknown_attack
            ├── test_04_unknown_vehicle_unknown_attack
            └── train_01

        25 directories
    """
    records: list[dict[str, str | Path | int | None]] = []
    patterns = {
        'set': r'(?i)set_(\d+)',
        'category': r'(?i)(test_\d+|train_\d+)',
        'vehicle': r'(?i)(known|unknown)_vehicle',
        'attack_scenario': r'(?i)(known|unknown)_attack',
        'data_label': (
            r'(?i)(DoS|force-neutral|rpm|standstill|double|'
            r'fuzzing|interval|speed|systematic|triple|'
            r'accessory|attack-free)(?=-\d)'),
        }

    for p in paths:
        results: dict[str, str | None] = {}
        for key, pattern in patterns.items():
            match_i = re.search(pattern, str(p))
            if match_i:
                results[key] = match_i.group(0).lower()
            else:
                results[key] = None

        attack: str = "unknown"
        n_rows: int | str = "unknown"

        if check_attack:
            attack, n_rows = _check_attack(p)

        records.append({
            'path': p,
            'name': p.name.split('.parquet')[0],
            'set': results.get('set'),
            'category': results.get('category'),
            'vehicle': results.get('vehicle'),
            'attack_scenario': results.get('attack_scenario'),
            'data_label': results.get('data_label'),
            'attack': attack,
            'n_rows': n_rows
        })

    return pd.DataFrame(records)


def _check_attack(path: Path) -> tuple[str, int | str]:
    """read the files and check if they are attacked or not"""
    try:
        df = pd.read_parquet(path, columns=["attack"])
        n_rows = df.shape[0]
    except (FileExistsError, FileNotFoundError, IOError, KeyError) as err:
        print(f"Error in reading: '{path}' with error:\n{err}\n")
        return "unknown", "unknown"

    if (df["attack"] == 1).any():
        del df
        return "attacked", n_rows

    return "attack-free", n_rows


@st.cache_data
def get_all_parquet_files(base_path: Path) -> dict[str, dict[str, list[Path]]]:
    """
    Recursively finds all parquet files and organizes them
    in a nested dictionary.
    Returns a dictionary like:
        {'set_01': {'train_01': [Path('file1.parquet'), ...], ...}, ...}
    """

    if not base_path.exists():
        st.error(f"Data path not found: {base_path}")
        return {}

    file_dict: dict[str, dict[str, list[Path]]] = {}

    for set_dir in base_path.iterdir():
        if not set_dir.is_dir():
            continue

        file_dict[set_dir.name] = {}

        for subset_dir in set_dir.iterdir():
            if not subset_dir.is_dir():
                continue

            files = sorted(f for f in subset_dir.glob("*.parquet.gzip"))
            if files:
                file_dict[set_dir.name][subset_dir.name] = files

    return file_dict


@st.cache_data
def load_data(file_path: Path) -> pd.DataFrame:
    """Loads a single parquet file into a DataFrame."""
    try:
        df = pd.read_parquet(file_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    except (FileNotFoundError, IOError, KeyError, ValueError) as e:
        st.error(f"Error loading file {file_path}: {e}")
        return pd.DataFrame()


@st.cache_data
def mk_files_df() -> pd.DataFrame:
    """Make the dataframe of all the files"""
    file_structure: dict[str, dict[str, list[Path]]] = \
        get_all_parquet_files(DATA_BASE_PATH)
    # flatten file_structure
    all_files: list[Path] = [
        file_path
        for set_dict in file_structure.values()
        for file_list in set_dict.values()
        for file_path in file_list
    ]
    files_df: pd.DataFrame = classify_from_path(all_files)
    return files_df



if __name__ == '__main__':
    fills_df: pd.DataFrame = mk_files_df()
