"""Schema and data designs"""
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PlotData:
    """Data for plot in the introduction chapter"""
    base_path: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: str = "data/processed/can-train-and-test"
    test_dir: str = "test_01"
    sub_test_dir: str = "test_01_known_vehicle_known_attack"
    test_files: list[str] = field(
        default_factory=lambda: ["DoS-1.parquet.gzip"])
    train_dir: str = "train_01"
    train_files: list[str] = field(
        default_factory=lambda: ['DoS-1.parquet.gzip'])

