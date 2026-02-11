from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
  
    root_dir: Path
    source_data: Path
    local_data_file: Path 

@dataclass(frozen=True)
class DataValidationConfig:

    root_dir: Path
    source_data: Path
    STATUS_FILE: str
    all_schema: dict

@dataclass(frozen=True)
class DataPreprocessingConfig:

    root_dir: Path
    source_data: Path
    data_validation_status_file: Path
    X: Path
    y: Path
    train_test_split: Path