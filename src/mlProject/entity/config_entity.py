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

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    train_target_path: Path
    test_target_path: Path
    model_path: Path
    rf_model_path: Path
    xgb_model_path: Path
    metrics_path: Path
    random_state: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    class_weight: str


@dataclass(frozen=True)
class PredictionConfig:
    root_dir: Path
    X_path: Path
    y_path: Path
    scaler_path: Path
    model_path: Path
    output_path: Path
