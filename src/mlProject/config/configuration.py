from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories, save_json
from mlProject.entity.config_entity import (DataIngestionConfig, DataValidationConfig,
                                            DataPreprocessingConfig,ModelTrainerConfig,PredictionConfig)
from pathlib import Path

class ConfigurationManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,  
                 params_filepath = PARAMS_FILE_PATH,
                 schema_filepaht = SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepaht)

        create_directories([self.config.artifacts_root]) 

    def get_data_ingestion_config(self) -> DataIngestionConfig:  
        config = self.config.data_ingestion

        create_directories([config.root_dir])   

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_data=Path(config.source_data),
            local_data_file= Path(config.local_data_file),
        )

        return data_ingestion_config
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir= config.root_dir,
            source_data = Path(config.source_data),
            STATUS_FILE = config.STATUS_FILE,
            all_schema = schema
    
        )
        return data_validation_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            source_data = Path(config.source_data),
            data_validation_status_file= config.data_validation_status_file,
            X = Path(config.X),
            y = Path(config.y),
            train_test_split = Path(config.train_test_split)
        )

        return data_preprocessing_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            train_target_path=Path(config.train_target_path),
            test_target_path=Path(config.test_target_path),
            model_path=Path(config.model_path),
            rf_model_path=Path(config.rf_model_path),
            xgb_model_path=Path(config.xgb_model_path),
            metrics_path=Path(config.metrics_path),
            random_state=params.random_state,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            subsample=params.subsample,
            colsample_bytree=params.colsample_bytree,
            class_weight=params.class_weight
        )

        return model_trainer_config
    
    def get_prediction_config(self) -> PredictionConfig:
        config = self.config.prediction
        create_directories([config.root_dir])
        return PredictionConfig(
            root_dir=config.root_dir,
            X_path=Path(config.X_path),
            y_path=Path(config.y_path) if config.y_path else None,
            scaler_path=Path(config.scaler_path),
            model_path=Path(config.model_path),
            output_path=Path(config.output_path)
        )