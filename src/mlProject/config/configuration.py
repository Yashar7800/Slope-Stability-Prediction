from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories, save_json
from mlProject.entity.config_entity import (DataIngestionConfig, DataValidationConfig,
                                            DataPreprocessingConfig)
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