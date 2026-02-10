import os
import urllib.request as request
import zipfile
from mlProject import logger
from mlProject.utils.common import get_size
from mlProject.entity.config_entity import DataIngestionConfig
from pathlib import Path
import pandas as pd


class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config
        


    def load_file(self):
        try:
            if not os.path.exists(self.config.local_data_file):
                path = self.config.source_data
                logger.info(f"\nThe path pf the main source is: {self.config.source_data}")
                data = pd.read_csv(path, sep=',')
                data.to_csv(self.config.local_data_file,index=False)
                logger.info(f"\nData is successfully loaded to {self.config.local_data_file}")
                logger.info(f"\nThe shape of the initial dataset is: {data.shape}")   
                logger.info(f"The head of the dataset is as below:\n{data.head(10)}")
            else:
                logger.info(f'file already exists of size: {get_size(Path(self.config.local_data_file))}')
        except Exception as e:
            raise FileExistsError(f"Error loading file: {e}")