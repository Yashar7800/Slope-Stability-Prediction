from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_preprocessing import DataPreprocessing
from mlProject import logger
from pathlib import Path

STAGE_NAME=  "Data Preprocessing Stage"

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):

        try:
            logger.info("--- Starting Stage 03: Data Preprocessing ---")

            config = ConfigurationManager()
            data_preprocessing_config = config.get_data_preprocessing_config()

            logger.info("\nConfiguration Loaded Successfully")

            data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
            data_preprocessing.initiate_data_preprocessing()
        except Exception as e:
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f'>>>> stage {STAGE_NAME} started <<<<<<')
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f'>>>>> stage {STAGE_NAME} completed <<<< \n\nx============x')
    except Exception as e:
        logger.exception(e)
        raise e