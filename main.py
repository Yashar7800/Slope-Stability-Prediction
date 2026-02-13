from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlProject.pipeline.stage_03_data_preprocessing import DataPreprocessingPipeline
from mlProject.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from mlProject.pipeline.stage_05_prediction import PredictionPipelineStage

logger.info('welcome to the custom logging')

STAGE_NAME = "Data Ingestion Stage"

try:
        
    logger.info(f"----->> stage {STAGE_NAME} started <<-----")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f"----->> stage {STAGE_NAME} completed <<----- \n\n")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f"-----> stage {STAGE_NAME} started <<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f"----->stage {STAGE_NAME} completed <<<<<< \n\nx============x")

except Exception as e:
    
        logger.exception(e)
        raise e


STAGE_NAME = "Data Preprocessing Stage"

try:
    logger.info(f"-----> stage {STAGE_NAME} started <<<<<<")
    obj = DataPreprocessingPipeline()
    obj.main()
    logger.info(f"----->stage {STAGE_NAME} completed <<<<<< \n\nx============x")

except Exception as e:
    
        logger.exception(e)
        raise e

STAGE_NAME = "Model Trainer Stage"

try:
    logger.info(f"-----> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainerPipeline()
    obj.main()
    logger.info(f"----->stage {STAGE_NAME} completed <<<<<< \n\nx============x")

except Exception as e:
    
        logger.exception(e)
        raise e


STAGE_NAME = "Prediction Stage"
try:
    logger.info(f"\n-----> stage {STAGE_NAME} started <<-----\n")
    obj = PredictionPipelineStage()
    obj.main()
    logger.info(f"\n-----> stage {STAGE_NAME} completed <<-----\n\nx============x")
except Exception as e:
    logger.exception(e)
    raise e
