from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

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