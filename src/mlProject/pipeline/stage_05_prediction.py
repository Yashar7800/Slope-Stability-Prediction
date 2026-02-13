from mlProject.config.configuration import ConfigurationManager
from mlProject.components.prediction import PredictionPipeline
from mlProject import logger

STAGE_NAME = "Prediction Stage"

class PredictionPipelineStage:
    def main(self):
        config = ConfigurationManager()
        pred_config = config.get_prediction_config()
        predictor = PredictionPipeline(config=pred_config)
        predictor.run()

if __name__ == '__main__':
    try:
        logger.info(f"\n-----> stage {STAGE_NAME} started <<-----\n")
        obj = PredictionPipelineStage()
        obj.main()
        logger.info(f"\n-----> stage {STAGE_NAME} completed <<-----\n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e