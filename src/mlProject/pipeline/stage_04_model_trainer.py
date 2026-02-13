from mlProject.config.configuration import ConfigurationManager
from mlProject.components.model_trainer import ModelTrainer
from mlProject import logger

STAGE_NAME = "Model Trainer Stage"


class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        metrics, best_model = model_trainer.initiate_model_trainer()
        logger.info(f"Best model: {best_model}, F1: {metrics[best_model]['f1']:.4f}")

if __name__ == "__main__":
    try:
        logger.info(f'>>>> stage {STAGE_NAME} started <<<<<<')
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f'>>>>> stage {STAGE_NAME} completed <<<< \n\nx============x')
    except Exception as e:
        logger.exception(e)
        raise e