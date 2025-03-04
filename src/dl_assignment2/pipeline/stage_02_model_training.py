from dl_assignment2.config.configuration import ConfigurationManager
from dl_assignment2.components.model_training import Training
from dl_assignment2 import logger



STAGE_NAME = "Training"



class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        X_train, X_test, y_train, y_test = training.load_data()
        model = training.define_model()
        training.model_train(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
