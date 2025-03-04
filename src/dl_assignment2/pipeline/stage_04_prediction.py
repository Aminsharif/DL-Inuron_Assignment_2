from dl_assignment2.config.configuration import ConfigurationManager
from dl_assignment2.components.prediction import Prediction

STAGE_NAME = "Prediction stage"


class PredictionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_prediction_config()
        evaluation = Prediction(eval_config)
        proces_img = evaluation.preprocess_image()
        result = evaluation.predict(proces_img)
        print(result)
        return result