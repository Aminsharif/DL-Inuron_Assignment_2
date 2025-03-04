from dl_assignment2.constants import *
from dl_assignment2.constants import *
from dl_assignment2.utils.common import read_yaml, create_directories
from dl_assignment2.entity.config_entity import (DataIngestionConfig, TrainingConfig, EvaluationConfig, PredictionConfig)
import os

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        params = self.params
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            train_data=Path(training.train_data),
            test_data=Path(training.test_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_rotation_range=params.ROTATION_RANGE,
            params_width_shift_range=params.WIDTH_SHIFT_RANGE,
            params_height_shift_range=params.HEIGHT_SHIFT_RANGE,
            params_horizontal_flip=params.HORIZONTAL_FLIP,
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=self.config.evaluation.path_of_model,
            training_data=self.config.evaluation.training_data,
            test_data=Path(self.config.evaluation.test_data),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config
    
    def get_prediction_config(self) -> PredictionConfig:
        eval_config = PredictionConfig(
            path_of_model=self.config.prediction.path_of_model,
            image_name = self.config.prediction.image_name,
            class_names = self.params.class_names
        )
        return eval_config