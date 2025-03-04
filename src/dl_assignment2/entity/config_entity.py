from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int




from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    train_data: Path
    test_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_rotation_range: int
    params_width_shift_range: float 
    params_height_shift_range: float
    params_horizontal_flip: bool



@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    test_data: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int

@dataclass(frozen=True)
class PredictionConfig:
    path_of_model: Path
    image_name: str
    class_names: list