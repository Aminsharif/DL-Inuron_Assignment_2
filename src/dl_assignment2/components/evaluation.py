from urllib.parse import urlparse
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from dl_assignment2.constants import *
from dl_assignment2.constants import *
from dl_assignment2.utils.common import  save_json
from dl_assignment2.entity.config_entity import EvaluationConfig
from pathlib import Path

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    
    def load_cifar10_batch(self,file_path):
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        return data_dict[b'data'], np.array(data_dict[b'labels'])

    def evaluation(self,):
 
        # Load testing data
        X_test, y_test = self.load_cifar10_batch(self.config.test_data)

        X_test = X_test.reshape(-1, 32, 32, 3)

        X_test = X_test.astype('float32') / 255.0

        y_test = to_categorical(y_test, 100)
        

        model = load_model(self.config.path_of_model)
        loss, acc = model.evaluate(X_test, y_test)
        scores = {"loss": loss, "accuracy": acc}
        save_json(path=Path("scores.json"), data=scores)