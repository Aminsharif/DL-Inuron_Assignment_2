import numpy as np
from tensorflow.keras.models import load_model
import cv2
from dl_assignment2.entity.config_entity import PredictionConfig

class Prediction:
    def __init__(self, config: PredictionConfig):
        self.config = config
    
    def preprocess_image(self):
        """Loads an image, resizes it to (32, 32, 3), and normalizes it."""
        img = cv2.imread(self.config.image_name)  # Read image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (32, 32))  # Resize to 32x32
        img = img.astype('float32') / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 32, 32, 3)
        return img

    def predict(self, processed_image):
        model = load_model(self.config.path_of_model)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions)
        return self.config.class_names[predicted_class]