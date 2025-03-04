import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dl_assignment2.entity.config_entity import TrainingConfig
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    # Function to load a CIFAR-10 batch
    def load_cifar10_batch(self,file_path):
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        return data_dict[b'data'], np.array(data_dict[b'labels'])

    def load_data(self,):
        X_train = []
        y_train = []
        for i in range(1, 6):  # Loop through 5 training batches
            data, labels = self.load_cifar10_batch(os.path.join(self.config.train_data, f"data_batch_{i}"))
            X_train.append(data)
            y_train.append(labels)

        # Convert list to numpy array
        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)

        # Load testing data
        X_test, y_test = self.load_cifar10_batch(self.config.test_data)

        # Reshape images from (num_samples, 3072) to (num_samples, 32, 32, 3)
        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)

        # Normalize pixel values to range [0,1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        # Convert class labels to one-hot vectors
        y_train = to_categorical(y_train, 100)
        y_test = to_categorical(y_test, 100)
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def save_model(path: Path, model):
        model.save(path)

    def define_model(self):
       # Define the CNN model
        model = Sequential([
                Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                MaxPooling2D(2, 2),

                Conv2D(64, (3,3), activation='relu', padding='same'),
                MaxPooling2D(2, 2),

                Conv2D(128, (3,3), activation='relu', padding='same'),
                MaxPooling2D(2, 2),

                Flatten(),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(100, activation='softmax')  # 100 classes
            ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def model_train(self,model, X_train, X_test, y_train, y_test):
        datagen = ImageDataGenerator(
            rotation_range=self.config.params_rotation_range,
            width_shift_range=self.config.params_width_shift_range,
            height_shift_range=self.config.params_height_shift_range,
            horizontal_flip=self.config.params_horizontal_flip
        )

        datagen.fit(X_train)

        # Train the model
        model.fit(datagen.flow(X_train, y_train, batch_size=self.config.params_batch_size),
                            validation_data=(X_test, y_test),
                            epochs=self.config.params_epochs,
                            verbose=1)

        self.save_model(
            path=self.config.trained_model_path,
            model=model
        )