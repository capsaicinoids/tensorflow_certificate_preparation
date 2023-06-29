import urllib.request
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import numpy as np

def solution_A1():
    # DO NOT CHANGE THIS CODE
    X = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0,
                 2.0, 3.0, 4.0, 5.0], dtype=float)
    Y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                 12.0, 13.0, 14.0, ], dtype=float)
    
    return X, Y

def solution_A2():
    # data_url_1 = 'https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip'
    # urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
    # local_file = 'horse-or-human.zip'
    # zip_ref = zipfile.ZipFile(local_file, 'r')
    # zip_ref.extractall('data/train-horse-or-human')

    # data_url_2 = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip'
    # urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
    # local_file = 'validation-horse-or-human.zip'
    # zip_ref = zipfile.ZipFile(local_file, 'r')
    # zip_ref.extractall('data/validation-horse-or-human')
    # zip_ref.close()

    TRAINING_DIR = 'data/train-horse-or-human'
    VALIDATION_DIR = 'data/validation-horse-or-human'
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(
        rescale=1/255
    )

    # YOUR IMAGE SIZE SHOULD BE 150x150
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    return train_generator, validation_generator

