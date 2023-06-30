from build_model import solution_A1, solution_A2, solution_A3, solution_A4
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Embedding, GlobalAveragePooling1D, LSTM, Bidirectional
from tensorflow.keras.applications.inception_v3 import InceptionV3

class AccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.83 and logs.get('val_accuracy') > 0.83):
            print("\Both Accuracy is more than 83%, stopping...")
            self.model.stop_training = True

class Hyperparameters:
    def __init__(self):
        self._neurons: Tuple[float] = None
        self._epochs: int = None
        self._batch_size: int = None
        self._activation: Tuple[str] = None
        self._optimizer: str = None
        self._loss: str = None
        self._metrics: List[str] = None

    @property
    def neurons(self) -> Tuple[float]:
        return self._neurons

    @neurons.setter
    def neurons(self, value: Tuple[float]):
        self._neurons = value

    @property
    def epochs(self) -> int:
        return self._epochs

    @epochs.setter
    def epochs(self, value: int):
        self._epochs = value

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value

    @property
    def activation(self) -> Tuple[str]:
        return self._activation

    @activation.setter
    def activation(self, value: Tuple[str]):
        self._activation = value

    @property
    def optimizer(self) -> str:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: str):
        self._optimizer = value

    @property
    def loss(self) -> str:
        return self._loss

    @loss.setter
    def loss(self, value: str):
        self._loss = value

    @property
    def metrics(self) -> List[str]:
        return self._metrics

    @metrics.setter
    def metrics(self, value: List[str]):
        self._metrics = value

class SubmissionA1(Hyperparameters):
    def __init__(self, input_shape: Tuple[int]):
        super().__init__()
        self.input_shape: Tuple[int] = input_shape
    
    @property
    def input_shape(self) -> Tuple[int]:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value: Tuple[int]):
        self._input_shape = value

    def build_layers(self):
        # layers = [Dense(units=neuron, activation=act, input_shape=self.input_shape if i == 0 else None) for i, (neuron, act) in enumerate(zip(self.neurons, self.activation))]
        raw_layers = []
        for i, (neuron, act) in enumerate(zip(self.neurons, self.activation)):
            if i == 0:
                raw_layers.append(Dense(units=neuron, activation=act, input_shape=self.input_shape))
            else:
                raw_layers.append(Dense(units=neuron, activation=act))
        model = tf.keras.Sequential(raw_layers)

        return model
    
    def build_model(self):
        model = self.build_layers()
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        
        return model
    
    def train_model(self):
        model = self.build_model()
        X, Y = solution_A1()
        
        model.fit(X, Y, epochs=self.epochs)

        return model
    
class SubmissionA2(Hyperparameters):
    def __init__(self, input_shape: Tuple[int], filters: Tuple[int], kernel_size: Tuple[int], activation_conv2d: str):
        super().__init__()
        self.input_shape: Tuple[int] = input_shape
        self.filters: Tuple[int] = filters
        self.kernel_size: Tuple[int] = kernel_size
        self.activation_conv2d: str = activation_conv2d

    def build_layers(self):
        raw_cnn_layers = []

        for index, filter in enumerate(self.filters): 
            if index == 0:
                raw_cnn_layers.append(Conv2D(filters=filter, kernel_size=self.kernel_size, activation=self.activation_conv2d, input_shape=self.input_shape))
            else:
                raw_cnn_layers.append(Conv2D(filters=filter, kernel_size=self.kernel_size, activation=self.activation_conv2d))
            raw_cnn_layers.append(MaxPool2D(pool_size=(2, 2)))

        raw_cnn_layers.append(Flatten())
        raw_cnn_layers.append(Dense(self.neurons[0], activation=self.activation[0]))
        raw_cnn_layers.append(Dense(self.neurons[1], activation=self.activation[1]))
        
        model = tf.keras.Sequential(raw_cnn_layers)

        return model
    
    def build_model(self):
        model = self.build_layers()
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        
        return model
    
    def train_model(self):
        model = self.build_model()
        callback = AccuracyCallback()
        X, Y = solution_A2(self.batch_size, self.input_shape)
        
        model.fit(X, validation_data=Y, epochs=self.epochs, callbacks=[callback])

        return model
    
class SubmissionA3(Hyperparameters):
    def __init__(self, input_shape: Tuple[int], weights: str, include_top: bool):
        super().__init__()
        self.input_shape: Tuple[int] = input_shape
        self.weights: str = weights
        self.include_top: bool = include_top
        
    def build_layers(self):
        # Initialize the base model.
        # Set the input shape and remove the dense layers.
        local_weights_file = solution_A3()
        pre_trained_model = InceptionV3(
            input_shape = self.input_shape, 
            include_top = self.include_top, 
            weights = self.weights
        )

        # Load the pre-trained weights we downloaded.
        pre_trained_model.load_weights(local_weights_file)

        # Freeze the weights of the layers.
        for layer in pre_trained_model.layers:
            layer.trainable = False
        
        # Choose `mixed_7` as the last layer of our base model
        last_layer = pre_trained_model.get_layer('mixed7')
        last_layer_output = last_layer.output
        
        return pre_trained_model, last_layer_output

    def build_model(self):
        pre_trained_model, last_layer_output = self.build_layers()

        x = Flatten()(last_layer_output)
        x = Dense(self.neurons[0], activation=self.activation[0])(x)
        x = Dropout(0.2)(x)
        x = Dense(self.neurons[1], activation=self.activation[1])(x)

        model = tf.keras.Model(pre_trained_model.input, x)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        return model

    def train_model(self):
        model = self.build_model()
        callback = AccuracyCallback()
        X, Y = solution_A2(self.batch_size, self.input_shape)
        
        model.fit(X, validation_data=Y, epochs=self.epochs, callbacks=[callback])

        return model

class SubmissionA4(Hyperparameters):
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 16
    MAX_LENGTH = 120
    TRUNCATING = 'post'
    OOV_TOKEN = '<OOV>'

    def __init__(self):
        super().__init__()

    def build_layers(self):
        model = tf.keras.Sequential([
            Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIM, input_length=self.MAX_LENGTH),
            Dropout(0.2),
            GlobalAveragePooling1D(),
            Dropout(0.2),
            Dense(self.neurons[0], activation=self.activation[0]),
        ])

        return model

    def build_model(self):
        model = self.build_layers()
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        return model

    def train_model(self):
        model = self.build_model()
        callback = AccuracyCallback()
        X_train, y_train, X_test, y_test = solution_A4(self.VOCAB_SIZE, self.MAX_LENGTH, self.TRUNCATING, self.OOV_TOKEN)

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.epochs, callbacks=[callback])

        return model
        

class SubmissionA5(Hyperparameters):
    pass