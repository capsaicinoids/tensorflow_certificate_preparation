from build_model import solution_A1
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, LSTM, Bidirectional

class Hyperparameters:
    def __init__(self):
        self.neurons: Tuple[float] = None
        self.epochs: int = None
        self.batch_size: int = None
        self.activation: str = None
        self.optimizer: str = None
        self.loss: str = None
        self.metrics: List[str] = None

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
    def activation(self) -> str:
        return self._activation

    @activation.setter
    def activation(self, value: str):
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

    def __init__(self):
        super().__init__()

    def build_layers(self):
        layers = [Dense(units=neuron, activation=self.activation, input_shape = [1]) for neuron in self.neurons]
        model = tf.keras.Sequential(layers)

        return model
    
    def build_model(self):
        model = self.build_layers()
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        
        return model
    
    def train_model(self):
        model = self.build_model()
        X, Y = solution_A1()
        
        model.fit(X, Y, epochs=self.epochs,)

        return model
    
class SubmissionA2(SubmissionA1):
    def __init__(self, filters: Tuple[int], kernel_size: Tuple[int], activation_conv2d: str):
        super().__init__()
        filters: Tuple[int] = filters
        kernel_size: Tuple[int] = kernel_size
        activation_conv2d: str = activation_conv2d

    def build_layers(self):
        dense_layers = super().build_layers()
        cnn_layers = []

        for filter in self.filters: 
            cnn_layers.append(Conv2D(filters=filter, kernel_size=self.kernel_size, activation=self.activation_conv2d))
            cnn_layers.append(MaxPool2D(pool_size=(2, 2)))
        
        model = tf.keras.Sequential(cnn_layers + dense_layers)

        return model
    
    def build_model(self):
        return super().build_model()
    