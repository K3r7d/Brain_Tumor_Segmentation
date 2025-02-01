# src/models/cnn_model.py

import numpy as np
import cv2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class SimpleCNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = None

    def _conv_layer(self, input_image, filters, kernel_size, stride=1, padding=0):
        """
        Implement the convolution layer.
        """
        pass

    def _max_pool(self, input_image, pool_size=2, stride=2):
        """
        Implement max pooling layer.
        """
        pass


    def _flatten(self, input_image):
        """
        Flatten the feature maps into a single vector.
        """
        return input_image.flatten()

    def _fully_connected(self, input_vector, weights, bias):
        """
        Implement fully connected layer.
        """
        return np.dot(input_vector, weights) + bias

    def forward(self, X):
        """
        Pass input through the CNN layers.
        """
        # Implement forward pass here with convolution, pooling, and fully connected layers
        pass

    def train(self, X_train, y_train):
        """
        Train the CNN model with gradient descent and backpropagation.
        """
        pass
