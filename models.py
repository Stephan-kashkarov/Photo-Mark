import cv2

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

class RPN(keras.models.Model):
    def __init__(self):
        pass

    def call(self, data):
        pass

    def train(self, labels, data):
        pass


class Faster_RCNN(keras.models.Model):
    def __init__(self):
        self.optimizer = keras.optimizers.Adam

        self.vgg16 = [
            keras.layers.ZeroPadding2D((1,1),input_shape=(3,224,224)),
            keras.layers.Convolution2D(64, 3, 3, activation='relu'),
            keras.layers.ZeroPadding2D((1,1)),
            keras.layers.Convolution2D(64, 3, 3, activation='relu'),
            keras.layers.MaxPooling2D((2,2), strides=(2,2)),

            keras.layers.ZeroPadding2D((1,1)),
            keras.layers.Convolution2D(128, 3, 3, activation='relu'),
            keras.layers.ZeroPadding2D((1,1)),
            keras.layers.Convolution2D(128, 3, 3, activation='relu'),
            keras.layers.MaxPooling2D((2,2), strides=(2,2)),

            keras.layers.ZeroPadding2D((1,1)),
            keras.layers.Convolution2D(256, 3, 3, activation='relu'),
            keras.layers.ZeroPadding2D((1,1)),
            keras.layers.Convolution2D(256, 3, 3, activation='relu'),
            keras.layers.ZeroPadding2D((1,1)),
            keras.layers.Convolution2D(256, 3, 3, activation='relu'),
            keras.layers.MaxPooling2D((2,2), strides=(2,2)),

            keras.layers.ZeroPadding2D((1,1)),
            keras.layers.Convolution2D(512, 3, 3, activation='relu'),
            keras.layers.ZeroPadding2D((1,1)),
            keras.layers.Convolution2D(512, 3, 3, activation='relu'),
            keras.layers.ZeroPadding2D((1,1)),
            keras.layers.Convolution2D(512, 3, 3, activation='relu'),
            keras.layers.MaxPooling2D((2,2), strides=(2,2)),

            keras.layers.ZeroPadding2D((1,1)),
            keras.layers.Convolution2D(512, 3, 3, activation='relu'),
            keras.layers.ZeroPadding2D((1,1)),
            keras.layers.Convolution2D(512, 3, 3, activation='relu'),
            keras.layers.ZeroPadding2D((1,1)),
            keras.layers.Convolution2D(512, 3, 3, activation='relu'),
            keras.layers.MaxPooling2D((2,2), strides=(2,2)),
        ]

        self.rpn = RPN()
    
    def call(self, data):
        pass

    def train(self, labels, data):
        pass