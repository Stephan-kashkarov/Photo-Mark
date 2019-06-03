import keras
import numpy as np
import pandas as pd
import tensorflow as tf

class RPN(keras.models.Model):
    def __init__(self, **kwargs):
        self.anchors = kwargs.get('anchors', 9)
        self.base = keras.layers.Conv2D(
            512,
            (3, 3),
            padding='same',
            activation='relu',
            kernel_initializer='normal'
        )
        self.classifier = keras.layers.Conv2D(
            self.anchors,
            (1, 1),
            activation='sigmoid',
            kernel_initializer='uniform',
        )
        self.regressor = keras.layers.Conv2D(
            self.anchors * 4,
            (1, 1),
            activation='linear',
            kernel_initializer='zero',
        )

    def call(self, data):
        x = self.base(data)
        return self.classifier(x), self.regressor(x)

    def train(self, labels, data):
        pass

class RCNN(keras.models.Model):
    def __init__(self):
        pass

    def call(self, data):
        pass
    
    def train(self, labels, data):
        pass

class Faster_RCNN(keras.models.Model):
    def __init__(self):
        self.optimizer = keras.optimizers.Adam

        self.vgg16 = keras.applications.vgg16(weights='imagenet')

        self.rpn = RPN()
        self.classifier = RCNN()
    
    def call(self, data):
        data = self.vgg16(data)
        regions, confidence = self.rpn(data)

    def train(self, labels, data):
        pass