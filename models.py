import keras
import numpy as np
import pandas as pd
import tensorflow as tf

class RPN(keras.models.Model):
    def __init__(self, **kwargs):
        self.num_anchors = kwargs.get('anchors', 9)
        self.objectness_limit = kwargs.get('objectness_limit', 0.7)

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
        for point in np.ndindex(data):
            for coords, anchor in self.anchors(point):
                if self.classifier(anchor) > self.objectness_limit:
                    yield self.regressor(coords)

        

    def train(self, labels, data):
        pass

    def anchors(self, point):
        pass

class Detector(keras.models.Model):
    def __init__(self):
        pass

    def call(self, data):
        pass
    
    def train(self, labels, data):
        pass

class Faster_RCNN(keras.models.Model):
    def __init__(self):
        self.optimizer = keras.optimizers.Adam

        self.vgg16 = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)

        self.rpn = RPN()
        self.detector = Detector()
    
    def call(self, data):
        data = self.vgg16(data)
        regions = self.rpn(data)
        for region in regions:
            yield region, self.detector(region, data)

    def train(self, labels, data):
        pass