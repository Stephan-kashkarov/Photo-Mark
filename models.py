import cv2

import keras
import numpy as np
import pandas as pd
import tensorflow as tf


def region_extraction(frame, regions):
    pass


class Region_Proposer(keras.models.Model):
    def __init__(self):
        super().__init__()

    def call(self, frame):
        pass

    def train(self, labels=None, images=None):
        pass


class Face_Classifier(keras.models.Model):
    def __init__(self):
        super().__init__()
        self.loss = keras.losses.sparse_categorical_crossentropy
        self.optimizer = keras.optimizers.Adam
        self.metrics = {
            "train": [
                tf.keras.metrics.Mean(name='train_loss'),
                tf.keras.metrics.SparseCategoricalAccuracy(
                    name='train_accuracy'
                ),
            ],
            "test": [
                tf.keras.metrics.Mean(name='test_loss'),
                tf.keras.metrics.SparseCategoricalAccuracy(
                    name='test_accuracy'
                ),
            ],
        }
        self.layers = [
            keras.layers.Conv2D(64, 8, activation='relu'),
            keras.layers.Conv2D(64, 4, activation='sigmoid'),
            keras.layers.MaxPool2D(pool_size=(4, 4)),
            keras.layers.Conv2D(16, 4, activation='relu'),
            keras.layers.Conv2D(16, 2, activation='sigmoid'),
            keras.layers.AvgPool2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='sigmoid'),
            keras.layers.Dense(8, activation='sigmoid'),
            keras.layers.Dense(2, activation='softmax'),
        ]

    def call(self, face):
        pass

    def train(self, labels=None, images=None):
        with tf.GradientTape() as tape:
            predictions = self(images)
            loss = self.loss(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_values)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_values))


class Faster_RCNN(keras.models.Model):
    def __init__(self):
        super().__init__()
        self.region_proposer = Region_Proposer()
        self.face_classifier = Face_Classifier()
        self.region_proposer.train()
        self.face_classifier.train()

    def call(self, frame):
        region_proposals = self.region_proposer.predict(frame)
        for index, region in region_extraction(frame, region_proposals):
            if self.face_classifier.predict(region, batch_size=1)[0]:
                regions.pop(index)
        return regions
