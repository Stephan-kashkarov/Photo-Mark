import keras
import numpy as np
import pandas as pd
import tensorflow as tf

class RPN(keras.models.Model):
    """
    ######################################
    #### Region Propositional Network ####
    ######################################

    This network is designed to take the last convolutional
    layer from a VGG16 pretrained network. 
    
    The network then iterates through every point in this
    image and creates n bounding boxes named anchors. 
    N is defined through the Kwarg "anchors" and defaults to 9. 
    Each of these anchors are then scored by "objectivness".
    In other words, the chance the object is background or an object. 
    Given the objectiveness of said anchor is higher then the threshold
    defined in the "objectiveness_limit" kwarg, then this
    anchor is selected to have its bounding box regressed.
    All matching bounding boxes are then given as output.

    Keyword Arguements:        
        -> objectivness_limit -- (default, 0.7) | objetivness threshold for selected object 

    Methods:
        -> call    -- Binding to object call method
        -> train   -- Training method, used to train network
        -> anchors -- Used to generate anchors given coord
    """
    def __init__(self, **kwargs):
        self.objectness_limit = kwargs.get('objectness_limit', 0.7)

        self.base = keras.layers.Conv2D(
            512,
            (3, 3),
            padding='same',
            activation='relu',
            kernel_initializer='normal'
        )
        self.classifier = keras.layers.Conv2D(
            9,
            (1, 1),
            activation='sigmoid',
            kernel_initializer='uniform',
        )
        self.regressor = keras.layers.Conv2D(
            9 * 4,
            (1, 1),
            activation='linear',
            kernel_initializer='zero',
        )

    def call(self, data):
        data = self.base(data)

        # sliding window predictor
        for anchor in self.anchors(np.ndindex(data)):
            objectness = self.classifier(anchor, data)
            if objectness > 0.7:
                regressed_box = self.regressor(anchor, data)
                if regressed_box[2] > 10 and regressed_box[3] > 10:
                    yield regressed_box, objectness

    def train(self, labels, data):
        # for anchor in prediction:
        #     with tf.GraidentTape() as tape: 
        #         loss = ()
        pass


    def anchors(self, point):
        for aspect in [(2, 1), (1, 1), (1, 2)]:
            for scale in [128, 192, 256]:
                yield (point[0], point[1], aspect[0]*scale, aspect[1]*scale)

class Detector(keras.models.Model):
    """
    ##################
    #### Detector ####
    ##################

    This network is designed to take the last convolutional
    layer from a VGG16 pretrained network and a Region Propositinal
    Network. 

    This network is designed to take coordinates from an RPN and
    classify the content in the given coordinates upon a convolutional
    layer given by a VGG16 network. This is where the bulk of the
    Faster R-CNN's computational power is used.

    Keyword Arguments:
        ->

    Methods:
        -> call    -- Binding to object call method
        -> train   -- Training method, used to train network
    """
    def __init__(self):
        pass

    def call(self, data):
        pass
    
    def train(self, labels, data):
        pass

class Faster_RCNN(keras.models.Model):
    """
    ######################################################
    #### Faster Regional Convolutional Neural Network ####
    ######################################################
    
    This network is a deep learning wrapper for a VGG16, RPN
    and a detector nerual network. Together these are used to
    generate bounding boxes on input images.

    The image is first put through a VGG16 pretrained CNN.
    The final convolution of which is fed through the RPN
    which returns a serise of bounding boxes. Each of these
    boxes is then classified by the detector network.

    Keyword Arguments:
        ->

    Methods:
        -> call    -- Binding to object call method
        -> train   -- Training method, used to train network
    """
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