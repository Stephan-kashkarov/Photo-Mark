""" Models.py

classes:
-> RPN : line 15
-> Classifier : line 145
-> Faster_RCNN : 225

"""

import keras
import numpy as np
import tensorflow as tf

from losses import smooth_l1_loss
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
        -> objectness_limit -- (default, 0.7) | objetivness threshold 
                                              | for selected object 

    Methods:
        -> call    -- Binding to object call method
        -> train   -- Training method, used to train network
        -> anchors -- Used to generate anchors given coord
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.objectness_limit = kwargs.get('objectness_limit', 0.7)
        self.const = kwargs.get('loss_const', 1)

        self.optimiser = tf.train.AdamOptimizer()

        self.base = keras.layers.Conv2D(
            256,
            (3, 3),
            padding='same',
            activation='relu',
            kernel_initializer='normal'
        )

        # TODO: figure out how the reg and cls layers work
        # with variable sized inputs
        self.classifier = keras.layers.Conv2D(
            9,
            (1, 1),
            activation='sigmoid',
            kernel_initializer='uniform',
        )

        self.regressor = keras.layers.Conv2D(
            9 * 4,
            (1, 1),
            kernel_initializer='zero',
        )

    def call(self, data, training=False):
        """
        Base class func Call

        method for calling network

        args:
            -> Data | System input | dtype: tf.Tensor
            -> training | variable output bool | dtype: bool
        
        """
        data = self.base(data)

        # sliding window predictor
        for anchor in self.anchors(np.ndindex(data)):
            objectness = self.classifier(anchor, data)

            # objectness threshold
            if objectness > self.objectness_limit:
                regressed_box = self.regressor(anchor, data)

                # min size boxes
                if regressed_box[2] > 10 and regressed_box[3] > 10: # size limit
                    if not training:
                        yield regressed_box, objectness
                    else:
                        yield regressed_box, objectness, anchor
            else:
                if not training:
                    yield None, objectness
                else:
                    # returns anchor for training math
                    yield None, objectness, anchor

    def train(self, labels, data):
        """
        Base class func Train

        Function called to train layer

        Args:
            -> labels | correct output | dtype: list
            -> Data | System input | dtype: tf.Tensor
        """
        prediction = self(data, training=True)

        # Magic TF bindings for gradient decent
        with tf.GradientTape() as tape:

            # iterate anchors
            for index, pred, anchor in enumerate(prediction):

                # loss calc (see losses.py)
                cls_loss += keras.losses.binary_crossentropy(labels[index][1], pred[1])
                reg_loss += smooth_l1_loss(labels[index][0], pred[0], anchor)
            loss = (cls_loss / 256) + (self.const * (reg_loss/(9*256)))

        # calculates gradients of last iteration
        grads = tape.gradient(loss, self.trainable_weights)
        # applies said graidents (with adam)
        self.optimiser.apply_gradients(zip(grads, self.trainable_weights))


    def anchors(self, point):
        """
        RPN.anchors

        Private method for generating anchors around a single point
        """
        for aspect in [(2, 1), (1, 1), (1, 2)]:
            for scale in [128, 192, 256]:
                yield (point[0], point[1], aspect[0]*scale, aspect[1]*scale)

class Classifier(keras.models.Model):
    """
    ####################
    #### Classifier ####
    ####################

    This network is designed to take the last convolutional
    layer from a VGG16 pretrained network and a Region Propositinal
    Network. 

    This network is designed to take coordinates from an RPN and
    classify the content in the given coordinates upon a convolutional
    layer given by a VGG16 network. This is where the bulk of the
    Faster R-CNN's computational power is used.


    TODO: This classifier currently chooses either person or not
    person. However as this is a full on classifier which is used
    sparingly within the network, it could easily be expanded to do
    much more intensive classification

    Ideas:
        -> Use this to do facial recognition
        -> Use this to detect between animals and simmilar things

    Methods:
        -> call    -- Binding to object call method
        -> train   -- Training method, used to train network
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.core = [
            keras.layers.Input(shape=(28, 28, 1)),
            # Conv block 1
            keras.layers.Conv2D(128, (3, 3), activation='relu'), # retrofitted for demo (inputsize)
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.AvgPool2D(2, 2),

            # Conv block 1
            keras.layers.Conv2D(256, (3, 3), activation='relu'),
            keras.layers.Conv2D(256, (3, 3), activation='sigmoid'),
            keras.layers.AvgPool2D(2, 2),

            # Dense layer
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(10, activation='softmax') # retrofitted for demo (classifications)
        ]

    def call(self, data):
        """
        Base class func Call

        method for calling network

        args:
            -> Data | System input | dtype: tf.Tensor
        
        """
        # TODO: Figure out how to do variable input sizes
        for layer in self.core:
            data = layer(data)
        return data


    def train(self, labels, data):
        """
        Base class func Train

        Function called to train layer

        Args:
            -> labels | correct output | dtype: list
            -> Data | System input | dtype: tf.Tensor
        """
        with tf.GradientTape() as tape:
            pred = self(data)
            loss = self.loss(pred)
        grad = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.trainable_weights))

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
    boxes is then classified by the classifier network.

    Methods:
        -> call    -- Binding to object call method
        -> train   -- Training method, used to train network
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = tf.train.AdamOptimizer()

        self.vgg16 = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)

        self.rpn = RPN()
        self.classifier = Classifier()
    
    def call(self, data):
        """
        Base class func Call

        method for calling network

        args:
            -> Data | System input | dtype: tf.Tensor
        """
        # runs vgg 16
        data = self.vgg16(data)

        # predicts regions
        regions = self.rpn(data)

        # runs regions through classifier
        for region in regions:
            pred = self.classifier(region)
            if np.sum(pred) > 0:
                yield region, pred

    def train(self, labels, data):
        """
        Base class func Train

        Function called to train layer

        This method has a unique training method,
        Anchors and scores are calculated by the rpn.
        The Classifier is then run on these anchors
        using the inverse losses from the scores as
        their respective labels. 

        This method allows for both networks to be
        trained at the same time. The classifier trains
        better the better then RPN performs however.

        Args:
            -> labels | correct output | dtype: list
            -> Data | System input | dtype: tf.Tensor
        """

        with tf.GradientTape() as tape:
            pred = self.rpn(data)
            # iterate anchors
            for index, pred, anchor in enumerate(pred):

                # rpn loss part 1
                _cls = keras.losses.binary_crossentropy(labels[index][1], pred[1])
                cls_loss += _cls

                # train the Classifier
                with tf.GraidentTape() as tape2:
                    pred2 = self.classifier(anchor)
                    class_loss = keras.losses.binary_crossentropy((1 - _cls), pred2)
                
                # Classifier grad decsent
                grad = tape.gradient(class_loss, self.classifier.trainable_weights)
                self.classifier.optimizer.apply_gradients(zip(grad, self.classifier.trainable_weights))

                # RPN loss prt 2 the reckoning
                reg_loss += smooth_l1_loss(labels[index][0], pred[0], anchor)
            rpn_loss = (cls_loss / 256) + (self.rpn.const * (reg_loss/(9*256)))

            # RPN grad decsent
            grad = tape.gradient(rpn_loss, self.rpn.trainable_weights)
            self.rpn.optimizer.apply_gradients(zip(grad, self.rpn.trainable_weights))