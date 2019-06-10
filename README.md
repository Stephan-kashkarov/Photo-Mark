# photo-mark
A group survellance and recognition application in the form of class roll taking

## Contents
Due to setting my expectations too high (as per usual) I did not finish my implementation of a Faster RCNN. The implementation is partially complete but all the logic is present. To make up for this lack of code I have provided some demo code in `Demo.py`

An explanation of the operation of the code itself can be found in the class docstrings within `models.py`

### Libraries
The following libraries where used in development:
    -> Tensorflow 1.13 
    -> Keras 2.2.4
    -> Numpy 1.16.3
    -> Opencv 4.1.0.25

The code is written mostly under the Keras ML framework. This framework is essentially a high level wrapper for tensorflow which makes code really readable. I used the keras.models.Model base class to implement my own three models as seen in `models.py`. This descition was inspired by the right code box on https://www.tensorflow.org/overview/. 

The tensorflow part of the code whilst not being too visible does all the heavy lifting for the program. It calculates the gradients from the loss of my layers from gradient descent (useing the adam optimiser) which it then changes weights to train the network.

## Setup
To setup this project first the requirements are to be installed through pip.
use the following command in terminal to install all required libraries: (tested in python 3.7.3)

```pip install -r requirements.txt```

once this is done the demo can be run with:

```python Demo.py```

### Demo Description

*Warning* This process might take a long time as the model is rather large (even though its downscaled from the final system) please have patience. It is possible to decrease training time by decreasing the ammount of epochs used in training (line 46).

If you wish to not try out your trained network with your own choice of mnist numbers(how exciting) i would reccoment running the file in interactive move as seen below:

```python -i Demo.py```

This will save your trained model for more testing.

Now to the important part, What is the demo actually doing? dosent look like much. The demo takes a widely used dataset called MNIST which is a massive collection of hand written digits which it then predicts. The process at the begining is the network repeatedly runing over a larger portion of the data to slowly improve the predictions of the model (as discussed before). This new model is then tested on a smaller section it has never seen to see how accurate the network really is. THis is the number that gets output at the end.

## References & Inspirations

This work would not be possible if it was not for (the original paper on faster RCNNs)[https://arxiv.org/pdf/1506.01497.pdf].

Places code was directly ctrl-c ctrl-ved from:
 - https://www.tensorflow.org/overview/ (for quick example code & inspiration)
 - https://stackoverflow.com/ (various places)
 - https://github.com/jinfagang/keras_frcnn/blob/master/keras_frcnn/vgg.py (for the cls and reg layers in RPN (didnt work tho lol))

