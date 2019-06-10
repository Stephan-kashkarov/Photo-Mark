import keras
import numpy as np

def smooth_l1_loss(label, pred, anchor):
    """
    Smooth L1 loss for bbox regression

    This loss calculates how off the anchors are
    from the correct anchor.

    It is used to calculate new weights for the 
    network.
    """
    label[0] = (label[0] - anchor[0])/anchor[2]
    label[1] = (label[1] - anchor[1])/anchor[3]
    label[2] = np.log10(label[2]/anchor[2])
    label[2] = np.log10(label[3]/anchor[3])

    pred[0] = (pred[0] - anchor[0])/anchor[2]
    pred[1] = (pred[1] - anchor[1])/anchor[3]
    pred[2] = np.log10(pred[2]/anchor[2])
    pred[2] = np.log10(pred[3]/anchor[3])

    return np.sum([pred[i]-label[i] for i in range(4)])