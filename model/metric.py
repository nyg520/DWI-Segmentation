import numpy as np

def sensitivity(y_true, y_pred):
    if np.sum(y_true) == 0:
        return np.nan
    tp = np.sum(y_true * y_pred)
    tp_fn = np.sum(y_true)

    stv = tp / tp_fn

    return stv

def specificity(y_true, y_pred):
    if np.sum(y_pred) == 0:
        return np.nan
    tn = np.sum((1-y_true) * (1-y_pred))
    tn_fp = np.sum((1-y_true))

    spc = tn / tn_fp

    return spc

def precision(y_true, y_pred):
    if np.sum(y_pred) == 0:
        return np.nan
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1-y_true) * y_pred)

    prc = tp / (tp + fp)

    return prc

def IOU(y_true, y_pred):
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return np.nan
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)

    return intersection / union

def dice_coef(y_true, y_pred):

    epsilon = 1e-7
    intersection = np.sum(y_true * y_pred)

    return (2. * intersection + epsilon) / (np.sum(y_true) + np.sum(y_pred) + epsilon)