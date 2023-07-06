from keras.losses import binary_crossentropy
import tensorflow as tf
import keras.backend as K

def dice_loss(smooth=1e-7):

    def loss(y_true, y_pred):
        smooth = K.epsilon()
        tp = y_true * y_pred
        fp = (1 - y_true) * y_pred
        fn = y_true * (1 - y_pred)

        nominator = 2 * tp + smooth
        denominator = 2 * tp + fp + fn + smooth

        dc = nominator / (denominator + 1e-8)
        dc = K.mean(dc)

        return -dc

    return loss

def dice_coef(y_true, y_pred):

    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(smooth=1e-7):
    def loss(y_true, y_pred):
        loss_val = binary_crossentropy(y_true, y_pred) + dice_loss(smooth=smooth)(y_true, y_pred)
        return 0.5 * loss_val
    return loss

def tversky_loss(alpha):
    def loss(y_true, y_pred):
        smooth = K.epsilon()
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
#         return loss
        return 1-tf.reduce_mean(tversky_index)
    return loss

def bce_tversky_loss(alpha):
    def loss(y_true, y_pred):
        smooth = K.epsilon()
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
#         return loss
        return (1-tf.reduce_mean(tversky_index)) + tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return loss

def focal_tversky_loss(alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-7):

    def loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        tp = K.sum((y_true_f * y_pred_f))
        fp = K.sum(((1-y_true_f) * y_pred_f))
        fn = K.sum((y_true_f * (1-y_pred_f)))

        tversky = (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)
        focal_tversky = K.pow((1 - tversky), gamma)

        return focal_tversky

    return loss
