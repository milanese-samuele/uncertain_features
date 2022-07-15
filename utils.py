import numpy as np
import random
import keras_uncertainty
from keras_uncertainty.utils.calibration import classifier_calibration_error, classifier_calibration_curve
from keras_uncertainty.utils.numpy_metrics import accuracy
import keras

## Expected Calibration error
def calib_error (preds, domain, bins = 10):
    y_confs = np.max (preds, axis=1)
    y_preds = np.argmax (preds, axis=1)
    y_true = np.argmax (domain, axis=1)
    return classifier_calibration_error (y_preds, y_true, y_confs, num_bins=bins, weighted=True), classifier_calibration_curve (y_preds, y_true, y_confs, num_bins=bins)

## accuracy
def score (preds, domain):
    y_pred = np.argmax (preds, axis = 1)
    y_true = np.argmax (domain, axis = 1)
    acc = accuracy (y_true, y_pred)
    # accuracy = len(np.where(pred_classes==correct_classes)[0])/len(correct_classes)
    return acc


'''
Load MNIST dataset for training models
'''
def load_data ():
    import tensorflow as tf
    nclasses = 10
    input_shape = (28,28,1)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") /255
    x_train = np.expand_dims(x_train, -1)
    x_test = x_test.astype("float32") /255
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, nclasses)
    y_test = keras.utils.to_categorical(y_test, nclasses)
    return x_train, y_train, x_test, y_test, input_shape, nclasses

def load_fashion_data (resize = None, aug_test = None):
    import tensorflow as tf
    nclasses = 10
    input_shape = (28,28,1)
    test_resize = resize if aug_test is None else resize * aug_test


    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") /255
    x_train = np.expand_dims(x_train, -1)
    x_test = x_test.astype("float32") /255
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, nclasses)
    y_test = keras.utils.to_categorical(y_test, nclasses)
    if resize is not None:
        train_idx = np.random.choice (len (x_train), resize, replace=False)
        test_idx = np.random.choice (len (x_test), test_resize, replace=False)
        x_train = x_train [train_idx,:,:]
        y_train = y_train [train_idx,:]
        x_test = x_test [test_idx,:,:]
        y_test = y_test [test_idx,:]
    return x_train, y_train, x_test, y_test, input_shape, nclasses
