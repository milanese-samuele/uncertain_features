import numpy as np
import keras
from keras import layers, models
import keras_uncertainty as ku


'''
Build ConvNet, inspired by example on Keras website, with Hyperparameters
found by random search with keras tuner
'''
def build_convnet (input_shape, nclasses):
    ## HYPER PARAMETERS CONVNET
    NLAYERS = 2
    CONVUNITS = [64, 96]
    DROP_RATE = 0.3
    inputs = keras.Input(name='inputs', shape=input_shape)
    x = inputs
    for i in range(NLAYERS):
        x = layers.Conv2D(
            CONVUNITS [i],
            kernel_size = (3, 3),
            activation="relu",
            name = f'Conv_{i}'
        )(x)
        x = layers.MaxPooling2D(name = f'pooling_{i}', pool_size=(2,2))(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(DROP_RATE,
                       name='dropout')(x)
    outputs = layers.Dense(nclasses, activation="softmax", name='outputs')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='convnet')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

'''
Build simple multi layer perceptron
'''
def build_mlp (input_shape, nclasses):
    DROP_RATE = 0.3
    inputs = keras.Input(name='inputs', shape=input_shape)
    x = inputs
    for i in range (2):
        x = layers.Dense (8, activation="relu", name=f'fully_connected_{i}') (x)
        x = layers.Dropout (DROP_RATE, name=f'dropout_{i}') (x)
    x = layers.Flatten (name='flatten') (x)
    outputs = layers.Dense(nclasses, activation="softmax", name='outputs')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='simple_mlp')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
