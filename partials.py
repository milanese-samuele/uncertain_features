import numpy as np
import keras
import keras_uncertainty

"""
builds a partial version of a Dropout model
"""
def build_dropout (input_shape, nclasses, name, num_uncertain_layers = 0):

    DROP_RATE = 0.12
    DEPTH = 3

    standard_nets = [None] * (DEPTH - num_uncertain_layers)
    stochastic_dropouts = [keras_uncertainty.layers.StochasticDropout ] * num_uncertain_layers
    dropouts = np.concatenate ((standard_nets, stochastic_dropouts))

    inputs = keras.Input(name='inputs', shape=input_shape)
    x = inputs

    for i in range (DEPTH):
        x = keras.layers.Dense (8, activation="relu", name=f'dense_{i}') (x)
        if dropouts [i] is not None:
            x = dropouts [i] (DROP_RATE, name=f'stochastic_dropout_{i}') (x)
    x = keras.layers.Flatten (name='flatten') (x)
    outputs = keras.layers.Dense(nclasses, activation="softmax", name='outputs')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

"""
builds a partial version of a Dropconnect model
"""
def build_dropconnect (input_shape, nclasses, name, num_uncertain_layers = 0):

    DROP_RATE = 0.12
    DEPTH = 3

    standard_layers = ["keras.layers.Dense (8, activation=\"relu\")(x)"] * (DEPTH - num_uncertain_layers)
    dc_layers = ["keras_uncertainty.layers.DropConnectDense (8, activation='relu', prob=DROP_RATE, use_learning_phase=True)(x)" ] * num_uncertain_layers
    layerz = np.concatenate ((standard_layers, dc_layers))

    inputs = keras.Input(name='inputs', shape=input_shape)
    x = inputs
    for exp in layerz:
        x = eval (exp)
    x = keras.layers.Dropout (DROP_RATE) (x)
    x = keras.layers.Flatten (name='flatten') (x)
    outputs = keras.layers.Dense(nclasses, activation="softmax", name='outputs')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
