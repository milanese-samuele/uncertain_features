import numpy as np
import keras
from keras import layers, models
import keras_uncertainty as ku
'''
Build Stochastic Dropout Model from a base model
'''
def build_stochastic_dropout (base_model):
    layer_dict = {
        layers.Dropout : lambda configs : ku.layers.StochasticDropout (configs ['rate']),
    }
    return translate (base_model, layer_dict, f'mc_dropout_{base_model.name}')


'''
Build DropConnect model from base model
'''
def build_dropconnect (base_model, prob=0.11):
    layer_dict = {
        layers.Conv2D : lambda configs : ku.layers.DropConnectConv2D (configs ['filters'],
                                                                      kernel_size = configs ['kernel_size'],
                                                                      activation = configs ['activation'],
                                                                      prob = prob,
                                                                      use_learning_phase=True),
        layers.Dense : lambda configs : ku.layers.DropConnectDense (configs ['units'],
                                                                    activation = configs ['activation'],
                                                                    prob = prob,
                                                                    use_learning_phase=True),
        layers.Dropout : lambda configs : ku.layers.StochasticDropout (configs ['rate']),
    }
    return translate (base_model, layer_dict, f'mc_dropconnect_{base_model.name}')

def build_flipout(base_model, nsamples):
    num_batches = nsamples/32
    kl_weight = 1.0/num_batches
    layer_dict = {
        layers.Conv2D : lambda configs : ku.layers.FlipoutConv2D (configs ['filters'],
                                                                      configs ['kernel_size'],
                                                                      kl_weight,
                                                                      activation = configs ['activation']),
        layers.Dense : lambda configs : ku.layers.FlipoutDense (configs ['units'],
                                                                    kl_weight,
                                                                    activation = configs ['activation']),
        layers.Dropout : lambda configs : ku.layers.StochasticDropout (configs ['rate']),
        layers.InputLayer : lambda configs : ku.backend.layers.Input (shape=configs ['batch_input_shape']),
    }
    return translate (base_model, layer_dict, f'flipout_{base_model.name}')

def translate (base_model, layer_dict, name):
    uq_model = models.Sequential (name = name)
    for layer in base_model.layers:
        clazz = layer.__class__
        config = layer.get_config ()
        if clazz in layer_dict.keys ():
            uq_model.add (layer_dict [clazz] (config))
        else:
            uq_model.add (clazz.from_config (config))

    uq_model.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
    return uq_model

'''
Build deep ensemble from a function that generates the base_model instances
'''
def build_deepensemble (base_fn, input_shape, nclasses, n_ensembles):
    gencnn = lambda : base_fn (input_shape, nclasses)
    model = ku.models.DeepEnsembleClassifier (gencnn, n_ensembles)
    return model
