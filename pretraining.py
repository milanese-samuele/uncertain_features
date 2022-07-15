import keras
import keras_uncertainty
import numpy as np

import utils
import base_models
import uqmethods
import partials

'''
pre-train all the architectures
'''
def full_pre_training (n_ens):
    x_train, y_train, x_test, y_test, input_shape, nclasses = utils.load_data ()
    ## build architectures
    archs = [base_models.build_convnet (input_shape, nclasses),
              base_models.build_mlp (input_shape, nclasses)]
    ## build uq
    base = lambda x : x
    methods = [uqmethods.build_dropconnect,
               uqmethods.build_stochastic_dropout]

    uncertain_models = []
    for arch in archs:
        uncertain_models += [method (arch) for method in methods]

    ## add ensembles
    ensembles = [uqmethods.build_deepensemble (base_models.build_convnet, input_shape, nclasses, n_ens),
                 uqmethods.build_deepensemble (base_models.build_mlp, input_shape, nclasses, n_ens)]
    ## train
    train_fn = lambda model : base_models.train_model (model,
                                                       x_train, y_train,
                                                       x_test, y_test,)

    archs = [train_fn (m) for m in archs]
    uncertain_models = [train_fn (m) for m in uncertain_models]
    ensembles = [train_fn (e) for e in ensembles]
    return archs, uncertain_models, ensembles

"""
pretrain partial models
"""
def partial_pre_training (_):

    DROP_RATE = 0.12
    DEPTH = 3

    x_train, y_train, x_test, y_test, input_shape, nclasses = utils.load_data ()

    archs = [partials.build_dropconnect (input_shape, nclasses, "dropconnect_0", 0),
             partials.build_dropout (input_shape, nclasses, "dropout_0", 0),]

    uncertain_models = [partials.build_dropconnect (input_shape, nclasses, f'dropconnect_{k}', k) for k in range (1,DEPTH+1)]
    uncertain_models += [partials.build_dropout (input_shape, nclasses, f'dropout_{k}', k) for k in range (1,DEPTH+1)]

    ## train
    train_fn = lambda model : base_models.train_model (model,
                                                       x_train, y_train,
                                                       x_test, y_test,)

    archs = [train_fn (m) for m in archs]
    uncertain_models = [train_fn (m) for m in uncertain_models]
    return archs, uncertain_models, []
