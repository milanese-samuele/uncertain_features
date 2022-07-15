import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import keras_uncertainty
import keras

import utils

"""
discards last layer (s) to make a feature extractor
"""
def make_uncertain_extractor (estimator, offset = 2):
    return keras_uncertainty.models.StochasticRegressor (keras.models.Model (estimator.input,
                                                                             estimator.layers [-offset].output,
                                                                             name=f'{estimator.name}_extractor'))

def make_point_extractor (estimator, offset = 2):
    return keras.models.Model (estimator.input,
                               estimator.layers [-offset].output,
                               name=f'{estimator.name}_extractor')

def make_ensemble_extractor (ensemble, offset = 2):
    estimators = []
    for est in ensemble.test_estimators:
        estimators.append (make_uncertain_extractor (est))
    return keras_uncertainty.models.DeepEnsembleRegressor (models = estimators)

def reduce_dims (train_feats, test_feats, var = 0.95):
    ## scaling features
    scaler = StandardScaler ()
    scaler.fit (train_feats)
    train_feats = scaler.transform (train_feats)
    test_feats = scaler.transform (test_feats)
    ## reducing dimensionality of features
    pca = PCA (var)
    pca.fit (train_feats)
    train_feats = pca.transform (train_feats)
    test_feats = pca.transform (test_feats)
    return train_feats, test_feats

def eval_tl (extractor, fwd_passes = None, samples = 5000, aug_test = None):
    if aug_test is None:
        X, Y, tests, domain, ishape, nclasses = utils.load_fashion_data (samples)
    else:
        X, Y, tests, domain, ishape, nclasses = utils.load_fashion_data (samples, aug_test)

    if extractor.__class__ is keras_uncertainty.models.DeepEnsembleRegressor:
        features = lambda X : extractor.predict (X) [0]
    elif fwd_passes is None :
        features = lambda X : extractor.predict (X)
    else:
        features = lambda X : extractor.predict (X, fwd_passes) [0]
    clf = svm.SVC (probability=True, cache_size=1000)
    print (f'fitting svm on features')
    train_feats, test_feats = reduce_dims (features (X), features (tests), var = 0.95)
    print (f'train_feats shape={train_feats.shape}')
    print (f'test_feats shape={test_feats.shape}')
    clf.fit (train_feats, np.argmax (Y, axis = 1))
    print ('svm fit')
    preds = clf.predict_proba (test_feats)
    acc = utils.score (preds, domain)
    cerr, cplot = utils.calib_error (preds, domain)
    return acc, cerr, cplot

def sampling_tl (extractor, fwd_passes = None, samples = 5000, aug_reps = None):
    if aug_reps is None:
        aug_reps = fwd_passes
    X, Y, tests, domain, ishape, nclasses = utils.load_fashion_data (samples, aug_reps+1)
    Y = np.argmax (Y, axis = 1)
    if extractor.__class__ is keras_uncertainty.models.DeepEnsembleRegressor:
        features = lambda X : extractor.predict (X)
    elif fwd_passes is None :
        features = lambda X : extractor.predict (X)
    else:
        features = lambda X : extractor.predict (X, fwd_passes)
    clf = svm.SVC (probability=True)
    print (f'fitting svm on features')
    if aug_reps is not None:
        Y = np.tile (Y, aug_reps + 1)
        train_mean, train_std = features (X)
        test_mean, test_std = features (tests)
        train_feats = train_mean
        test_feats = test_mean
        for i in range (aug_reps):
            train_feats = np.vstack ((train_feats, [np.random.normal (train_mean [i], train_std [i]) for i in range (len (train_mean))]))
    else:
        train_feats = features (X)
        train_feats = features (tests)

    train_feats, test_feats = reduce_dims (train_feats, test_feats, var = 0.9)
    print (f'train_feats shape={train_feats.shape}')
    print (f'test_feats shape={test_feats.shape}')
    clf.fit (train_feats, Y)
    print ('svm fit')
    preds = clf.predict_proba (test_feats)
    acc = utils.score (preds, domain)
    cerr, cplot = utils.calib_error (preds, domain)
    return acc, cerr, cplot
