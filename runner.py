import numpy as np
import configparser as cp
import csv
import keras
import keras_uncertainty
import sys
import time
## local files
import utils
import transfer
import pretraining

"""
selects pre_training function and filename according to mode chosen
"""
def select_mode (mode):
    if mode == 0:
        return (pretraining.full_pre_training, 'comparison')
    else:
        return (pretraining.partial_pre_training, 'partial')

"""
implementation of the main loop of the experiment
"""
def run_experiment (fwdp, nsamples, nreps, mode):
    import csv

    ## initializations
    mode_fn, mode_name = select_mode (mode)
    filename = f'./data/{mode_name}{fwdp}'
    timestart = time.perf_counter ()

    for i in range (nreps):
        print (f'repetition: {i}')

        ## pre-training phase
        base_models, uncertain_models, ensembles = mode_fn (fwdp)
        base_models = [transfer.make_point_extractor (m) for m in base_models]
        uncertain_models = [transfer.make_uncertain_extractor (m) for m in uncertain_models]
        ensembles = [transfer.make_ensemble_extractor (m) for m in ensembles]

        ## Base models
        for m in base_models:
            m.summary ()
            acc, cerr, _ = transfer.eval_tl (m, samples=nsamples)
            write_csv ([m.name, acc, cerr], filename)

        keras.backend.clear_session ()
        ## BNN
        for m in uncertain_models:
            m.model.summary ()
            acc, cerr, _ = transfer.eval_tl (m, fwd_passes = fwdp, samples=nsamples)
            write_csv ([m.model.name, acc, cerr], filename)

        keras.backend.clear_session ()
        ## ensembles
        for m in ensembles:
            acc, cerr, _ = transfer.eval_tl (m, fwd_passes = fwdp, samples=nsamples)
            write_csv ([f'ens_{m.name}', acc, cerr], filename)

        keras.backend.clear_session ()
        print (f'repetition {i} took {time.perf_counter() - timestart:0.4f} seconds')

    print ("Experiment completed!")

"""
helper function that writes to csv file
"""
def write_csv (row, filename):
    with open (filename, 'a+', newline='') as f:
        writer = csv.writer (f)
        writer.writerow (row)

'''
Read experiment configs from confinf file in command line arguments
'''
def read_config (filename):
    config = cp.ConfigParser ()
    config.read (filename)
    return config


def main (argv):
    exp_confs = read_config (argv [0])
    fwd_passes = int (exp_confs ['experiment_settings'] ['fwd_passes'])
    num_samples = int (exp_confs ['experiment_settings'] ['num_samples'])
    num_samples = None if num_samples == 0 else num_samples
    num_reps = int (exp_confs ['experiment_settings'] ['num_reps'])
    mode =  int (exp_confs ['experiment_settings'] ['mode'])
    run_experiment (fwd_passes, num_samples, num_reps, mode)

if __name__ == '__main__':
    main (sys.argv [1:])
