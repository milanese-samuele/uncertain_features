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

pretraining_mode = [(pretraining.full_prep, "comparison"), # mode 0 = uq comparison
                    (pretraining.partial_prep, "partial"), # mode 1 = partial uq
                    (pretraining.full_prep, "sampling")] # mode 2 sampling


"""
implementation of the main loop of the experiment
"""
def run_experiment (fwdp, nsamples, nreps, mode, target_model):
    import csv
    AUG = 10 # data augmentation factor

    ## initializations
    mode_fn, mode_name = pretraining_mode [mode]
    filename = f'./data/{mode_name}{fwdp}.csv'
    timestart = time.perf_counter ()


    X, Y, val_x, val_y, input_shape, nclasses = utils.load_data ()

    for i in range (nreps):
        print (f'repetition: {i}')

        ## pre-training phase
        base_models, uncertain_models, ensembles = mode_fn (fwdp//2) ## ensembles half the number of forward passes

        ## Base models
        for m in base_models:
            print("base models")
            m = pretraining.train_model (m, X, Y, val_x, val_y)
            m = transfer.make_point_extractor (m)
            if mode == 2:
                acc, cerr, _ = transfer.eval_tl (m, samples=nsamples, aug_test=AUG)
            else:
                if target_model == 'svm':
                    acc, cerr, _ = transfer.eval_tl (m, samples=nsamples)
                else :
                    acc, cerr, _ = transfer.transfer_network (m)
            write_csv ([m.name, acc, cerr], filename)

            keras.backend.clear_session ()
        ## BNN
        for m in uncertain_models:
            print("uncertain models")
            m = pretraining.train_model (m, X, Y, val_x, val_y)
            m = transfer.make_uncertain_extractor (m)
            if mode == 2:
                acc, cerr, _ = transfer.sampling_tl (m, fwd_passes=fwdp, samples=nsamples, aug_reps=AUG)
            else:
                if target_model == 'svm':
                    acc, cerr, _ = transfer.eval_tl (m, fwd_passes = fwdp, samples=nsamples)
                else :
                    acc, cerr, _ = transfer.transfer_network (m, fwp)
            write_csv ([m.model.name, acc, cerr], filename)

            keras.backend.clear_session ()
        ## ensembles
        for m in ensembles:
            print("ensembles")
            m = pretraining.train_model (m, X, Y, val_x, val_y)
            m = transfer.make_ensemble_extractor (m)
            if mode == 2:
                acc, cerr, _ = transfer.sampling_tl (m, fwd_passes=fwdp, samples=nsamples, aug_reps=AUG)
            else:
                if target_model == 'svm':
                    acc, cerr, _ = transfer.eval_tl (m, fwd_passes = fwdp, samples=nsamples)
                else :
                    acc, cerr, _ = transfer.transfer_network (m, fwp)
            write_csv ([f'ens_{m.test_estimators[0].model.name}', acc, cerr], filename)

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
    target_model= str (exp_confs ['experiment_settings'] ['target_model'])
    run_experiment (fwd_passes, num_samples, num_reps, mode, target_model)

if __name__ == '__main__':
    main (sys.argv [1:])
