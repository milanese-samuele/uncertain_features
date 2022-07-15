import configparser as cp
import sys

## GET INPUT FROM USER
fwd_passes = int (input ("number of forward passes for uq methods: "))
num_samples = int (input ("number of samples from target dataset: "))
num_reps = int (input ("number of repetitions per experiment: "))

mode_selection = int (input ("select mode 0: comparison; 1: partial: "))
mode = mode_selection if mode_selection == 1 else 0

## BUILD CONFIG FILE
config_file = cp.ConfigParser ()

config_file ['experiment_settings']={
    "fwd_passes" : fwd_passes,
    "num_samples": num_samples,
    "num_reps"   : num_reps,
    "mode"       : mode,
}

## WRITE FILE
dir_name = 'configs/'
filename = sys.argv [1]
with open (f'{dir_name}{filename}.ini', 'w') as conf:
    config_file.write (conf)
    conf.flush ()
