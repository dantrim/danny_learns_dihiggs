#!/usr/bin/env python3

from train import DataScaler, Sample, floatify

import argparse
import sys
import os

# h5py
import h5py

# keras
from keras.models import Model
import keras

# numpy
import numpy as np
seed = 347
np.random.seed(seed)

def load_input_file(args) :

    if not os.path.isfile(args.input) :
        print("ERROR provided input file (={}) is not found or is not a regular file".format(args.input))
        sys.exit()

    samples_group_name = "samples"
    scaling_group_name = "scaling"
    scaling_data_name = "scaling_data"

    samples = []
    data_scaler = None

    with h5py.File(args.input, 'r', libver = 'latest') as input_file :

        # look up the scaling first
        if scaling_group_name in input_file :
            scaling_group = input_file[scaling_group_name]
            scaling_dataset = scaling_group[scaling_data_name]
            data_scaler = DataScaler( scaling_dataset = scaling_dataset, ignore_features = ['eventweight'] )
            print("DataScaler found {} features to use as inputs (there were {} total features in the input)".format( len(data_scaler.feature_list()), len(data_scaler.raw_feature_list())))
        else :
            print("scaling group (={}) not found in file".format(scaling_group_name))
            sys.exit()

        # build the samples
        if samples_group_name in input_file :
            sample_group = input_file[samples_group_name]
            for p in sample_group :
                process_group = sample_group[p]
                class_label = process_group.attrs['training_label']
                s = Sample(name = p, class_label = int(class_label),
                    input_data = floatify( process_group['validation_features'][tuple(data_scaler.feature_list())], data_scaler.feature_list()))
                samples.append(s)
        else :
            print("samples group (={}) not found in file".format(samples_group_name))
            sys.exit()

    return samples, data_scaler
    

def main() :

    parser = argparse.ArgumentParser(description = "Run validation plots over a trained model")
    parser.add_argument("-i", "--input",
        help = "Provide input, pre-processed HDF5 files with validation and scaling data",
        required = True)
    parser.add_argument("--outdir", help = "Provide an output directory for plots", default = ".")
    parser.add_argument("-n", "--name", help = "Provide an output filename suffix", default = "")
    parser.add_argument("-v", "--verbose", action = "store_true", default = False, help = "Be loud about it")
    args = parser.parse_args()

    validation_samples, data_scaler = load_input_file(args)

if __name__ == "__main__" :
    main()
