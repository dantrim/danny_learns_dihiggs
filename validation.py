#!/usr/bin/env python3

from train import DataScaler, Sample, floatify, build_combined_input

import argparse
import sys
import os

# h5py
import h5py

# keras
from keras.models import Model
import keras

# plotting
import matplotlib.pyplot as plt

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

def load_model(args) :

    print("Loading model architecture and weights ({arch}, {weights})".format(arch=args.arch, weights=args.weights))
    from keras.models import model_from_json
    json_file = open(args.arch, 'r')
    loaded_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model)
    loaded_model.load_weights(args.weights)
    return loaded_model

def valid_idx(input_array) :
    valid_lo = (input_array > -np.inf)
    valid_hi = (input_array < np.inf)
    valid = (valid_lo & valid_hi)
    return valid

def build_discriminant(scores_dict, labels) :

    discriminant_dict = {}

    for label in scores_dict :
        numerator_scores = scores_dict[label][:,label]
        denominator_scores = np.empty(numerator_scores.shape, dtype=numerator_scores.dtype)
        for clabel in labels :
            if clabel == label : continue # skip numerator
#            if denominator_scores == None :
            if not denominator_scores.any() :
                denominator_scores = scores_dict[clabel]
            else :
                denominator_scores += scores_dict[clabel]
        log_ratio = np.log(numerator_scores / denominator_scores)
        print("log ratio {} = {}".format(label, log_ratio.shape))
        idx = valid_idx(log_ratio)
        discriminant_dict[label] = log_ratio[idx]

    return discriminant_dict

def make_nn_output_plots( model = None, inputs = None, samples = None, targets = None) :

    # set of scores for each label: shape = (n_samples, n_outputs)
    nn_scores = model.predict(inputs)

    class_labels = set(targets)
    targets_list = list(targets)
    nn_scores_dict = {}

    # index the sample names by their class label
    names = {}
    for sample in samples :
        names[sample.class_label()] = sample.name()

    # break up the predicted scores by the class label
    for ilabel, label in enumerate(class_labels) :
        # left-most appearance of the label
        left = targets_list.index(label)
        # right-most appearance of the label
        right = len(targets_list) - 1 - targets_list[::-1].index(label)
        nn_scores_dict[label] = nn_scores[left:right+1]

    # start plotting
    for label in class_labels :
        fig, ax = plt.subplots(1,1)
        ax.grid(color='k', which='both', linestyle='--', lw=0.5, alpha=0.1, zorder = 0)
        ax.set_xlabel( "NN output for label {}".format(names[label]), horizontalalignment='right', x=1)
        #ax.set_xlim([1e-2,1.0])
        ax.set_xlim([0,1])
        ax.set_yscale('log')
        binning = np.arange(0,1,0.02)
        centers = (binning[1:-2] + binning[2:-1])/2
        for sample_label in nn_scores_dict :
            sample_scores_for_label = nn_scores_dict[sample_label][:,label]

            yields, _ = np.histogram(sample_scores_for_label, bins = binning)
            yields = yields/yields.sum()
            ax.step(centers, yields[1:-1], label = names[sample_label], where = 'mid')
            
            #ax.hist(sample_scores_for_label, bins = binning, alpha = 0.3, label = names[sample_label], density = True)
        ax.legend(loc='best', frameon = False)
        fig.savefig("test_nn_output_class{}.pdf".format(label), bbox_inches='tight', dpi = 200)

def build_discriminants(scores = None, labels = [], targets_list = None) :

    # get the NN output for each discriminant
    nn_dict = {}
    for ilabel, label in enumerate(labels) :
        nn_dict[label] = scores[:,label]

    idx_dict = {}
    for ilabel, label in enumerate(labels) :
        # left-most
        left = targets_list.index(label)
        # right-most
        right = len(targets_list) - 1 - targets_list[::-1].index(label)
        idx_dict[label] = [left,right+1]

    disc_dict = {}
    idx_dict = {}
    for ilabel, label in enumerate(labels) :
        num_scores = nn_dict[label]
        den_scores = np.empty(num_scores.shape, dtype=num_scores.dtype)
        for dlabel in labels :
            if dlabel == label : continue # skip numerator
            if not den_scores.any() :
                den_scores = nn_dict[dlabel]
            else :
                den_scores += nn_dict[dlabel]
        log_ratio = np.log( num_scores / den_scores )
        disc_dict[label] = log_ratio
    return disc_dict

def make_discriminant_plots( model = None, inputs = None, samples = None, targets = None ) :

    nn_scores = model.predict(inputs) 
    class_labels = set(targets)
    targets_list = list(targets)

    # index the sample names by their class label
    names = {}
    for sample in samples :
        names[sample.class_label()] = sample.name()

    discriminants = build_discriminants(scores = nn_scores, labels = class_labels, targets_list = targets_list)

    idx_map = {}
    for ilabel, label in enumerate(class_labels) :
        # left-most
        left = targets_list.index(label)
        # right-most
        right = len(targets_list) - 1 - targets_list[::-1].index(label)
        idx_map[label] = [left,right+1]

    for label in class_labels :
        fig, ax = plt.subplots(1,1)
#        ax.set_xlim([-40,15])
#        ax.set_ylim([1e-2,2])
        binning = np.arange(-40,12,1)
        centers = (binning[1:-2] + binning[2:-1])/2
        ax.set_yscale('log')

        for sample_label in discriminants :
            left, right = idx_map[sample_label][0], idx_map[sample_label][1]
            disc_scores_for_sample = discriminants[label][left:right]

            # since we took the log_ratio, lets clear out any invalid numbers
            ok_idx = valid_idx(disc_scores_for_sample)
            disc_scores_for_sample = disc_scores_for_sample[ok_idx]

            yields, _ = np.histogram(disc_scores_for_sample, bins = binning)
            yields = yields / yields.sum()
            ax.step(centers, yields[1:-1], label = names[sample_label], where='mid')
        ax.legend(loc = 'best', frameon = False)

        savename = "test_nn_disc_class{}.pdf".format(label)
        fig.savefig(savename, bbox_inches = 'tight', dpi = 200)

def main() :

    parser = argparse.ArgumentParser(description = "Run validation plots over a trained model")
    parser.add_argument("-i", "--input",
        help = "Provide input, pre-processed HDF5 files with validation and scaling data",
        required = True)
    parser.add_argument("-w", "--weights", help = "Provide the NN weights file", required = True)
    parser.add_argument("-a", "--arch", help = "Provide the NN architecture file", required = True)
    parser.add_argument("--outdir", help = "Provide an output directory for plots", default = ".")
    parser.add_argument("-n", "--name", help = "Provide an output filename suffix", default = "")
    parser.add_argument("-v", "--verbose", action = "store_true", default = False, help = "Be loud about it")
    args = parser.parse_args()

    model = load_model(args)
    validation_samples, data_scaler = load_input_file(args)
    input_features, targets = build_combined_input(validation_samples, data_scaler = data_scaler, scale = True)

    make_nn_output_plots( model, samples = validation_samples, inputs = input_features, targets = targets )
    make_discriminant_plots( model, samples = validation_samples, inputs = input_features, targets = targets )
    
    print("done")


if __name__ == "__main__" :
    main()
