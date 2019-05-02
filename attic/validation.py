#!/usr/bin/env python3

from train import DataScaler, Sample, floatify, build_combined_input
from preprocess import mkdir_p, unique_filename

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

def sample_with_label(label, samples) :

    for s in samples :
        if s.class_label() == label :
            return s
    return None

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
                s.eventweights = floatify( process_group['validation_features'][tuple(['eventweight'])], ['eventweight'])
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

def make_nn_output_plots( model = None, inputs = None, samples = None, targets = None, args = None) :

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
        ax.set_xlim([-0.01,1.01])
        ax.set_yscale('log')
        binning = np.arange(0,1,0.02)
        centers = (binning[1:-2] + binning[2:-1])/2
        ax.set_xlim((centers[0]-0.1, centers[-1]+0.1)) 
        for sample_label in nn_scores_dict :
            sample_scores_for_label = nn_scores_dict[sample_label][:,label]
            sample_weights = sample_with_label(sample_label, samples).eventweights

            yields, _ = np.histogram(sample_scores_for_label, bins = binning,
                weights = sample_weights.reshape((sample_scores_for_label.shape[0],)))
            yields = yields/yields.sum()
            ax.step(centers, yields[1:-1], label = names[sample_label], where = 'mid')
            
            #ax.hist(sample_scores_for_label, bins = binning, alpha = 0.3, label = names[sample_label], density = True)
        ax.legend(loc='best', frameon = False)
        savename = "nn_outputs_{}_class_{}.pdf".format(args.name, names[label])
        if args.outdir != "" :
            mkdir_p(args.outdir)
        savename = "{}/{}".format(args.outdir, savename) 
        fig.savefig(savename, bbox_inches = 'tight', dpi = 200)

    return nn_scores

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

def make_discriminant_plots( model = None, inputs = None, samples = None, targets = None, args = None ) :

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
        binning = np.arange(-40,20,1)
        centers = (binning[1:-2] + binning[2:-1])/2
        ax.set_xlim((centers[0]-0.1, centers[-1]+0.1)) 
        ax.set_yscale('log')

        for sample_label in discriminants :
            left, right = idx_map[sample_label][0], idx_map[sample_label][1]
            disc_scores_for_sample = discriminants[label][left:right]

            # since we took the log_ratio, lets clear out any invalid numbers
            ok_idx = valid_idx(disc_scores_for_sample)
            disc_scores_for_sample = disc_scores_for_sample[ok_idx]
            sample_weights = sample_with_label(sample_label, samples).eventweights
            sample_weights = sample_weights[ok_idx]
            yields, _ = np.histogram(disc_scores_for_sample, bins = binning,
                weights = sample_weights.reshape((disc_scores_for_sample.shape[0],)))
            yields = yields / yields.sum()
            ax.step(centers, yields[1:-1], label = names[sample_label], where='mid')

        ax.legend(loc = 'best', frameon = False)

        savename = "nn_discriminant_{}_class_{}.pdf".format(args.name, names[label])
        if args.outdir != "" :
            mkdir_p(args.outdir)
        savename = "{}/{}".format(args.outdir, savename)
        fig.savefig(savename, bbox_inches = 'tight', dpi = 200)

def make_nn_roc_curve( output_scores = None, samples = [], inputs = None, targets = None, signal_class = 0, args = None) :

    class_labels = set(targets)
    targets_list = list(targets)
    nn_scores_dict = {}

    names = {}
    for sample in samples :
        names[sample.class_label()] = sample.name()

    for ilabel, label in enumerate(class_labels) :
        left = targets_list.index(label)
        right = len(targets_list) - 1 - targets_list[::-1].index(label)
        nn_scores_dict[label] = output_scores[left:right+1]

    lowbin = 0
    highbin = 1


    edges = np.concatenate(
        [[-np.inf], np.linspace(lowbin,highbin,500), [np.inf]]) 

    # we want the sample efficiency to pass the signal eff
    sample_eff = {}
    h_total = []
    w_total = []
    for label in nn_scores_dict :
        # select out the scores for class 'label' for NN output 'signal_class'
        scores = nn_scores_dict[label][:,signal_class]

        weights = sample_with_label(label, samples).eventweights
        h_nn, _ = np.histogram( scores, bins = edges, weights = weights.reshape((scores.shape[0],) ))
        if label != signal_class :
            h_total.append(h_nn)
            w_total.append(weights)

        # We want to integrate from the high end and then flip
        # to give the yield "to the right" of the value at
        # which the integration starts, since "to the right" is
        # signal like. We also normalize to give the value as
        # a relative fraction, or efficiency, of selecting that sample at the
        # given value where we integrate from.
        eff = np.cumsum( h_nn[::-1] )[::-1] / h_nn.sum()
        sample_eff[label] = eff

    summed_bkg = h_total[0]
    for h in h_total[1:] :
        summed_bkg += h
    summed_weights = w_total[0]
    for h in w_total[1:] :
        summed_weights += h
    eff_total_bkg = np.cumsum( summed_bkg[::-1] )[::-1]/summed_bkg.sum()

    signal_eff = None
    bkg_eff = {}
    for e in sample_eff :
        if e == signal_class :
            signal_eff = sample_eff[e]
        else :
            bkg_eff[e] = sample_eff[e]


    fig, ax = plt.subplots(1,1)
    for bkg_label in bkg_eff :
    
        bkg = bkg_eff[bkg_label]
        valid_rej = bkg > 0
        sig = np.array(signal_eff[:])

        valid_sig = (sig != 1.0)
        valid = valid_rej & valid_sig

        bkg = bkg[valid]
        sig = sig[valid]

        bkg_rej = 1/bkg
        ax.plot(sig, bkg_rej, label = names[bkg_label])

    valid_rej_total = eff_total_bkg > 0
    sig = np.array(signal_eff[:])
    valid_sig_total = sig != 1.0
    valid_total = valid_rej_total & valid_sig_total

    bkg_total = eff_total_bkg[valid_total]
    sig_total = sig[valid_total]
    bkg_rej_total = 1/bkg_total
    ax.plot(sig_total, bkg_rej_total, label = "Total Bkg")

    ax.set_yscale('log')
    ax.set_xlabel('$hh$ efficiency', horizontalalignment='right', x=1)
    ax.set_ylabel('Background rejection, $1/\\epsilon_{bkg}$', horizontalalignment='right', y=1)
    ax.legend(loc='best', frameon = False)

    # save
    savename = "nn_output_ROC_{}.pdf".format(args.name)
    if args.outdir != "" :
        mkdir_p(args.outdir)
    savename = "{}/{}".format(args.outdir, savename)
    fig.savefig(savename, bbox_inches = 'tight', dpi = 200)

def make_nn_disc_roc_curve( scores, samples = [], inputs = [], targets = None, signal_class = 0, args = None) :

    class_labels = set(targets)
    targets_list = list(targets)
    

    names = {}
    for sample in samples :
        names[sample.class_label()] = sample.name()

    discriminants = build_discriminants( scores = scores, labels = class_labels, targets_list = targets_list )
    idx_map = {}
    for ilabel, label in enumerate(class_labels) :
        left = targets_list.index(label)
        right = len(targets_list) - 1 - targets_list[::-1].index(label)
        idx_map[label] = [left,right+1]

    lowbin = -40
    highbin = 40
    edges = np.concatenate(
        [[-np.inf], np.linspace(lowbin,highbin,500), [np.inf]])

    sample_eff = {}

    fig, ax = plt.subplots(1,1)

    signal_eff = None
    bkg_eff = {}
    h_total = []
    w_total = []
    for label in class_labels :

        # get the discriminant for the 'signal_class' specified
        left, right = idx_map[label][0], idx_map[label][1]
        disc = discriminants[signal_class][left:right]
        weights = sample_with_label(label, samples).eventweights

        ok_idx = valid_idx(disc)
        disc = disc[ok_idx]
        weights = weights[ok_idx]

        h_d, _ = np.histogram( disc, bins = edges, weights = weights.reshape( (disc.shape[0],)) )
        eff = np.cumsum( h_d[::-1] )[::-1] / h_d.sum()
        if label == signal_class :
            signal_eff = eff
        else :
            bkg_eff[label] = eff
            h_total.append(h_d)
            w_total.append(weights)

    summed_bkg = h_total[0]
    for h in h_total[1:] :
        summed_bkg += h
    #summed_weights = w_total[0]
    #for h in w_total[1:] :
    #    summed_weights += h
    eff_total_bkg = np.cumsum( summed_bkg[::-1] )[::-1]/summed_bkg.sum()

    fig, ax = plt.subplots(1,1)
    for bkg_label in bkg_eff :
        bkg = bkg_eff[bkg_label]
        valid_rej = bkg > 0
        sig = np.array(signal_eff[:])
        valid_sig = sig != 0
        valid = valid_rej & valid_sig

        bkg = bkg[valid]
        sig = sig[valid]
        bkg_rej = 1.0 / bkg
        ax.plot(sig, bkg_rej, label = names[bkg_label])

    valid_rej_total = eff_total_bkg > 0
    sig = np.array(signal_eff[:])
    valid_sig_total = sig != 1.0
    valid_total = valid_rej_total & valid_sig_total

    bkg_total = eff_total_bkg[valid_total]
    sig_total = sig[valid_total]
    bkg_rej_total = 1/bkg_total
    ax.plot(sig_total, bkg_rej_total, label = "Total Bkg")

    ax.set_yscale('log')
    ax.set_xlabel('$hh$ efficiency', horizontalalignment = 'right', x =1)
    ax.set_ylabel('Background rejection, $1/\\epsilon_{bkg}$', horizontalalignment = 'right', y=1)
    ax.legend(loc='best', frameon = False)

    # save
    savename = "nn_output_ROC_disc_{}.pdf".format(args.name)
    if args.outdir != "" :
        savename = "{}/{}".format(args.outdir, savename)
    fig.savefig(savename, bbox_inches = 'tight', dpi = 200)

def main() :

    parser = argparse.ArgumentParser(description = "Run validation plots over a trained model")
    parser.add_argument("-i", "--input",
        help = "Provide input, pre-processed HDF5 files with validation and scaling data",
        required = True)
    parser.add_argument("-w", "--weights", help = "Provide the NN weights file", required = True)
    parser.add_argument("-a", "--arch", help = "Provide the NN architecture file", required = True)
    parser.add_argument("--outdir", help = "Provide an output directory for plots", default = "./")
    parser.add_argument("-n", "--name", help = "Provide an output filename suffix", default = "")
    parser.add_argument("-v", "--verbose", action = "store_true", default = False, help = "Be loud about it")
    args = parser.parse_args()

    model = load_model(args)
    validation_samples, data_scaler = load_input_file(args)
    input_features, targets, _ = build_combined_input(validation_samples, data_scaler = data_scaler, scale = True)


    # plots
    nn_scores = make_nn_output_plots( model, samples = validation_samples, inputs = input_features, targets = targets, args = args )
    make_discriminant_plots( model, samples = validation_samples, inputs = input_features, targets = targets, args = args )

    # roc curves
    make_nn_roc_curve( nn_scores, samples = validation_samples, inputs = input_features, targets = targets, args = args)
    make_nn_disc_roc_curve( nn_scores, samples = validation_samples, inputs = input_features, targets = targets, args = args )

    print("done, stuff saved to: {}".format(args.outdir))
    


if __name__ == "__main__" :
    main()
