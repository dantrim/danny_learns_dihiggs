#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import argparse

# h5py
import h5py

# numpy
import numpy as np

# plotting
import matplotlib.pyplot as plt

def get_class_dict(args) :

    if args.score_labels == "" :
        return dict()

    out = {}
    label_string = args.score_labels
    labels = label_string.split(",")
    for label in labels :
        class_string = label.split(":")
        class_label = int(class_string[0])
        class_name = str(class_string[1])
        out[class_label] = class_name
    return out

def get_discriminant( signal_prob = None, bkg_probs = [] ) :

    denominator = bkg_probs[0]
    for bkg_prob in bkg_probs[1:] :
        denominator += bkg_prob
    return np.log( signal_prob / denominator )

def count_things(args) :

    input_file = args.input
    dataset_name = "nn_scores"

    class_dict = get_class_dict(args)
    print("class_dict loaded : {}".format(class_dict))

    class_probs = {}
    sample_weights = None

    with h5py.File(input_file, 'r', libver = 'latest') as infile :
        if dataset_name not in infile :
            print("ERROR expected dataset (={}) not found in input file".format(dataset_name))
            sys.exit()
        dataset = infile[dataset_name]

        if dataset.dtype.names[0] != "eventweight" :
            print("ERROR expected dataset is not of expected type (first fields is not the eventweight)")
            sys.exit()
        score_names = dataset.dtype.names[1:]
        if class_dict and len(score_names) != len(class_dict.keys()) :
            print("ERROR expected number of NN scores based on user input labels provided \
                (={}) does not match the number of score fields in the input file (={})"\
                .format(len(class_dict.keys()), len(score_names)))
            sys.exit()

        sample_weights = dataset['eventweight']

        for iscore, score_name in enumerate(score_names) :
            label = int(score_name.split("_")[-1])
            scores = dataset[score_name]
            class_probs[label] = scores

    # assume for now that signal is label == 0
    lowbin = 0
    highbin = 1
    edges = np.concatenate(
        [[-np.inf], np.linspace(lowbin, highbin, 505), [np.inf]])
    hscore, _ = np.histogram( class_probs[0], bins = edges, weights = sample_weights.reshape((class_probs[0].shape[0],)))

    # yield as a function of cutvalue
    yield_by_cut = np.cumsum( hscore[::-1] )[::-1]
    # total yield
    total_yield = hscore.sum()
    eff_by_cut = yield_by_cut / total_yield
    #print("total = {}".format(total_yield))
    #print("cut vals = {}".format(edges))
    #print("yields   = {}".format(yield_by_cut))
    #print("eff by cut = {}".format(list(eff_by_cut)))
    #print("edges = {}, hist = {}".format(len(edges), len(yield_by_cut)))
    #fig, ax = plt.subplots(1,1)
    #ax.set_yscale('log')
    #ax.set_xlim([0,1])
    #binning = np.arange(0,1,0.02)
    #centers = (binning[1:-2]+binning[2:-1])/2
    #yields, _ = np.histogram( class_probs[0], bins = binning, weights = sample_weights.reshape((class_probs[0].shape[0],)))
    #yields = yields / yields.sum()
    #ax.step(centers, yields[1:-1], label = "yep", where = 'mid')
    #fig.savefig("test.pdf", bbox_inches = 'tight', dpi = 200)

    # trim off the right most edge at the bin edge and remove under/overflow
    cutvals = edges[1:-2]
    effs = eff_by_cut[1:-1]
    for icut, cut in enumerate(cutvals) :
        print("score {0:.5f} {1:.5f}".format(cutvals[icut], effs[icut]))

    if args.disc :

        p_sig = None
        p_bkg = []
        for label in class_probs :
            if label == 0 :
                p_sig = class_probs[label]
            else :
                p_bkg.append(class_probs[label])
        d_sig = get_discriminant( signal_prob = p_sig, bkg_probs = p_bkg )

        xmin = np.min(d_sig)
        xmax = np.max(d_sig)
        xmin = int(0.95 * xmin)
        xmax = int(1.05*xmax)

        edges = np.concatenate(
            [[-np.inf], np.linspace(xmin, xmax, 505), [np.inf]])
        hdisc, _ = np.histogram( d_sig, bins = edges, weights = sample_weights.reshape((class_probs[0].shape[0],)))
        yields_by_cut = np.cumsum( hdisc[::-1] )[::-1]
        total_yield = hdisc.sum()
        effs_by_cut = yields_by_cut / total_yield

        cutvals = edges[1:-2]
        effs = effs_by_cut[1:-1]
        for icut, cut in enumerate(cutvals) :
            print("disc {0:.5f} {1:.5f}".format(cutvals[icut], effs[icut]))

        xmin = -40
        xmax = 20
        fig, ax = plt.subplots(1,1)
        ax.set_yscale('log')

        edges = np.concatenate(
            [[-np.inf], np.linspace(xmin, xmax, 505), [np.inf]])
        centers = (edges[1:-2] + edges[2:-1])/2
        yields, _ = np.histogram( d_sig, bins = edges, weights = sample_weights.reshape((class_probs[0].shape[0],)))
        yields = yields / yields.sum()
        ax.step(centers, yields[1:-1], label = 'yep', where = 'mid')
        fig.savefig("test.pdf", bbox_inches = 'tight', dpi = 200)

    
    #lowbin = 0
    #highbin = 1
    #edges = np.concatenate(
    #    [[-np.inf], np.linspace(lowbin, highbin, 505), [np.inf]])
    #hscore, _ = np.histogram( class_probs[0], bins = edges, weights = sample_weights.reshape((class_probs[0].shape[0],)))

    ## yield as a function of cutvalue
    #yield_by_cut = np.cumsum( hscore[::-1] )[::-1]
    ## total yield
    #total_yield = hscore.sum()

    

        
        


def main() :

    parser = argparse.ArgumentParser(description = "Build discriminants and count things")
    parser.add_argument("-i", "--input", help = "Input HDF5 file with eventweights and\
        NN scores", required = True)
    parser.add_argument("-n", "--name", help = "Provide a name", required = True)
    parser.add_argument("--score-labels", help = "Provide correspondence between\
        NN output label and class label (will assume label 0 is signal by default)", default = "")
    parser.add_argument("-d", "--disc", help = "Get numbers for log ratio discriminant", default = False,
        action = "store_true")
    args = parser.parse_args()

    count_things(args)

if __name__ == "__main__" :
    main()
