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

# stats
import significance


score_filedir = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/score_files2/"
ttbar_file = "{}/CENTRAL_410009_scores.h5".format(score_filedir)
zll_file = "{}/sherpa_zll_scores.h5".format(score_filedir)
ztt_file = "{}/sherpa_ztt_scores.h5".format(score_filedir)
wt_file = "{}/wt_bkg_scores.h5".format(score_filedir)
background_files = [ttbar_file, zll_file, ztt_file, wt_file]
truth_sig_file = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/score_files2/wwbb_truth_123456_aug6_custom_scores.h5"

def chunk_generator(input_file, chunksize = 100000, dataset_name = "") :

    """
    Construct a generator for iterating over chunks
    of an input HDF5 file.

    Args :
        input_file : input HDF5 file to iterate through
        chunksize : size of chunks to readout during each iteration
        dataset_name : name of input dataset to process
    """

    with h5py.File(input_file, 'r', libver = 'latest') as f :
        dataset = f[dataset_name]
        for x in range(0, dataset.size, chunksize) :
            yield dataset[x:x+chunksize]

def chunk_generator_dataset(input_dataset, chunksize = 10000) :

    for x in range(0, input_dataset.size, chunksize) :
        yield input_dataset[x:x+chunksize]

class AcceptanceHolder :
    def __init__(self, name = "", thresholds = [], yields = [], efficiencies = [], total_yield = 0, is_disc = False) :
        self._name = name
        self._total_yield = total_yield
        self._thresholds = thresholds
        self._yields_at_cut = yields
        self._efficiencies = efficiencies
        self._is_disc = is_disc

    def name(self) :
        return self._name
    def thresholds(self) :
        return self._thresholds
    def yields(self) :
        return self._yields_at_cut
    def total_yield(self) :
        return self._total_yield
    def efficiencies(self) :
        return self._efficiencies
    def is_nn_score(self) :
        return not self._is_disc
    def is_discriminant(self) :
        return self._is_disc
    def index_of_threshold(self, thr) :
        return list(self._thresholds).index(thr)

def valid_idx(input_array) :
    valid_lo = (input_array > -np.inf)
    valid_hi = (input_array < np.inf)
    valid = (valid_lo & valid_hi)
    return valid

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

def load_file(input_files = [], class_dict = {}, sample_type = "") :

    dataset_name = "nn_scores"

    histo_score = None
    histo_disc = None

    load_histo_score = True
    load_histo_disc = True


    score_lowbin = 0
    score_highbin = 1
    score_edges = np.concatenate(
        [[-np.inf], np.linspace(score_lowbin, score_highbin, 101), [np.inf]])
    #edges = np.arange(0,1,0.1)

    disc_lowbin = -30
    disc_highbin = 20
    disc_edges = np.concatenate([[-np.inf], np.linspace(disc_lowbin,disc_highbin,202), [np.inf]])


    score_holders = []
    disc_holders = []


    for infile in input_files :

        with h5py.File(infile, 'r', libver = 'latest') as sample_file :

            print("opening {}".format(infile))

            for dataset in sample_file :

                if dataset_name not in dataset :
                    print("WARNING expected dataset (={}) not found in input file".format(dataset_name))
                    continue

                input_dataset = sample_file[dataset]

#                chunks = chunk_generator(infile, dataset_name = dataset_name, chunksize = 1000)
                chunks = chunk_generator_dataset(input_dataset = input_dataset)

                n_total = 0
                for chunk in chunks :

                    n_total += chunk.size

                    if chunk.dtype.names[0] != "eventweight" :
                        print("ERROR dataset is not of expected type (first field is not the eventweight)")
                        sys.exit()

                    score_names = chunk.dtype.names[1:]
                    if class_dict and len(score_names) != len(class_dict.keys()) :
                        print("ERROR expected number of NN scores based on user input labels provided \
                            (={}) does not match the number of score fields in the input file (={})"\
                            .format(len(class_dict.keys()), len(score_names)))
                        sys.exit()

                    weights = chunk['eventweight']
                    class_probs = {}
                    for iscore, score_name in enumerate(score_names) :
                        label = int(score_name.split("_")[-1])
                        scores = chunk[score_name]
                        class_probs[label] = scores

                    sig_scores = class_probs[0]
                    bkg_scores = []
                    for label in class_probs :
                        if label == 0 : continue
                        bkg_scores.append(class_probs[label])
                    d_sig = get_discriminant(sig_scores, bkg_scores)

                    idx = valid_idx(sig_scores)
                    valid_scores = sig_scores[idx]
                    valid_weights = weights[idx]

                    idx = valid_scores != 0
                    valid_scores = valid_scores[idx]
                    valid_weights = valid_weights[idx]

                    hscore, _ = np.histogram(valid_scores, bins = score_edges, weights = valid_weights)#.reshape((valid_scores.shape[0],)))

                    if load_histo_score :
                        load_histo_score = False
                        histo_score = hscore
                    else :
                        histo_score += hscore

                    idx = valid_idx(d_sig)
                    valid_d_sig = d_sig[idx]
                    valid_weights = weights[idx]
                    hdisc, _ = np.histogram(valid_d_sig, bins = disc_edges, weights = valid_weights.reshape((valid_d_sig.shape[0],)))

                    if load_histo_disc :
                        load_histo_disc = False
                        histo_disc = hdisc
                    else :
                        histo_disc += hdisc


    # scores
    yield_by_cut = np.cumsum(histo_score[::-1])[::-1]
    total_yield = histo_score.sum()
    eff_by_cut = yield_by_cut / total_yield
    
    yield_by_cut = yield_by_cut[1:-1]
    eff_by_cut = eff_by_cut[1:-1]
    
    cutvals = score_edges
    centers = (cutvals[1:-2] + cutvals[2:-1])/2
    cutvals = score_edges[1:-2]
    yields = histo_score[1:-1]
    total = yields.sum()
    effs = yields / total
    
    #for icut, cut in enumerate(cutvals) :
    #    print("score {0:.5f} {1:.5f}".format(cutvals[icut], eff_by_cut[icut]))
    
    score_holder = AcceptanceHolder(name = sample_type, thresholds = cutvals, total_yield = total_yield, yields = yield_by_cut, efficiencies = eff_by_cut, is_disc = False)
    score_holders.append(score_holder)
    
    fig, ax = plt.subplots(1,1)
    ax.set_yscale('log')
    ax.step(centers, eff_by_cut, label = 'yep', where = 'mid')
    fig.savefig('test_scor_{}.pdf'.format(sample_type), bbox_inches = 'tight', dpi = 200)
    
    # disc
    yield_by_cut = np.cumsum(histo_disc[::-1])[::-1]
    total_yield = histo_disc.sum()
    eff_by_cut = yield_by_cut / total_yield
    
    yield_by_cut = yield_by_cut[1:-1]
    eff_by_cut = eff_by_cut[1:-1]
    
    
    cutvals = disc_edges
    centers = (cutvals[1:-2] + cutvals[2:-1])/2
    cutvals = disc_edges[1:-2]
    #yields = histo_disc[1:-1]
    #total = yields.sum()
    #effs = yields / total
    #for icut, cut in enumerate(cutvals) :
    #    print("disc {0:.5f} {1:.5f}".format(cutvals[icut], eff_by_cut[icut]))
    
    disc_holder = AcceptanceHolder(name = sample_type, thresholds = cutvals, total_yield = total_yield, yields = yield_by_cut, efficiencies = eff_by_cut, is_disc = True)
    disc_holders.append(disc_holder)
    
    fig, ax = plt.subplots(1,1)
    ax.set_yscale('log')
    ax.step(centers, eff_by_cut, label = 'yep', where = 'mid')
    fig.savefig('test_disc_{}.pdf'.format(sample_type), bbox_inches = 'tight', dpi = 200)
      
    return score_holders[0], disc_holders[0]

def count_things(class_probs = None, sample_weights = None, run_disc = False) :

    # assume for now that signal is label == 0
    print("FUCK class_probs = {}".format(class_probs[0]))
    lowbin = 0
    highbin = 1
    edges = np.concatenate(
        [[-np.inf], np.linspace(lowbin, highbin, 505), [np.inf]])
    #edges = np.linspace(lowbin, highbin, 505)
    #edges = np.arange(0,1,0.01)
    #print("edges = {}".format(edges))
    #edges = [0,1]
    #edges = np.arange(0,1,100)
    idx = ~np.isnan(class_probs[0])



#    idx = valid_idx(class_probs[0])
    probs = class_probs[0][idx]
    weights = sample_weights[idx]
    #weights = sample_weights
    #idx = ~np.isnan(sample_weights)
    #probs = probs[idx]
    #weights = sample_weights[idx]
    hscore, _ = np.histogram( list(probs),  bins = edges, weights = weights.reshape((probs.shape[0],)))

    # yield as a function of cutvalue
    yield_by_cut = np.cumsum( hscore[::-1] )[::-1]
    # total yield
    total_yield = hscore.sum()
    eff_by_cut = yield_by_cut / total_yield
    print("total = {}".format(total_yield))
    print("cut vals = {}".format(edges))
    print("yields   = {}".format(yield_by_cut))
    print("eff by cut = {}".format(list(eff_by_cut)))
    print("edges = {}, hist = {}".format(len(edges), len(yield_by_cut)))
    fig, ax = plt.subplots(1,1)
    ax.set_yscale('log')
    ax.set_xlim([0,1])
    binning = np.arange(0,1,0.02)
    centers = (binning[1:-2]+binning[2:-1])/2
    idx = ~np.isnan(sample_weights)
    sample_weights = sample_weights[idx]
    #yields, _ = np.histogram( sample_weights, 1000) #, weights = sample_weights[idx].reshape((class_probs[0][idx].shape[0],)))
    yields, _ = np.histogram( class_probs[0][idx], bins = binning, weights = sample_weights[idx].reshape((class_probs[0][idx].shape[0],)))
    print("yields = {}".format(yields))
    yields = yields / yields.sum()
    ax.step(centers, yields[1:-1], label = "yep", where = 'mid')
    fig.savefig("test.pdf", bbox_inches = 'tight', dpi = 200)
    print("yup")

    # trim off the right most edge at the bin edge and remove under/overflow
    cutvals = edges[1:-2]
    effs = eff_by_cut[1:-1]
    for icut, cut in enumerate(cutvals) :
        print("score {0:.5f} {1:.5f}".format(cutvals[icut], yield_by_cut[icut]))
        #print("score {0:.5f} {1:.5f}".format(cutvals[icut], effs[icut]))

    score_holder = AcceptanceHolder(name = "signal", thresholds = cutvals, yields = yield_by_cut, efficiencies = effs)
    disc_holder = None

    if run_disc :

        p_sig = None
        p_bkg = []
        for label in class_probs :
            if label == 0 :
                p_sig = class_probs[label]
            else :
                p_bkg.append(class_probs[label])
        d_sig = get_discriminant( signal_prob = p_sig, bkg_probs = p_bkg )

        #xmin = np.min(d_sig)
        #xmax = np.max(d_sig)
        #xmin = int(0.95 * xmin)
        #xmax = int(1.05*xmax)
        xmin = -40
        xmax = 40

        edges = np.concatenate(
            [[-np.inf], np.linspace(xmin, xmax, 505), [np.inf]])
        idx = valid_idx(d_sig)
        d_sig = d_sig[idx]
        weights = sample_weights[idx]
        hdisc, _ = np.histogram( d_sig, bins = edges, weights = weights.reshape((d_sig.shape[0],)))
        yields_by_cut = np.cumsum( hdisc[::-1] )[::-1]
        total_yield = hdisc.sum()
        effs_by_cut = yields_by_cut / total_yield

        cutvals = edges[1:-2]
        effs = effs_by_cut[1:-1]
        for icut, cut in enumerate(cutvals) :
            print("disc {0:.5f} {1:.5f}".format(cutvals[icut], effs[icut]))

        disc_holder = AcceptanceHolder(name = "signal", thresholds = cutvals, yields = yields_by_cut, efficiencies = effs, is_disc = True)

        #xmin = -40
        #xmax = 20
        #fig, ax = plt.subplots(1,1)
        #ax.set_yscale('log')

        #edges = np.concatenate(
        #    [[-np.inf], np.linspace(xmin, xmax, 505), [np.inf]])
        #centers = (edges[1:-2] + edges[2:-1])/2
        #yields, _ = np.histogram( d_sig, bins = edges, weights = sample_weights.reshape((class_probs[0].shape[0],)))
        #yields = yields / yields.sum()
        #ax.step(centers, yields[1:-1], label = 'yep', where = 'mid')
        #fig.savefig("test.pdf", bbox_inches = 'tight', dpi = 200)

    
    #lowbin = 0
    #highbin = 1
    #edges = np.concatenate(
    #    [[-np.inf], np.linspace(lowbin, highbin, 505), [np.inf]])
    #hscore, _ = np.histogram( class_probs[0], bins = edges, weights = sample_weights.reshape((class_probs[0].shape[0],)))

    ## yield as a function of cutvalue
    #yield_by_cut = np.cumsum( hscore[::-1] )[::-1]
    ## total yield
    #total_yield = hscore.sum()

    return score_holder, disc_holder

def count_group(score_list = [], weights_list = [], do_disc = False) :

    lowbin = 0
    highbin = 1
    edges = np.concatenate(
        [[-np.inf], np.linspace(lowbin, highbin, 505), [np.inf]])

    hscore_total = None
    load_hist = True

    for ibkg, bkg_scores in enumerate(score_list) :

        idx = valid_idx(bkg_scores[0])
        scores = bkg_scores[0][idx]
        weights = weights_list[ibkg][idx]
        print("scores {}".format(scores))
        hscore, _ = np.histogram( scores, bins = edges, weights = weights.reshape((scores.shape[0],)))
        print("hscore {}".format(hscore))
        if load_hist :
            load_hist = False
            hscore_total = hscore
            #print("hscore_total {}".format(hscore_total))
        else :
            hscore += hscore
            #print("hscore_total {}".format(hscore_total))
    sys.exit()
   
    yield_by_cut = np.cumsum( hscore_total[::-1] )[::-1]
    total_yield = hscore_total.sum()
    eff_by_cut = yield_by_cut / total_yield
    
    cutvals = edges[1:-2]
    effs = eff_by_cut[1:-1]

    score_holder = AcceptanceHolder(name = "bkg", thresholds = cutvals, yields = yield_by_cut, efficiencies = effs)

    disc_holder = None
    if do_disc :

        hdisc_total = None
        load_hist = True

        for ibkg, bkg_scores in enumerate(score_list) :
            p_sig = None
            p_bkg = []
            for label in bkg_scores :
                if label == 0 :
                    p_sig = bkg_scores[label]
                else :
                    p_bkg.append(bkg_scores[label])
            d_sig = get_discriminant( signal_prob = p_sig, bkg_probs = p_bkg )

            idx = valid_idx(d_sig)
            d_sig = d_sig[idx]
            weights = weights_list[ibkg][idx]

            xmin = -40
            xmax = 40
            edges = np.concatenate(
                [[-np.inf], np.linspace(xmin, xmax, 505), [np.inf]])
            hdisc, _ = np.histogram( d_sig, bins = edges, weights = weights.reshape((d_sig.shape[0],)))
            #print("hdisc {}".format(hdisc))

            if load_hist :
                load_hist = False
                hdisc_total = hdisc
            else :
                hdisc_total += hdisc

        yield_by_cut = np.cumsum( hdisc_total[::-1] )[::-1]
        total_yield = hdisc_total.sum()
        eff_by_cut = yield_by_cut / total_yield

        cutvals = edges[1:-2]
        effs = eff_by_cut[1:-1]
        yield_by_cut = yield_by_cut[1:-1]
        print("blah {}".format(yield_by_cut))
        sys.exit()
        disc_holder = AcceptanceHolder(name = "bkg", thresholds = cutvals, yields = yield_by_cut, efficiencies = effs, is_disc = True)

    return score_holder, disc_holder

def get_upperlimit(nbkg = 0) :

    if np.isnan(nbkg) :
        print("invalid bkg value!")
        sys.exit()

    sig = 0.5
    z = 0

    while True :

        z = significance.binomial_exp_z(sig, nbkg, 0.3)

        if z > 1.64 :
            break
        if sig > 1000 :
            sig = -1
            break
        #print("nbkg = {}, sig is at {}, Z -> {}".format(nbkg, sig, z))
        sig += 0.01

    return sig
    

def calculate_upperlimits(counts_holder) :

    cutvals = counts_holder.thresholds()
    yields = counts_holder.yields()

    yields = np.array(yields)
    #idx = valid_idx(yields) #yields != 0
    #yields = list(yields[idx])

    #yields = list(yields[idx])
    #cutvals = list( np.array(cutvals)[idx] )

    s95_vals = {}

    for icut, cutval in enumerate(cutvals) :

        if cutval < 1 : continue

        nbkg = yields[icut]

        if nbkg < 0 : continue
        if nbkg < 1 : continue

        delta_b = 0.3
        sig_95 = get_upperlimit(nbkg)

        s95_vals[cutval] = sig_95

    return s95_vals
        

        

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

    # signal stuff
    class_dict = get_class_dict(args)
    sig_class_counts, sig_disc_counts = load_file([args.input], class_dict, sample_type = 'sig')
    truth_sig_class_counts, truth_sig_disc_counts = load_file([truth_sig_file], class_dict, sample_type = "sig_truth")
    bkg_class_counts, bkg_disc_counts = load_file(background_files, class_dict, sample_type = 'bkg')
    s95_dict = calculate_upperlimits(bkg_disc_counts)

    print("s95_dict {}".format(s95_dict.keys()))
    for key in s95_dict :
        cut_idx = sig_disc_counts.index_of_threshold(key)
        sig_eff = sig_disc_counts.efficiencies()[cut_idx]
        sig_acc = truth_sig_disc_counts.efficiencies()[cut_idx]
        br = 0.071        
        print(" CUT {} -> e x A x BR = {}".format(key, sig_eff * sig_acc * br))

        #print("key {} -> idx {}, idx {} == S95 = {}".format(key, sig_disc_counts.index_of_threshold(key), truth_sig_disc_counts.index_of_threshold(key), s95_dict[key]))
    

if __name__ == "__main__" :
    main()
