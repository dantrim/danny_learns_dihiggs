#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import argparse
import math

# h5py
import h5py

# numpy
import numpy as np

# plotting
import matplotlib.pyplot as plt

# stats
import significance


score_filedir = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/score_files2/"
score_filedir = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/ml_inputs_aug8_2/score_files/"
ttbar_file = "{}/CENTRAL_410009_scores.h5".format(score_filedir)
zll_file = "{}/sherpa_zll_scores.h5".format(score_filedir)
ztt_file = "{}/sherpa_ztt_scores.h5".format(score_filedir)
wt_file = "{}/wt_bkg_scores.h5".format(score_filedir)
background_files = [ttbar_file, zll_file, ztt_file, wt_file]
#truth_sig_file = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/score_files2/wwbb_truth_123456_aug6_custom_scores.h5"
#truth_sig_file = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/score_files2/wwbb_truth_342053_aug6_scores.h5"
truth_sig_file = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/ml_inputs_aug8_2/score_files/wwbb_truth_342053_aug6_scores.h5"
truth_sig_file = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/ml_inputs_aug8_2/score_files/wwbb_truth_123456_aug6_custom_scores.h5"

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
        [[-np.inf], np.linspace(score_lowbin, score_highbin, 1010), [np.inf]])
    #edges = np.arange(0,1,0.1)

    disc_lowbin = -30
    disc_highbin = 20
    disc_edges = np.concatenate([[-np.inf], np.linspace(disc_lowbin,disc_highbin,1010), [np.inf]])


    score_holders = []
    disc_holders = []


    for infile in input_files :

        yields_for_process = 0.0
        weights_for_process = []

        with h5py.File(infile, 'r', libver = 'latest') as sample_file :

            print("opening {}".format(infile))

            for dataset in sample_file :

                if dataset_name not in dataset :
                    print("WARNING expected dataset (={}) not found in input file".format(dataset_name))
                    continue

                input_dataset = sample_file[dataset]

#                chunks = chunk_generator(infile, dataset_name = dataset_name, chunksize = 1000)
                chunks = chunk_generator_dataset(input_dataset = input_dataset)

                for chunk in chunks :

                    #n_total += chunk.size

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
                    lumis = np.ones(len(weights)) * 36.1
                    weights = lumis * weights

                    scores_start_idx = 1
                    disc_start_idx = 0
                    for name in score_names :
                        if 'disc' in name : break
                        disc_start_idx += 1

                    disc_names = score_names[disc_start_idx:]
                    score_names = score_names[:disc_start_idx]
    
                    scores_by_class = {}
                    disc_by_class = {}

                    for iscore, score_name in enumerate(score_names) :
                        label = int(score_name.split("_")[-1])
                        scores_by_class[label] = chunk[score_name]
                    for idisc, disc_name in enumerate(disc_names) :
                        label = int(disc_name.split("_")[-1])
                        disc_by_class[label] = chunk[disc_name]

                    p_sig = np.array(scores_by_class[0]) # assume class 0 is signal
                    d_sig = disc_by_class[0]

                    valid_p = valid_idx(p_sig)
                    valid_d = valid_idx(d_sig)
                    idx = valid_p & valid_d

                    n_before = idx.sum()
                    # selection on a discriminant
                    #valid_z = valid_idx( disc_by_class[3] )
                    #idx = valid_p & valid_d & valid_z
                    #cut_z_idx = disc_by_class[3] < -20
                    #idx = idx & cut_z_idx
                    #n_after = idx.sum()

                    #valid_w = valid_idx( disc_by_class[2] )
                    #valid_t = valid_idx( disc_by_class[1] )
                    #valid_z = valid_idx( disc_by_class[3] )

                    #idx = valid_p & valid_d
                    ##idx = idx & valid_w
                    ##idx = idx & valid_t
                    #idx = idx & valid_z

                    #d_t = disc_by_class[1][idx]
                    #d_w = disc_by_class[2][idx]
                    #d_z = disc_by_class[3][idx]

                    #cut_z_idx = d_z < -10
                    #idx = idx & cut_z_idx

#                    cut_t_idx = p_t < -1
#                    cut_w_idx = d_t < -1
#                    idx = idx & cut_w_idx
                   # idx = idx & cut_t_idx & cut_w_idx

                    #valid_t = valid_idx( disc_by_class[1] )
                    #idx = idx & valid_t
                    #cut_t_idx = disc_by_class[1] < -10
                    #idx = idx & cut_t_idx
                    #print("N before Z cut = {}, N after Z cut = {}".format( n_before, n_after ) )

                    p_sig = p_sig[idx]
                    d_sig = d_sig[idx]
                    weights = weights[idx]

                    yields_for_process += weights.sum()
                    weights_for_process.append(weights**2)

                    h_p, _ = np.histogram( p_sig, bins = score_edges, weights = weights )
                    h_d, _ = np.histogram( d_sig, bins = disc_edges, weights = weights )

                    if load_histo_score :
                        load_histo_score = False
                        histo_score = h_p
                    else :
                        histo_score += h_p

                    if load_histo_disc :
                        load_histo_disc = False
                        histo_disc = h_d
                    else :
                        histo_disc += h_d

#            weightts_for_process = math.sqrt( np.array(weights_for_process).sum())
#            print("yield for {0} : {1:.3f} +/- {2:.3f}".format(infile.split("/")[-1], yield_for_process, weights_for_process))
#            print("yield for {0} : {1:.3f}".format(infile.split("/")[-1], yields_for_process))
    # scores
    yield_by_cut = np.cumsum(histo_score[::-1])[::-1]
    total_yield = histo_score.sum()
    print("total yield {}".format(total_yield))
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

def get_upperlimit(nbkg = 0) :

    if np.isnan(nbkg) :
        print("invalid bkg value!")
        sys.exit()

    sig = 0.5
    z = 0

    while True :

        z = significance.binomial_exp_z(sig, nbkg, 0.33)

        if z > 1.64 :
            break
        if sig > 100 :
            sig = -1
            break
        #print("nbkg = {}, sig is at {}, Z -> {}".format(nbkg, sig, z))
        sig += 0.01

    return sig
    

def calculate_upperlimits(counts_holder) :

    cutvals = counts_holder.thresholds()
    yields = counts_holder.yields()
    print("UL yields = {}".format(yields[:5]))

    yields = np.array(yields)
    #idx = valid_idx(yields) #yields != 0
    #yields = list(yields[idx])

    #yields = list(yields[idx])
    #cutvals = list( np.array(cutvals)[idx] )

    s95_vals = {}

    for idx, cutval in enumerate(cutvals) :
        print("BLAHBLAH cutval {} = {}".format(cutval, yields[idx]))

    for icut, cutval in enumerate(cutvals) :

        #if cutval < 5 : continue
        if cutval < 5 : continue
        if cutval > 10 : continue
        #if cutval < 0.85 : continue
        print("########## cutval = {}".format(cutval))

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
    parser.add_argument
    args = parser.parse_args()

    # signal stuff
    class_dict = get_class_dict(args)
    sig_class_counts, sig_disc_counts = load_file([args.input], class_dict, sample_type = 'sig')
    truth_sig_class_counts, truth_sig_disc_counts = load_file([truth_sig_file], class_dict, sample_type = "sig_truth")
    bkg_class_counts, bkg_disc_counts = load_file(background_files, class_dict, sample_type = 'bkg')

#    s95_dict = calculate_upperlimits(bkg_class_counts)
    s95_dict = calculate_upperlimits(bkg_disc_counts)
    print("s95_dict {}".format(s95_dict.keys()))

    lowest_xsec_ul = 99999999
    best_threshold = 0
    lumi_factor = 36.1
    for key in s95_dict :
        # use disc
        cut_idx = sig_disc_counts.index_of_threshold(key)
        truth_counts = truth_sig_disc_counts.yields()[cut_idx]

        acceptance = truth_counts / (590 * lumi_factor) #  * 36.1)
        reco_eff = sig_disc_counts.yields()[cut_idx]
        reco_eff = reco_eff / truth_counts
        e_times_a = reco_eff * acceptance
        #e_times_a = sig_disc_counts.yields()[cut_idx] / (590)
        br = 2 * 0.57 * 0.21 
        print(50 * "-")
        n_sig_ul = s95_dict[key]
        print("CUT VAL = {}".format(key))
        print("n bkg = {}".format(bkg_disc_counts.yields()[cut_idx]))
        print("S95 = {}, acceptance = {}, efficiency {} : e x A = {}".format(n_sig_ul, acceptance, reco_eff, acceptance * reco_eff))
        xsec_ul = n_sig_ul / lumi_factor
        xsec_ul = xsec_ul / ( br * e_times_a )
        if xsec_ul < lowest_xsec_ul and xsec_ul > 0 :
            lowest_xsec_ul = xsec_ul
            best_threshold = key
        print(" ->> S95 = {} ==> xsec UL = {}".format(n_sig_ul, xsec_ul))



        # usc scores
 #       cut_idx = sig_class_counts.index_of_threshold(key)
 #       truth_counts = truth_sig_class_counts.yields()[cut_idx]
 #       acceptance = truth_counts / (590 * lumi_factor)
 #       reco_eff = sig_class_counts.yields()[cut_idx]
 #       reco_eff = reco_eff / truth_counts
 #       e_times_a = reco_eff * acceptance
 #       br = 2 * 0.57 * 0.21
 #       print(50 * "-")
 #       print("CUT VAL = {}".format(key))
 #       print("n bkg = {}".format(bkg_class_counts.yields()[cut_idx]))
 #       print("acceptance = {}, efficiency {} : e x A = {}".format(acceptance, reco_eff, acceptance * reco_eff))
 #       n_sig_ul = s95_dict[key]
 #       xsec_ul = n_sig_ul / lumi_factor
 #       xsec_ul = xsec_ul / ( br * e_times_a )
 #       if xsec_ul < lowest_xsec_ul :
 #           if xsec_ul > 0 :
 #               lowest_xsec_ul = xsec_ul
 #               best_threshold = key
 #       print(" ->> S95 = {} ==> xsec UL = {}".format(n_sig_ul, xsec_ul))
        
        


    #    cut_idx = sig_class_counts.index_of_threshold(key)
    #    truth_counts = truth_sig_class_counts.yields()[cut_idx]
    #    reco_eff = sig_class_counts.yields()[cut_idx] 
    #    reco_eff = reco_eff / truth_counts
#   #     sig_eff = sig_class_counts.efficiencies()[cut_idx]
#   #     sig_eff = sig_class_counts.yields()[cut_idx]
    #    sig_acc = truth_sig_class_counts.efficiencies()[cut_idx]
    #    br = 0.24        
    #    print(" CUT {} N = {} -> e x A x BR = {} x {} x BR = {} (BKG = {}, SIG = {})".format(key, s95_dict[key], reco_eff, sig_acc, reco_eff * sig_acc * br, bkg_class_counts.yields()[cut_idx], sig_class_counts.yields()[cut_idx]))

        #print("key {} -> idx {}, idx {} == S95 = {}".format(key, sig_disc_counts.index_of_threshold(key), truth_sig_disc_counts.index_of_threshold(key), s95_dict[key]))

    print(59 * "*")
    print("LIMIT ON XSEC = {} at threshold {}".format(lowest_xsec_ul, best_threshold))
    

if __name__ == "__main__" :
    main()
