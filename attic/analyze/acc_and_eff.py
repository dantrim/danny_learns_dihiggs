#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import argparse
import math

import h5py
import numpy as np

truth_file="/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/ml_inputs_aug22/scores_ht2ratio0p8/wwbb_truth_123456_aug23_scores.h5"
reco_file= "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/ml_inputs_aug22/scores_ht2ratio0p8/CENTRAL_123456_scores.h5"

def chunk_generator(input_dataset, chunksize = 100000) :

    for x in range(0, input_dataset.size, chunksize) :
        yield input_dataset[x:x+chunksize]

def parse_ul(args) :

    out = {}

    filename = args.ulfile
    with open(filename, 'r') as infile :
        for line in infile :
            line = line.strip().split()
            if not line : continue
            cutval = float(line[0])
            ul = float(line[1])
            out[cutval] = ul/ 36.1
    return out

def valid_idx(input_array) :
    valid_lo = input_array > -np.inf
    valid_hi = input_array < np.inf
    return valid_lo & valid_hi

def get_acc(args) :

    dataset_name = 'nn_scores'

    total = 0.0
    with h5py.File(args.input, 'r', libver = 'latest') as sample_file :
        for dataset in sample_file :
            if dataset_name not in dataset :
                print("WARNING expected dataset (={}) not found input file".format(dataset_name))
                sys.exit()
            input_dataset = sample_file[dataset]
            for chunk in chunk_generator(input_dataset = input_dataset) :
                if chunk.dtype.names[0] != "eventweight" :
                    print("ERROR dataset is not of expected format (first field is not the eventweight)")
                    sys.exit()
                weights = chunk['eventweight']
                #disc = chunk[args.var]
                #idx = disc > float(args.cut)
                #dhh = chunk["nn_disc_0"]
                hh_score = chunk['nn_score_0']
                tt_score = chunk['nn_score_1']
                wt_score = chunk['nn_score_2']
                ptt = hh_score[:] / tt_score[:]
                pwt = hh_score[:] / wt_score[:]
                #dtt = chunk["nn_disc_1"]
                #idx = (dhh > 7) & (dtt < -0.5)
                val = 40
                idx_tt = ptt > val
                idx_wt = pwt > 1.1*val
                idx = idx_tt & idx_wt
                #idx = idx & (chunk['nn_disc_3']<-15)

#                disc = disc[idx]
                weights = weights[idx]
                weights *= 36.1
                total += np.sum(weights)

    print("acceptance: {}".format(total))

def get_counts(ul_dict, filename) :

    counts = {}
    for cutval in ul_dict :
        counts[cutval] = 0.0

    total = 0.0
    with h5py.File(filename, 'r', libver = 'latest') as sample_file :
        for dataset in sample_file :
            input_dataset = sample_file[dataset]
            for chunk in chunk_generator(input_dataset = input_dataset) :
                weights = chunk['eventweight']
                hh_score = chunk['nn_score_0']
                tt_score = chunk['nn_score_1']
                wt_score = chunk['nn_score_2']

                ptt = hh_score / tt_score
                pwt = hh_score / wt_score

                hh_disc = chunk['nn_disc_0']

                for cutval in counts :
                    #idx_tt = ptt > cutval
                    #idx_wt = pwt > cutval
                    #idx = idx_tt & idx_wt
                    idx = hh_disc > cutval
                    cut_w = weights[idx]
                    cut_w *= 36.1
                    counts[cutval] += np.sum(cut_w)
    return counts

def get_ul_from_file(args, ul_dict) :

    dataset_name = 'nn_scores'

    reco_counts = get_counts(ul_dict, reco_file)
    truth_counts = get_counts(ul_dict, truth_file)

    k = 0.051
    BR = 2 * 0.5809 * 0.21

    for cutval in ul_dict :
        acc = truth_counts[cutval] / 21298.8
        eff = reco_counts[cutval] / truth_counts[cutval]
        den = acc * eff * k * BR
        xsec_ul = ul_dict[cutval] / den
        print(55 * '-')
        print("CUT {} -->  xsec UL = {}".format(cutval,xsec_ul))
        print("CUT {} --> (RECO {}, TRUTH {}, ULN {}) xsec UL = {}".format(cutval, reco_counts[cutval], truth_counts[cutval], ul_dict[cutval],xsec_ul))


def main() :
    parser = argparse.ArgumentParser(description = "Dump the acc. for a given cut on a given discriminant")
    #parser.add_argument("-i", "--input", help = "Input HDF5 scores file", required = True)
    #parser.add_argument("-v", "--var", help = "Var to cut on", default = "nn_disc_0")
    #parser.add_argument("-c", "--cut", help = "Cut threshold on the input variable", required = True)
    parser.add_argument("--ulfile", help = "Text file with cut val and UL", required = True)
    args = parser.parse_args()

    #if not os.path.isfile(args.input) :
    #    print("provided input file {} not found".format(args.input))
    #    sys.exit()

    ul_dict = parse_ul(args)
#    get_acc(args)
    get_ul_from_file(args, ul_dict)

if __name__ == "__main__" :
    main()
