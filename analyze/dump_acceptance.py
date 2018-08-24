#!/usr/bin/env python

from __future__ import print_function # in case I run this elsewhere

import sys
import os
import argparse
import math

# h5py
import h5py

# numpy
import numpy as np

def chunk_generator(input_dataset, chunksize = 10000) :

    for x in range(0, input_dataset.size, chunksize) :
        yield input_dataset[x:x+chunksize]

def valid_idx(input_array) :
    valid_lo = input_array > -np.inf
    valid_hi = input_array < np.inf
    valid = valid_lo & valid_hi
    return valid

def dump(args) :

    print("args = {}".format(args))

    input_files = args.input

    dataset_name = 'nn_scores'

    score_lowbin = 0
    score_highbin = 1
    score_edges = np.concatenate( [[-np.inf], np.linspace(score_lowbin, score_highbin, 1010), [np.inf]] )

    disc_lowbin = -30
    disc_highbin = 30
    disc_edges = np.concatenate( [[-np.inf], np.linspace(disc_lowbin, disc_highbin, 1010), [np.inf]] )

    create_arrays = True
    histo_score = None
    histo_disc = None

    for infile in input_files :

        yields_for_process = 0.0
        weights_for_process = []

        print("opening {}".format(infile))
        with h5py.File(infile, 'r', libver = 'latest') as sample_file :


            for dataset in sample_file :
                if dataset_name not in dataset :
                    print("WARNING expected dataset (={}) not found in input file".format(dataset_name))
                    continue
                input_dataset = sample_file[dataset]

                for chunk in chunk_generator(input_dataset = input_dataset) :
                    if chunk.dtype.names[0] != "eventweight" :
                        print("ERROR dataset is not of expected type (first field is not the eventweight)")
                        sys.exit()
                    field_names = chunk.dtype.names[1:]

                    weights = chunk['eventweight']
                    lumis = np.ones(weights.size) * 36.1
                    weights = lumis * weights

                    scores_idx_start = 1
                    disc_idx_start = 0
                    for field_name in field_names :
                        if 'disc' in field_name : break
                        disc_idx_start += 1

                    disc_field_names = field_names[disc_idx_start:]
                    score_field_names = field_names[:disc_idx_start]

                    scores_by_class = {}
                    disc_by_class = {}

                    for iscore, score_name in enumerate(score_field_names) :
                        label = int(score_name.split("_")[-1])
                        scores_by_class[label] = chunk[score_name]
                    for idisc, disc_name in enumerate(disc_field_names) :
                        label = int(disc_name.split("_")[-1])
                        disc_by_class[label] = chunk[disc_name]

                    # here we assume label 0 is for the signal
                    p_sig = np.array(scores_by_class[0])
                    d_sig = np.array(disc_by_class[0])

                    valid_p_idx = valid_idx(p_sig)
                    valid_d_idx = valid_idx(d_sig)
                    valid_indices = valid_p_idx & valid_d_idx

                    p_sig = p_sig[valid_indices]
                    d_sig = d_sig[valid_indices]

                    h_p, _ = np.histogram( p_sig, bins = score_edges, weights = weights )
                    h_d, _ = np.histogram( d_sig, bins = disc_edges, weights = weights )

                    if create_arrays :
                        create_arrays = False
                        histo_score = h_p
                        histo_disc = h_d
                    else :
                        histo_score += h_p
                        histo_disc += h_d

    # scores
    score_yield_by_cut = np.cumsum(histo_score[::-1])[::-1]
    score_total_yield = histo_score.sum()
    score_efficiency_by_cut = score_yield_by_cut / score_total_yield

    score_yield_by_cut = score_yield_by_cut[1:-1]
    score_efficiency_by_cut = score_efficiency_by_cut[1:-1]
    score_cutvals = score_edges[1:-2]

    # discriminants
    disc_yield_by_cut = np.cumsum(histo_disc[::-1])[::-1]
    disc_total_yield = histo_disc.sum()
    disc_efficiency_by_cut = disc_yield_by_cut / disc_total_yield
    disc_cutvals = disc_edges[1:-1]

    outfilename = "score_info"
    if args.suffix != "" :
        outfilename += "_{}".format(args.suffix)
    outfilename += ".txt"
    header = "CUT\tYIELD\tEFF\tTOTALYIELD\n"
    with open(outfilename, 'w') as ofile :
        ofile.write(header)
        for icut, cut in enumerate(score_cutvals) :
            line = "{cutval}\t{yields}\t{eff}\t{total}\n".format(cutval=cut, yields = score_yield_by_cut[icut], eff = score_efficiency_by_cut[icut], total = score_total_yield)
            ofile.write(line)

    outfilename = "disc_info"
    if args.suffix != "" :
        outfilename += "_{}".format(args.suffix)
    outfilename += ".txt"
    header = "CUT\tYIELD\tEFF\tTOTALYIELD\n"
    with open(outfilename, 'w') as ofile :
        ofile.write(header)
        for icut, cut in enumerate(disc_cutvals) :
            line = "{cutval}\t{yields}\t{eff}\t{total}\n".format(cutval = cut, yields = disc_yield_by_cut[icut], eff = disc_efficiency_by_cut[icut], total = disc_total_yield)
            ofile.write(line)

def main() :

    parser = argparse.ArgumentParser(description = "Build discriminants and calculate signal acceptances across it")
    parser.add_argument("-i", "--input", help = "Input HDF5 file with eventweights and\
        NN scores", required = True)
    parser.add_argument("-s", "--suffix", help = "Provie a suffix to append to any outputs", default = "")
    args = parser.parse_args()

    args.input = args.input.split(",")
    print("loading {} inputs".format(len(args.input)))
    dump(args)

if __name__ == "__main__" :
    main()
