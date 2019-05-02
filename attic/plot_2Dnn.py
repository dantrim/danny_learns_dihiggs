#!/usr/bin/env python3

from __future__ import print_function # just in case

import argparse

import sys
import os

# h5py
import h5py

# numpy
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

def chunk_generator(input_h5_dataset, chunksize = 10000) :

    for x in range(0, input_h5_dataset.size, chunksize) :
        yield input_h5_dataset[x:x+chunksize]

def valid_idx(input_array) :

    valid_lo = input_array > -np.inf
    valid_hi = input_array < np.inf
    return valid_lo & valid_hi

def make_twod_hists(args) :

    edges_dict = {}
    if "123456" in args.input :
        edges_dict["nn_disc_0"] = np.arange(-6, 8, 0.1)
        edges_dict["nn_disc_1"] = np.arange(-17, 2, 0.1)
        edges_dict["nn_disc_2"] = np.arange(-8, 0, 0.05)
        edges_dict["nn_disc_3"] = np.arange(-25, 2, 0.1)
        edges_dict["nn_score_0"] = np.arange(0, 1, 0.01)
        edges_dict["nn_score_1"] = np.arange(0, 0.6, 0.01)
        edges_dict["nn_score_2"] = np.arange(0, 0.6, 0.01)
        edges_dict["nn_score_3"] = np.arange(0, 1, 0.01)
    elif "410009" in args.input :
        edges_dict["nn_disc_0"] = np.arange(-17, 7, 0.1)
        edges_dict["nn_disc_1"] = np.arange(-12, 2, 0.1)
        edges_dict["nn_disc_2"] = np.arange(-5, 0, 0.05)
        edges_dict["nn_disc_3"] = np.arange(-25, 2, 0.1)
        edges_dict["nn_score_0"] = np.arange(0, 1, 0.01)
        edges_dict["nn_score_1"] = np.arange(0, 1, 0.01)
        edges_dict["nn_score_2"] = np.arange(0, 0.6, 0.01)
        edges_dict["nn_score_3"] = np.arange(0, 1, 0.01)
    elif "wt" in args.input :
        edges_dict["nn_disc_0"] = np.arange(-17, 7, 0.1)
        edges_dict["nn_disc_1"] = np.arange(-12, 2, 0.1)
        edges_dict["nn_disc_2"] = np.arange(-5, 0, 0.05)
        edges_dict["nn_disc_3"] = np.arange(-25, 2, 0.1)
        edges_dict["nn_score_0"] = np.arange(0, 1, 0.01)
        edges_dict["nn_score_1"] = np.arange(0, 0.6, 0.01)
        edges_dict["nn_score_2"] = np.arange(0, 1, 0.01)
        edges_dict["nn_score_3"] = np.arange(0, 1, 0.01)

    x_array = []
    y_array = []
    w_array = []

    dataset_name = "nn_scores"

    with h5py.File(args.input, 'r', libver = 'latest') as sample_file :

        for dataset in sample_file :

            if dataset_name not in dataset :
                print("WARNING expected dataset (={}) not found in input file".format(dataset_name))
                continue

            dset = sample_file[dataset]
            for chunk in chunk_generator(input_h5_dataset = dset) :

                if chunk.dtype.names[0] != "eventweight" :
                    print("ERROR dataset is not of expected type (first field is not the eventweight)!")
                    sys.exit()

                score_names = chunk.dtype.names[1:]
                weights = chunk['eventweight']
                scores_start_idx = 1
                disc_start_idx = 0
                for name in score_names :
                    if "disc" in name : break 
                    disc_start_idx += 1

                disc_names = score_names[disc_start_idx:]
                score_names = score_names[:disc_start_idx]

                valid_x = valid_idx(chunk[args.varX])
                valid_y = valid_idx(chunk[args.varY])
                valid = valid_x & valid_y
                chunk = chunk[valid]
                weights = weights[valid]

                varx = chunk[args.varX]
                vary = chunk[args.varY]

                x_array.extend(varx)
                y_array.extend(vary)
                w_array.extend(weights)

    x_array = np.array(x_array)#, dtype = np.float64)
    y_array = np.array(y_array)#, dtype = np.float64)
    w_array = np.array(w_array)#, dtype = np.float64)


    fig, ax = plt.subplots(1,1)
    xedges = np.arange(-10,15,1)
    yedges = np.arange(-30,5,1)
    bins = [xedges, yedges]
    bins = [edges_dict[args.varX], edges_dict[args.varY]]
    h = ax.hist2d(x_array, y_array, bins = bins,  norm = LogNorm())
    ax.set_xlabel(args.varX)
    ax.set_ylabel(args.varY)
    #ax.set_ylim([min(y_array), max(y_array)])
    fig.colorbar(h[3], ax = ax)
    #plt.gca().invert_yaxis()
    dirname = ""
    if "123456" in args.input :
        dirname = "nn_plots"
    elif "410009" in args.input :
        dirname = "nn_plots_ttbar"
    elif "wt" in args.input :
        dirname = "nn_plots_wt"
    fig.savefig("./{}/hist2d_{}_{}.pdf".format(dirname, args.varX, args.varY), bbox_inches = "tight", dpi = 200)


def main() :

    parser = argparse.ArgumentParser(description = "Plot the NN outputs and discriminants (1D and 2D etc) for an input scores file")
    parser.add_argument("-i", "--input", help = "Input scores file for a sample", required = True)
    parser.add_argument("-two", help = "Make 2D histograms", action = "store_true", default = False)
    parser.add_argument("--varX", help = "For 2D histograms, provide variable for X axis", default = "")
    parser.add_argument("--varY", help = "For 2D histograms, provide variable for Y axis", default = "")
    args = parser.parse_args()

    if not os.path.isfile(args.input) :
        print("error did not find provided input file (={})".format(args.input))
        sys.exit()

    if args.two :
        make_twod_hists(args)

if __name__ == "__main__" :
    main()
