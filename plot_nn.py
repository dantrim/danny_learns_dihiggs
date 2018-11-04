#!/usr/bin/env python3

import os
import sys

import h5py
import numpy as np
import argparse

import matplotlib.pyplot as plt

from overlay_nn import Sample, chunk_generator, valid_idx
from train import DataScaler, floatify
from roc_nn_and_cut import load_stored_model

def make_plots(args) :

    sample = Sample("sample", args.input, "")
    data_scaler, model = load_stored_model(args.nn_dir)

    lwtnn_data = []
    otf_data = []
    weight_data = []
    weight2_data = []

    with h5py.File(sample.filename, 'r', libver = 'latest') as sample_file : 
        if 'superNt' not in sample_file :
            print('ERROR "superNt" dataset not found in input file (={})'.format(sample.filename))
            sys.exit(1)
        dataset = sample_file['superNt']
        for chunk in chunk_generator(dataset) :
            weights = chunk['eventweight']

            # LWTNN var
            lwtnn_var = chunk['NN_p_hh']

            # OTF
            input_features = chunk[data_scaler.feature_list()]
            input_features = floatify(input_features, data_scaler.feature_list())
            input_features = (input_features - data_scaler.mean()) / data_scaler.scale()
            scores = model.predict(input_features)

            nn_p_hh = scores[:,0]
            nn_p_tt = scores[:,1]
            nn_p_wt = scores[:,2]
            nn_p_zjets = scores[:,3]

            otf_var = nn_p_hh
            #otf_var = np.log( nn_p_tt / (nn_p_hh + nn_p_wt + nn_p_zjets) )
            #ok_idx = valid_idx(otf_var)
            #otf_var = otf_var[ok_idx]
            #weights = weights[ok_idx]
            #lwtnn_var = lwtnn_var[ok_idx]

            lwtnn_data.extend(lwtnn_var)
            otf_data.extend(otf_var)
            weight_data.extend(weights)
            weight2_data.extend(weights**2)

    ## histos
    bw = 0.05
    bins = np.arange(0,1+bw,bw)
    hist_lwtnn, _ = np.histogram(lwtnn_data, bins = bins, weights = weight_data)
    hist_otf, _ = np.histogram(otf_data, bins = bins, weights = weight_data)
    sumw2_hist, _ = np.histogram(lwtnn_data, bins = bins, weights = weight2_data)

    print('lwtnn: {}'.format(hist_lwtnn[:20]))
    print('otf  : {}'.format(hist_otf[:20]))

    bin_centers = bins + 0.5 * bw
    ratio_hist = hist_lwtnn / hist_otf
    bin_centers = bin_centers[:-1]

    fig, ax = plt.subplots(2,1)
    #ax[0].hist( [otf_data], bins = bins, weights = [weight_data], label = ['otf'], histtype = 'step', color = ['b'] )
    ax[0].hist( [lwtnn_data, otf_data], bins = bins, weights = [weight_data, weight_data], label = ['lwtnn', 'otf'], histtype = 'step', color = ['r', 'b'] )
    ax[1].plot( bin_centers, ratio_hist, label = 'lwtnn/otf' )
    ax[0].set_ylabel('Entries')
    ax[1].set_xlabel('$hh$ NN score')
    ax[1].set_ylabel('lwtnn compute / keras compute')
    fig.savefig('test_otf_lwtnn_comp.pdf', bbox_inches = 'tight', dpi = 200)


def main() :
    parser = argparse.ArgumentParser(description = "Plot stored NN variables from LWTNN and those calculated OTF")
    parser.add_argument("-i", "--input", help = "Provide input file", required = True)
    parser.add_argument("--nn-dir", help = "Provide a neural network dir and load this (loads whatever is inside of training/ and scaling/", required = True)
    args = parser.parse_args()

    make_plots(args)

    

if __name__ == "__main__" :
    main()
