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

from matplotlib.lines import Line2D

filedir = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/samples/bdefs/"
hh_file = "{}/CENTRAL_123456.h5".format(filedir)
tt_file = "{}/CENTRAL_410009_0to10.h5".format(filedir)
wt_file = "{}/wt_bkg.h5".format(filedir)
zll_file = "{}/sherpa_zll.h5".format(filedir)
ztt_file = "{}/sherpa_ztt.h5".format(filedir)

model_dir = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/ml_training_hlvl_nombbmt2/"

def get_data(sample, scaler, model, to_do) :

    data = []
    w = []
    name = ""
    total_read = 0.0

    with h5py.File(sample.filename, 'r', libver = 'latest') as sample_file :
        if 'superNt' not in sample_file :
            print('ERROR superNt dataset not found in input file (={})'.format(sample.filename))
            sys.exit()
        dataset = sample_file['superNt']
        if 'hh' in sample.name :
            dataset = dataset[19000:]
        for chunk in chunk_generator(dataset) :
            total_read += chunk.size
            if total_read > 1e6 : break
            print('{} > {}'.format(sample.name, total_read))
            chunk = chunk[ (chunk['nBJets'] >= 1 ) ]
            weights = chunk['eventweight'] * 36.1
            input_features = chunk[scaler.feature_list()]
            input_features = floatify(input_features, scaler.feature_list())
            input_features = (input_features - scaler.mean()) / scaler.scale()
            scores = model.predict(input_features)

            #to_do = "hh"
            p_hh = scores[:,0]
            p_tt = scores[:,1]
            p_wt = scores[:,2]
            p_z  = scores[:,3]

            hist_data = None
            if 'd_' in to_do :
                num_data = None
                den_data = None
                if to_do == 'd_hh' :
                    num_data = p_hh
                    den_data = (p_tt + p_wt + p_z)
                elif to_do == 'd_tt' :
                    num_data = p_tt
                    den_data = (p_hh + p_wt + p_z)
                elif to_do == 'd_wt' :
                    num_data = p_wt
                    den_data = (p_hh + p_tt + p_z)
                elif to_do == 'd_z' :
                    num_data = p_z
                    den_data = (p_hh + p_tt + p_wt)

                d = np.log(num_data / den_data)
                idx = (d > -np.inf) & (d < np.inf)
                d = d[idx]
                weights = weights[idx]
                hist_data = d[:]

            else :
                hist_data = { "hh" : p_hh,
                                "tt" : p_tt,
                                "wt" : p_wt,
                                "z" : p_z } [ to_do ]

            name = { "hh" : "$p_{hh}$",
                        "tt" : "$p_{t\\bar{t}}$",
                        "wt" : "$p_{Wt}$",
                        "z" : "$p_{Z}$" } [ to_do.replace('d_','') ]

            if 'd_' in to_do :
                name = name.replace('p_', 'd_')

            data.extend(hist_data)
            w.extend(weights)

    return data, w, name

def make_plots(signal_samples, bkg_samples, to_do) :

    all_samples = [signal_samples[0]]
    all_samples.extend([b for b in bkg_samples])

    histos = []
    weights = []
    x_label = ""
    labels = [s.name for s in all_samples]

    nice_names = { 'hh' : '$hh$',
                    'tt' : '$t\\bar{t}$',
                    'wt' : '$Wt$',
                    'zjets_ll' : '$Z+$jets' }

    data_scaler, loaded_model = load_stored_model(model_dir)

    for sample in all_samples :
        hist_data, hist_weights, x_label = get_data(sample, data_scaler, loaded_model, to_do)
        histos.append(hist_data)
        weights.append(hist_weights)

    fig, ax = plt.subplots(1,1)
    ax.set_yscale('log')
    ax.set_ylim([1e-2,1e2])
    ax.tick_params(axis = 'both', which = 'both', labelsize = 16, direction = 'in',
                    labelleft = True, bottom = True, top = True, right = True, left = True)
    ax.grid(color = 'k', which = 'both', linestyle = '--', lw = 0.5, alpha = 0.1)

    bounds = [0.05, 0, 1]
    bw = bounds[0]
    xlo = bounds[1]
    xhi = bounds[2]

    if 'd_' in to_do :
        ax.set_ylim([1e-3,1])
#        bounds = [2, -30, 20] # hh
        bounds = [1, -20, 8] # tt
        bounds = [1, -15, 6] # wt
        bounds = [2, -30, 12] # z
        #bounds = [2, -30, 20]
        bw = bounds[0]
        xlo = bounds[1]
        xhi = bounds[2]

    bin_edges = np.arange(xlo, xhi + bw, bw)
    bin_centers = bin_edges + 0.5 * bw
    bin_centers = bin_centers[:-1]

    ax.set_xlim(xlo, xhi)
    ax.set_xlabel(x_label, horizontalalignment = 'right', x = 1.0)
    ax.set_ylabel('a.u.', horizontalalignment = 'right', y = 1.0)

    for ihist, hist in enumerate(histos) :
        ax.hist(hist, weights = weights[ihist], bins = bin_edges, label = labels[ihist], color = all_samples[ihist].color, density = True, histtype = 'step', lw = 1.5)
#    ax.hist(histos, weights = weights, bins = bin_edges, label = labels, density = True, histtype = 'step', lw = 1.5)
    handles = [Line2D([0],[0], color = s.color) for s in all_samples]
    labels = [nice_names[s.name] for s in all_samples]
    ax.legend(handles, labels, loc = 'upper left', frameon = False)

    outname = 'nn_score_{}.pdf'.format(to_do)
    fig.savefig(outname, bbox_inches = 'tight', dpi = 200)

def main() :
    #parser = argparse.ArgumentParser(description = 'Plot stored NN scores and discriminants')

    hh_sample = Sample("hh", hh_file, "b")
    tt_sample = Sample("tt", tt_file, "r")
    wt_sample = Sample("wt", wt_file, "m")
    z_sample = Sample("zjets_ll", zll_file, "g")

    signal_samples = [hh_sample]
    #bkg_samples = [wt_sample, z_sample]
    bkg_samples = [tt_sample, wt_sample, z_sample]

#    for to_do in [ 'hh', 'tt', 'wt', 'z' ][1:] :
    for to_do in [ 'd_z' ] : #, 'd_tt', 'd_wt', 'd_z' ] :
        make_plots(signal_samples, bkg_samples, to_do)

if __name__ == '__main__' :
    main()
