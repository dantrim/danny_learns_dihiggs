#!/usr/bin/env python

from __future__ import print_function # just in case

import argparse
import sys
import os

import h5py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from train import DataScaler, floatify
from overlay_nn import Sample, chunk_generator, valid_idx
from roc_nn_and_cut import load_stored_model

def all_vars() :

    v = {}
    v['p_hh'] = { 'bounds' : [0.05, 0, 1], 'name' : '$p_{hh}$' }
    v['p_tt'] = { 'bounds' : [0.05, 0, 1], 'name' : '$p_{t\\bar{t}}$' }
    v['d_hh'] = { 'bounds' : [0.5, -12, 10], 'name' : '$d_{hh}$' }
    #v['d_hh'] = { 'bounds' : [1, -30, 10], 'name' : '$d_{hh}$' }
    v['d_tt'] = { 'bounds' : [0.5, -15, 5], 'name' : '$d_{t\\bar{t}}$' }
    #v['d_tt'] = { 'bounds' : [1, -30, 30], 'name' : '$d_{t\\bar{t}}$' }
    v['d_wt'] = { 'bounds' : [0.2, -10, 3], 'name' : '$d_{Wt}$' }
    v['d_z'] = { 'bounds' : [1, -40, 8], 'name' : '$d_{Z}$' }

    v['mll'] = { 'bounds' : [2, 15, 130], 'name' : '$m_{\\ell \\ell}$ [GeV]' }
    v['mt2_bb'] = { 'bounds' : [5, 0, 200], 'name' : '$m_{t2}^{bb}$' }
    v['mbb'] = { 'bounds' : [1, 80, 180], 'name' : '$m_{bb}$ [GeV]' }
    v['met'] = { 'bounds' : [5, 0, 200], 'name' : 'Missing Transverse Momentum [GeV]' }
    v['HT2'] = { 'bounds' : [10, 0, 500], 'name' : '$H_{T2}$ [GeV]' }
    v['HT2Ratio'] = { 'bounds' : [0.05, 0, 1], 'name' : '$H_{T2}^{R}$' }
    v['dRll'] = { 'bounds' : [0.1, 0, 6], 'name' : '$\\Delta R _{\\ell \\ell}$' }
    v['bj0_pt'] = { 'bounds' : [5, 0, 200], 'name' : 'Lead $b$-jet $p_{T}$ [GeV]' }
    v['bj1_pt'] = { 'bounds' : [2, 0, 150], 'name' : 'Sub-lead $b$-jet $p_{T}$ [GeV]' }
    v['bj0_phi'] = { 'bounds' : [0.1, -3.2, 3.2], 'name' : 'Lead $b$-jet $\\phi$' }
    v['bj1_phi'] = { 'bounds' : [0.1, -3.2, 3.2], 'name' : 'Sub-lead $b$-jet $\\phi$' }

    return v

def make_plot(var_dict, sample, scaler, model, args) :

    x_data = []
    y_data = []
    w_data = []

    total_read = 0

    disc_for_x = args.varX

    with h5py.File(sample.filename, 'r', libver = 'latest') as sample_file :
        if 'superNt' not in sample_file :
            print('ERROR superNt dataset not found in input file (={})'.format(sample.filename))
            sys.exit()
        dataset = sample_file['superNt']
        is_first = True
        if 'data' in sample.filename :
            is_first = False
        for chunk in chunk_generator(dataset, chunksize = 19000) :
            if is_first :
                is_first = False
                continue

            total_read += chunk.size
            if total_read > 1e6 : break

            idx = (chunk['nBJets'] >= 2)

            weights = chunk[idx]
            chunk = chunk[idx]

            input_features = chunk[scaler.feature_list()]
            input_features = floatify(input_features, scaler.feature_list())
            input_features = (input_features - scaler.mean()) / scaler.scale()
            scores = model.predict(input_features)

            p_hh = scores[:,0]
            p_tt = scores[:,1]
            p_wt = scores[:,2]
            p_z  = scores[:,3]

            i_x_data = p_hh
            if 'p_' in disc_for_x :
                i_x_data = { 'p_hh' : p_hh, 'p_tt' : p_tt, 'p_wt' : p_wt, 'p_z' : p_z } [ disc_for_x ]

            elif 'd_' in disc_for_x :
                num_data = { 'd_hh' : p_hh, 'd_tt' : p_tt, 'd_wt' : p_wt, 'd_z' : p_z } [disc_for_x]
                den_data = { 'd_hh' : (p_tt + p_wt + p_z), 'd_tt' : (p_wt + p_hh + p_z), 'd_wt' : (p_hh + p_tt + p_z), 'd_z' : (p_tt + p_wt + p_hh) } [ disc_for_x ]
                disc = np.log(num_data / den_data)
                idx = valid_idx(disc)

                p_hh = p_hh[idx]
                p_tt = p_tt[idx]
                p_wt = p_wt[idx]
                p_z = p_z[idx]

                disc = disc[idx]
                weights = weights[idx]
                chunk = chunk[idx]
                i_x_data = disc

            i_y_data = None
            if 'p_' in args.varY or 'd_' in args.varY :
                if 'p_' in args.varY :
                    i_y_data = { 'p_hh' : p_hh, 'p_tt' : p_tt, 'p_wt' : p_wt, 'p_z' : p_z } [ args.varY ]
                elif 'd_' in args.varY :
                    num_data = { 'd_hh' : p_hh, 'd_tt' : p_tt, 'd_wt' : p_wt, 'd_z' : p_z } [ args.varY ]
                    den_data = { 'd_hh' : (p_tt + p_wt + p_z), 'd_tt' : (p_wt + p_hh + p_z), 'd_wt' : (p_hh + p_tt + p_z), 'd_z' : (p_tt + p_wt + p_hh) } [ args.varY ]
                    y_disc = np.log(num_data / den_data)
                    #idx = valid_idx(y_disc)
                    #y_disc = y_disc[idx]
                    #weights = weights[idx]
                    #chunk = chunk[idx]
                    i_y_data = y_disc
            else :
                i_y_data = chunk[args.varY]

            x_data.extend(list(i_x_data))
            y_data.extend(list(i_y_data))
            w_data.extend(list(weights))

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    w_data = np.array(w_data)

    fig, ax = plt.subplots(1,1)
    ax.grid(color = 'k', which = 'both', linestyle = '-', lw = 0.5, alpha = 0.1)
    ax.tick_params(axis = 'both', which = 'both', direction = 'in',
        labelleft = True, bottom = True, top = True, right = True, left = True)

    var_dict = all_vars()
    x_bounds = var_dict[disc_for_x]['bounds']
    y_bounds = var_dict[args.varY]['bounds']

    x_label = var_dict[disc_for_x]['name']
    y_label = var_dict[args.varY]['name']

    x_edges = np.arange(x_bounds[1], x_bounds[2] + x_bounds[0], x_bounds[0])
    y_edges = np.arange(y_bounds[1], y_bounds[2] + y_bounds[0], y_bounds[0])

    bins = [x_edges, y_edges]

    ax.set_xlabel(x_label, horizontalalignment = 'right', x = 1)
    ax.set_ylabel(y_label, horizontalalignment = 'right', y = 1)

    print('x_data shape = {}'.format(x_data.shape))
    print('y_data shape = {}'.format(y_data.shape))
    h, x, y = np.histogram2d(x_data, y_data, bins = bins, normed = False)
    #integral = h.sum()
    #h = h / integral
    imextent = list( (min(x_edges), max(x_edges))) + list((min(y_edges), max(y_edges)))
    ax.set_facecolor('lightgrey')
    h = h.T
    im = ax.imshow(h, origin = 'lower', cmap = 'coolwarm', aspect = 'auto', interpolation = 'nearest', extent = imextent, norm = LogNorm())
    ax.contour(h, levels = [1,3,10], colors = 'black', extent = imextent)
    cb = fig.colorbar(im)

    process = ''
    if 'wt' in sample.filename :
        process = 'wt'
    elif '123456' in sample.filename :
        process = 'hh'
    elif 'ttbar' in sample.filename or '410009' in sample.filename :
        process = 'ttbar'
    elif 'zll' in sample.filename or 'zjets' in sample.filename :
        process = 'zjets'
    elif 'ztt' in sample.filename :
        process = 'zjets_tt'
    elif 'data' in sample.filename :
        process = 'data'

    ax.text(0.85,0.93, process, weight = 'bold', transform = ax.transAxes)

    outname = './plots_input_output/input_output_2D_{}_{}_{}.pdf'.format(process, disc_for_x, args.varY)
    print(' >> saving plot to: {}'.format(os.path.abspath(outname)))
    fig.savefig(outname, bbox_inches = 'tight', dpi = 200)

def make_plots(var_dict, sample, args) :


    data_scaler, loaded_model = load_stored_model(args.nn_dir)

    n_vars = len(var_dict)
    for ivar, var in enumerate(var_dict) :
        print(' >> [{}/{}] {}'.format(ivar+1, n_vars, var))
#        if 'd_' in var or 'p_' in var : continue
        args.varY = var

        make_plot(var_dict[var], sample, data_scaler, loaded_model, args)


def main() :
    parser = argparse.ArgumentParser(description = 'Plot NN inputs vs NN outputs')
    parser.add_argument('-i', '--input', required = True,
        help = 'Input sample to make plots from')
    parser.add_argument('-n', '--name', required = True,
        help = 'Sample name')
    parser.add_argument('--varY', default = '',
        help = 'Provide input variable to plot against the NN outputs')
    parser.add_argument('--varX', required = True,
        help = 'Provide output variable to plot on the X axis')
    parser.add_argument('-d', '--disc', default = False, action = 'store_true',
        help = 'Plot inputs vs output log-ratio discriminants')
    parser.add_argument('--nn-dir', required = True,
        help = 'Provide a neural network dir and load this')
    args = parser.parse_args()

    if not os.path.isfile(args.input) :
        print('ERROR did not find provided input file (={})'.format(args.input))
        sys.exit()

    var_dict = all_vars()
    if args.varY != '' :
        if args.varY not in var_dict :
            print('ERROR requested variable ({}) not found in var dict'.format(args.varY))
            sys.exit()
        tmp = {}
        tmp[args.varY] = var_dict[args.varY]
        var_dict = tmp

    sample = Sample(args.name, args.input, '')
    make_plots(var_dict, sample, args)


if __name__ == '__main__' :
    main()

