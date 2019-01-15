#!/usr/bin/env python

from __future__ import print_function

import os, sys, argparse
import h5py
import numpy as np

from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#filedir = '/data/uclhc/uci/user/dantrim/ntuples/n0306/c_dec20/mc/'
filedir = '/Users/dantrim/Desktop/'
filename = '{}/z_all.h5'.format(filedir)

class Sample :
    def __init__(self, name = "", filename = "", color = "") :
        self.name = name
        self.filename = filename
        self.color = color

def chunk_generator(input_h5_dataset, chunksize = 100000) :
    for x in range(0, input_h5_dataset.size, chunksize) :
        yield input_h5_dataset[x:x+chunksize]

def make_z_plot(sample) :

    print('make_z_plot')

    fig = plt.figure(figsize = (7,8))
    grid = GridSpec(100, 100)
    upper_pad = fig.add_subplot(grid[0:75,:])
    lower_pad = fig.add_subplot(grid[80:100,:], sharex = upper_pad)

    pads = [upper_pad, lower_pad]
    for ax in pads :
        ax.tick_params(axis = 'both', which = 'both', labelsize = 12, direction = 'in',
            labelleft = True, bottom = True, top = True, left = True)
        ax.grid(color = 'k', which = 'both', linestyle = '-', lw = 1, alpha = 0.1)

    lower_pad.set_xlabel('$d_{hh}$', horizontalalignment = 'right', x = 1, fontsize = 16)
    upper_pad.set_ylabel('a.u.', horizontalalignment = 'right', y = 1, fontsize = 14)
    lower_pad.set_ylabel('On-Z / Off-Z', fontsize = 14)

    hist_data_veto = []
    weight_data_veto = []

    hist_data_noveto = []
    weight_data_noveto = []

    total_read = 0

    print('about to open file {}'.format(sample.filename))
    with h5py.File(sample.filename, 'r', libver = 'latest') as sample_file :

        if 'superNt' not in sample_file :
            print('ERROR superNt dataset not found in input file (={})'.format(sample.filename))
            sys.exit()
        dataset = sample_file['superNt']
        print('about to start loop')

        for chunk in chunk_generator(dataset) :
            #if total_read >= 5e5 : break
            total_read += chunk.size
            print('total read: {}'.format(total_read))


            # require at least 2 b-tagged jets
            bjet_mult = chunk['nBJets']
            idx_bj = bjet_mult >= 2
            chunk = chunk[idx_bj]

            #mtbb = chunk['mt2_bb']
            #idx_mtbb = mtbb > 30
            #chunk = chunk[idx_mtbb]

            weights = chunk['eventweight']
            #pos_w = weights >= 0
            #weights = weights[pos_w]
            #chunk = chunk[pos_w]
            
            mll = chunk['mll']

            # Z-veto
            pass_lo = mll < 71.2
            pass_hi = mll > 111.2
            idx_veto = (pass_lo | pass_hi)
            data_veto = chunk[idx_veto]['NN_d_hh']
            weights_veto = weights[idx_veto]

            hist_data_veto.extend(data_veto)
            weight_data_veto.extend(weights_veto)
            

            # Inverted Z-veto
            pass_lo = mll > 71.2
            pass_hi = mll < 111.2
            idx_noveto = (pass_lo & pass_hi)
            data_noveto = chunk[idx_noveto]['NN_d_hh']
            weights_noveto = weights[idx_noveto]

            hist_data_noveto.extend(data_noveto)
            weight_data_noveto.extend(weights_noveto)


    x_high = 10
    x_low = 0
    bin_width = 0.5
    bin_edges = np.arange(x_low - bin_width, x_high + 2*bin_width, bin_width)
    bin_centers = bin_edges + 0.5 * bin_width
    bin_centers = bin_centers[:-1]

    data_veto = np.array(hist_data_veto)
    weight_veto = np.array(weight_data_veto)
    data_noveto = np.array(hist_data_noveto)
    weight_noveto = np.array(weight_data_noveto)

    y_veto, _ = np.histogram(data_veto, bins = bin_edges, weights = weight_veto)
    y_noveto, _ = np.histogram(data_noveto, bins = bin_edges, weights = weight_noveto)

    total_yield_veto = np.sum(y_veto)
    total_yield_noveto = np.sum(y_noveto)

    y_veto /= total_yield_veto
    y_noveto /= total_yield_noveto

    # set x-ranges
    upper_pad.set_ylim([0, 0.3])
    lower_pad.set_ylim([0, 4])
    upper_pad.set_xlim([x_low, x_high])
    lower_pad.set_xlim([x_low, x_high])

    # draw upper pad
    print('len bin_centers = {}'.format(len(bin_centers)))
    print('len y_veto      = {}'.format(len(y_veto)))
    where = 'mid'

    upper_pad.step(bin_centers, y_veto, where = where, color = 'k', label = 'Off-Z')
    upper_pad.step(bin_centers, y_noveto, where = where, color = 'r', label = 'On-Z')
    upper_pad.legend(loc = 'best', frameon = False)

    # draw ratio
    ratio_y = y_noveto / y_veto
    lower_pad.step(bin_centers, ratio_y, where = 'mid', color = 'r')
    lower_pad.plot([x_low, x_high], [1.0, 1.0], 'r--', lw = 1)
    
    print('saving...')
    fig.align_ylabels()
    fig.savefig('plot_zcr_dhh_zveto.pdf', bbox_inches = 'tight', dpi = 200)


def main() :

    z_sample = Sample('zjets', filename, '')
    make_z_plot(z_sample)
#_______________________________
if __name__ == '__main__' :
    main()
