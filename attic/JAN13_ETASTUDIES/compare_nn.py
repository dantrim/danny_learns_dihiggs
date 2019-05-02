#!/usr/bin/env python

from __future__ import print_function # jic

import os, sys, argparse
import h5py
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from overlay_nn import Sample, chunk_generator, valid_idx
from train import DataScaler, floatify
from roc_nn_and_cut import load_stored_model
import scipy.interpolate as interpolate
from scipy.interpolate import spline

from compare_strategy import build_and_draw_roc_curve

filedir = '/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/samples/nov2_cmscompstuff/'
hh_file = '{}/CENTRAL_123456_nov2.h5'.format(filedir)
top_file = '{}/top_bkg.h5'.format(filedir)
zll_file = '{}/sherpa_zll.h5'.format(filedir)
ztt_file = '{}/sherpa_ztt.h5'.format(filedir)

def draw_roc_and_ratio(sigs = [], bkgs = [], label = '', pads = []) :

    x_def = sigs[0][::-1]
    y_def = bkgs[0][::-1]

    x_com = sigs[1][::-1]
    y_com = bkgs[1][::-1]

    sig_idx = x_def < 0.27
    x_def = x_def [ sig_idx ]
    y_def = y_def [ sig_idx ]

    com_idx = x_com < 0.27
    x_com = x_com [ com_idx ]
    y_com = y_com [ com_idx ]

    x_new = np.linspace(x_def.min(), x_def.max(), 100)
    y_def_smooth = spline(x_def, y_def, x_new)
    y_com_smooth = spline(x_com, y_com, x_new)

    colors = {}
    colors['Default'] = 'k'
    colors['DefaultEta'] = 'r'
    colors['NoHlvlPlusEta'] = 'g'
    colors['DefaultMt2'] = 'b'
    colors['DefaultMt2Eta'] = 'm'


    ratio_smooth = np.divide(y_com_smooth, y_def_smooth)
    if label != 'Default' :
        pads[1].plot(x_new[1:-1], ratio_smooth[1:-1], '{}-'.format(colors[label]))
    


def compare_networks(sample_dict, signal_name, bkg_names) :

    fig = plt.figure(figsize = (7,8))
    grid = GridSpec(100,100)
    upper_pad = fig.add_subplot(grid[0:75, :])
    lower_pad = fig.add_subplot(grid[80:100,:], sharex = upper_pad)

    xlow, xhigh = 0.02, 0.26
    upper_pad.set_xlim([xlow, xhigh])
    lower_pad.set_xlim([xlow, xhigh])
    upper_pad.set_yscale('log')
    for ax in [upper_pad, lower_pad] :
        ax.tick_params(axis = 'both', which = 'both', labelsize = 16, direction = 'in',
            labelleft = True, bottom = True, top = True, left = True)
        ax.grid(color = 'k', which = 'both', linestyle = '-', lw = 1, alpha = 0.1)

    lower_pad.set_xlabel('Signal Efficiency, $\\varepsilon_{s}$', horizontalalignment = 'right', x = 1)
    upper_pad.set_ylabel('Total Background Rejection, $1/\\varepsilon_{B}$', horizontalalignment = 'right', y = 1)
    lower_pad.set_ylabel('Ratio')

    lower_pad.set_ylim([0, 2])

    #NN dir
    nn_dir = './training_default/'
    sig_eff_def, bkg_rej_def = build_and_draw_roc_curve(upper_pad, nn_dir, sample_dict, signal_name, bkg_names, label = 'Default')
    nn_dir = './training_default_plusEta/'
    sig_eff_defEta, bkg_rej_defEta = build_and_draw_roc_curve(upper_pad, nn_dir, sample_dict, signal_name, bkg_names, label = 'DefaultEta')
    nn_dir = './training_nohlvl_eta/'
    sig_eff_redEta, bkg_rej_redEta = build_and_draw_roc_curve(upper_pad, nn_dir, sample_dict, signal_name, bkg_names, label = 'NoHlvlPlusEta')
    nn_dir = './training_default_plusMt2/'
    sig_eff_defMt2, bkg_rej_defMt2 = build_and_draw_roc_curve(upper_pad, nn_dir, sample_dict, signal_name, bkg_names, label = 'DefaultMt2')
    nn_dir = './training_default_plusEtaplusMt2/'
    sig_eff_defMt2Eta, bkg_rej_defMt2Eta = build_and_draw_roc_curve(upper_pad, nn_dir, sample_dict, signal_name, bkg_names, label = 'DefaultMt2Eta')

    draw_roc_and_ratio(sigs = [sig_eff_def, sig_eff_def], bkgs = [bkg_rej_def, bkg_rej_def], label = 'Default', pads = [upper_pad, lower_pad])
    draw_roc_and_ratio(sigs = [sig_eff_def, sig_eff_defEta], bkgs = [bkg_rej_def, bkg_rej_defEta], label = 'DefaultEta', pads = [upper_pad, lower_pad])
    draw_roc_and_ratio(sigs = [sig_eff_def, sig_eff_redEta], bkgs = [bkg_rej_def, bkg_rej_redEta], label = 'NoHlvlPlusEta', pads = [upper_pad, lower_pad])
    draw_roc_and_ratio(sigs = [sig_eff_def, sig_eff_defMt2], bkgs = [bkg_rej_def, bkg_rej_defMt2], label = 'DefaultMt2', pads = [upper_pad, lower_pad])
    draw_roc_and_ratio(sigs = [sig_eff_def, sig_eff_defMt2Eta], bkgs = [bkg_rej_def, bkg_rej_defMt2Eta], label = 'DefaultMt2Eta', pads = [upper_pad, lower_pad])

    upper_pad.legend(loc = 'best', frameon = False)
    fig.align_ylabels()
    fig.savefig('roc_compare_nn.pdf', bbox_inches = 'tight', dpi = 200)


def main() :
    hh_sample = Sample('hh', hh_file, '')
    top_sample = Sample('top', top_file, '')
    zll_sample = Sample('zll', zll_file, '')
    ztt_sample = Sample('ztt', ztt_file, '')

    samples = {}
    samples['hh'] = hh_sample
    samples['top'] = top_sample
    samples['zll'] = zll_sample
    samples['ztt'] = ztt_sample

    signal_name = 'hh'
    bkg_names = [key for key in samples if key is not 'hh']
    compare_networks(samples, signal_name, bkg_names)

#_______
if __name__ == '__main__' :
    main()

