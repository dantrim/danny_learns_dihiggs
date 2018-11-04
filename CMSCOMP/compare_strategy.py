#!/usr/bin/env python

from __future__ import print_function # just in case

import os, sys, argparse
import h5py
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from overlay_nn import Sample, chunk_generator, valid_idx
from train import DataScaler, floatify
from roc_nn_and_cut import load_stored_model
import scipy.interpolate as interpolate
from scipy.interpolate import spline # for smoothing

filedir = '/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/samples/nov2_cmscompstuff/'
hh_file = '{}/CENTRAL_123456_nov2.h5'.format(filedir)
top_file = '{}/top_bkg.h5'.format(filedir)
zll_file = '{}/sherpa_zll.h5'.format(filedir)
ztt_file = '{}/sherpa_ztt.h5'.format(filedir)

class Histo :
    def __init__(self, name = '') :
        self.name = name
        self.lcd = 0
        self.histo_data = None
        self.weights = None
        self.sumw2_histo_data = None

def get_targetted_multi_score(sample, target = '', scaler = None, model = None) :

    lcd = 0.0
    histo_data = []
    weight_data = []
    w2_data = []

    total_read = 0

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
            weights = chunk['eventweight']

            lcd_idx = (chunk['nBJets']>=1)
            weights = weights[lcd_idx] * 36.1
            lcd += np.sum(weights)

            chunk = chunk[lcd_idx]
            more_idx = (chunk['nBJets']>=2) & (chunk['mbb']>110) & (chunk['mbb']<140)
            chunk = chunk[more_idx]
            weights = weights[more_idx]

            input_features = chunk[scaler.feature_list()]
            input_features = floatify(input_features, scaler.feature_list())
            input_features = (input_features - scaler.mean()) / scaler.scale()
            scores = model.predict(input_features)

            num_data = scores[:,0] # hh score

            bkg_score_idx = { 'top' : 1,
                            'zll' : 2,
                            'ztt': 3 } [target]
            den_data = scores[:,bkg_score_idx]

            ok_idx = den_data != 0
            num_data = num_data[ok_idx]
            den_data = den_data[ok_idx]
            weights = weights[ok_idx]

            data = np.log(num_data / den_data)
            ok_idx = (data > -np.inf) & (data < np.inf)
            data = data[ok_idx]
            weights = weights[ok_idx]

            histo_data.extend(data)
            weight_data.extend(weights)
            w2_data.extend(np.power(weights,2))

    h = Histo(sample.name)
    h.lcd = lcd
    h.weights = weight_data
    h.histo_data = histo_data
    h.sumw2_histo_data = w2_data

    return h

def get_single_nn_histo(sample, scaler, model) :

    lcd = 0.0
    histo_data = []
    weight_data = []
    w2_data = []

    total_read = 0

    with h5py.File(sample.filename, 'r', libver = 'latest') as sample_file :
        if 'superNt' not in sample_file :
            print('ERROR superNt dataset not found in input file (={})'.format(sample.filename))
            sys.exit()
        dataset = sample_file['superNt']
        if 'hh' in sample.name :
            dataset = dataset[19000:]
        for chunk in chunk_generator(dataset) :
            total_read += chunk.size
            if total_read > 1e6 : continue
            weights = chunk['eventweight']
            lcd_idx = (chunk['nBJets']>=1)
            weights = weights[lcd_idx] * 36.1
            lcd += np.sum(weights)
            chunk = chunk[lcd_idx]

            more_idx = (chunk['nBJets']>=2) & (chunk['mbb']>110) & (chunk['mbb']<140)
            chunk = chunk[more_idx]
            weights = weights[more_idx]

            input_features = chunk[scaler.feature_list()]
            input_features = floatify(input_features, scaler.feature_list())
            input_features = (input_features - scaler.mean()) / scaler.scale()
            scores = model.predict(input_features)

            num_data = scores[:,0]
            den_data = scores[:,1]

            ok_idx = den_data != 0
            num_data = num_data[ok_idx]
            den_data = den_data[ok_idx]
            weights = weights[ok_idx]

            data = np.log(num_data/den_data)
            ok_idx = (data>-np.inf) & (data<np.inf)
            data = data[ok_idx]
            weights = weights[ok_idx]

            histo_data.extend(data)
            weight_data.extend(weights)
            w2_data.extend(np.power(weights,2))

    h = Histo(sample.name)
    h.lcd = lcd
    h.weights = weight_data
    h.histo_data = histo_data
    h_sumw2_histo_data = w2_data

    return h

def draw_roc_curve(ax, lcds, sig_histo, bkg_histo, style = '-') :

    total_sig_yield, total_bkg_yield = lcds

    sig_eff = np.cumsum(sig_histo.histogram[::-1])[::-1]
    bkg_eff = np.cumsum(bkg_histo.histogram[::-1])[::-1]

    # balls BALLS
    min_cum_idx = bkg_eff > 1
    sig_eff = sig_eff[min_cum_idx]
    bkg_eff = bkg_eff[min_cum_idx]

    sig_eff = sig_eff / total_sig_yield
    bkg_eff = bkg_eff / total_bkg_yield
    bkg_rej = 1.0 / bkg_eff

    colors = {}
    colors['top'] = 'r'
    colors['zll'] = 'g'
    colors['ztt'] = 'm'
    colors['totalbkg'] = 'k'
    colors['totalbkg_top'] = 'r'
    colors['totalbkg_zll'] = 'g'
    colors['totalbkg_ztt'] = 'm'

    if style == '-' and 'totalbkg' not in bkg_histo.name :
        label = '{} - Multi-output NN'.format(bkg_histo.name)
    elif style == '--' and 'totalbkg' not in bkg_histo.name :
        label = '{} - Single-output NN'.format(bkg_histo.name)
    elif 'totalbkg' in bkg_histo.name :
        if bkg_histo.name == 'totalbkg' :
            label = 'Multi-output NN'
        elif 'top' in bkg_histo.name :
            label = 'Single-output NN ($t\\bar{t} + Wt$)'
        elif 'zll' in bkg_histo.name :
            label = 'Single-output NN ($Z\\rightarrow \\ell \\ell$)'
        elif 'ztt' in bkg_histo.name :
            label = 'Single-output NN ($Z\\rightarrow \\tau \\tau$)'
    else :
        label = bkg_histo.name


    ax.plot(sig_eff, bkg_rej, linewidth = 1, label = label, ls = style, color = colors[bkg_histo.name])

    return sig_eff, bkg_rej

def get_default_nn_histos(sample_dict, signal_name, bkg_names) :

    '''
    For each process, build up the discriminant histograms
    for the discriminants from the multi-output NN targetting that process.

    There will be N histograms for the signal process and
    N histograms for the N backgrounds
    '''

    # directory holding the multi-output nn
    nn_dir = './training_default_4output/'
    data_scaler, loaded_model = load_stored_model(nn_dir)

    fig, ax = plt.subplots(1,1)
    ax.set_yscale('log')
    ax.set_xlabel('Signal Efficiency, $\\varepsilon_{s}$', horizontalalignment = 'right', x = 1)
    ax.set_ylabel('Background Process Rejection, $1/\\varepsilon_{b}$', horizontalalignment = 'right', y = 1)
    ax.grid(color = 'k', which = 'both', linestyle = '-', lw = 0.5, alpha = 0.1)
    ax.tick_params(axis = 'both', which = 'both', labelsize = 16, direction = 'in',
        labelleft = True, bottom = True, top = True, right = True, left = True)
    ax.set_ylim([100, 2e5])

    bw = 1
    bins = np.arange(-100, 100+bw, bw)

    # get targeted curves for multi-output NN
    for bkg in bkg_names :

        h_sig = get_targetted_multi_score(sample_dict['hh'], target = bkg, scaler = data_scaler, model = loaded_model)
        h_bkg = get_targetted_multi_score(sample_dict[bkg], target = bkg, scaler = data_scaler, model = loaded_model)


        h_sig.histo_data = np.clip(h_sig.histo_data, bins[0], bins[-1])
        h_bkg.histo_data = np.clip(h_bkg.histo_data, bins[0], bins[-1])

        histo_sig, _ = np.histogram( h_sig.histo_data, bins = bins, weights = h_sig.weights)
        sumw2_sig, _ = np.histogram( h_sig.histo_data, bins = bins, weights = h_sig.sumw2_histo_data)
        histo_bkg, _ = np.histogram( h_bkg.histo_data, bins = bins, weights = h_bkg.weights)
        sumw2_bkg, _ = np.histogram( h_bkg.histo_data, bins = bins, weights = h_bkg.sumw2_histo_data)

        h_sig.histogram = histo_sig
        h_sig.sumw2 = sumw2_sig
        h_bkg.histogram = histo_bkg
        h_bkg.sumw2 = sumw2_bkg

        draw_roc_curve(ax, lcds = [h_sig.lcd,  h_bkg.lcd], sig_histo = h_sig, bkg_histo = h_bkg)

    # now get curves for each of the single-output NN's targetting individual backgrounds
    for bkg in bkg_names :

        nn_dir = { 'top' : './training_default_1output_top/',
                    'zll' : 'training_default_1output_zll/',
                    'ztt' : 'training_default_1output_ztt/' }[bkg]
        data_scaler, loaded_model = load_stored_model(nn_dir)

        h_sig = get_single_nn_histo(sample_dict['hh'], scaler = data_scaler, model = loaded_model)
        h_bkg = get_single_nn_histo(sample_dict[bkg], scaler = data_scaler, model = loaded_model)

        histo_sig, _ = np.histogram( h_sig.histo_data, bins = bins, weights = h_sig.weights)
        sumw2_sig, _ = np.histogram( h_sig.histo_data, bins = bins, weights = h_sig.sumw2_histo_data)
        histo_bkg, _ = np.histogram( h_bkg.histo_data, bins = bins, weights = h_bkg.weights)
        sumw2_bkg, _ = np.histogram( h_bkg.histo_data, bins = bins, weights = h_bkg.sumw2_histo_data)

        h_sig.histogram = histo_sig
        h_sig.sumw2 = sumw2_sig
        h_bkg.histogram = histo_bkg
        h_bkg.sumw2 = sumw2_bkg

        draw_roc_curve(ax, lcds = [h_sig.lcd, h_bkg.lcd], sig_histo = h_sig, bkg_histo = h_bkg, style = '--')

    

    ax.legend(loc = 'best', frameon = False)
    fig.savefig('roc_multi_vs_single_targetted.pdf', bbox_inches = 'tight', dpi = 200)


def make_targetted_roc_curves(sample_dict) :

    signal_name = 'hh'
    bkg_names = [key for key in sample_dict if key is not 'hh']

    #sig_histos_default, bkg_histos_dict_default = get_default_nn_histos(sample_dict, signal_name, bkg_names)
    get_default_nn_histos(sample_dict, signal_name, bkg_names)

def get_total_bkg_disc(sample, scaler = None, model = None) :

    lcd = 0.0
    histo_data = []
    weight_data = []
    w2_data = []
    total_read = 0

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
            weights = chunk['eventweight']
            lcd_idx = (chunk['nBJets']>=1)
            weights = weights[lcd_idx] * 36.1
            lcd += np.sum(weights)

            chunk = chunk[lcd_idx]
            more_idx = (chunk['nBJets']>=2) & (chunk['mbb']>100) & (chunk['mbb']<140)
            chunk = chunk[more_idx]
            weights = weights[more_idx]

            input_features = chunk[scaler.feature_list()]
            input_features = floatify(input_features, scaler.feature_list())
            input_features = (input_features - scaler.mean()) / scaler.scale()
            scores = model.predict(input_features)

            num_data = scores[:,0]
            den_data = scores[:,1]
            if scores.shape[1] > 2 :
                den_data += scores[:,2]
                den_data += scores[:,3]

            ok_idx = den_data != 0
            num_data = num_data[ok_idx]
            den_data = den_data[ok_idx]

            weights = weights[ok_idx]
            data = np.log(num_data / den_data)
            ok_idx = (data>-np.inf) & (data<np.inf)
            data = data[ok_idx]
            weights = weights[ok_idx]

            histo_data.extend(data)
            weight_data.extend(weights)
            w2_data.extend(np.power(weights,2))

    h = Histo(sample.name)
    h.lcd = lcd
    h.weights = weight_data
    h.histo_data = histo_data
    h.sumw2_histo_data = w2_data

    return h

def build_and_draw_roc_curve(ax, nn_dir, sample_dict, signal_name, bkg_names, label) :

    bw = 1
    bins = np.arange(-100,100+bw, bw)

    data_scaler, loaded_model = load_stored_model(nn_dir)

    all_bkg_histos = []
    h_sig = get_total_bkg_disc(sample_dict['hh'], scaler = data_scaler, model = loaded_model)
    for bkg in bkg_names :
        h_bkg = get_total_bkg_disc(sample_dict[bkg], scaler = data_scaler, model = loaded_model)
        all_bkg_histos.append(h_bkg)

    h_bkg = Histo(label)
    h_bkg.lcd = 0.
    h_bkg.weights = all_bkg_histos[0].weights
    h_bkg.histo_data = all_bkg_histos[0].histo_data
    h_bkg.sumw2_histo_data = all_bkg_histos[0].sumw2_histo_data
    for b in all_bkg_histos[1:] :
        h_bkg.lcd += b.lcd
        h_bkg.weights.extend(b.weights)
        h_bkg.histo_data.extend(b.histo_data)
        h_bkg.sumw2_histo_data.extend(b.sumw2_histo_data)

    h_sig.histo_data = np.clip(h_sig.histo_data, bins[0], bins[-1])
    h_bkg.histo_data = np.clip(h_bkg.histo_data, bins[0], bins[-1])

    histo_sig, _ = np.histogram( h_sig.histo_data, bins = bins, weights = h_sig.weights )
    histo_bkg, _ = np.histogram( h_bkg.histo_data, bins = bins, weights = h_bkg.weights )

    h_sig.histogram = histo_sig
    h_bkg.histogram = histo_bkg

    sig_eff_curve, bkg_rej_curve = draw_roc_curve(ax, lcds = [h_sig.lcd, h_bkg.lcd], sig_histo = h_sig, bkg_histo = h_bkg)

    return sig_eff_curve, bkg_rej_curve

def get_total_bkg_roc_curves(sample_dict, signal_name, bkg_names) :

    fig, ax = plt.subplots(1,1)
    ax.set_yscale('log')
    ax.set_xlabel('Signal Efficiency, $\\varepsilon_{s}$', horizontalalignment = 'right', x = 1)
    ax.set_ylabel('Total Background Rejection, $1/\\varepsilon_{B}$', horizontalalignment = 'right', y = 1)
    ax.grid(color = 'k', which = 'both', linestyle = '-', lw = 0.5, alpha = 0.1)
    ax.tick_params(axis = 'both', which = 'both', labelsize = 16, direction = 'in',
        labelleft = True, bottom = True, top = True, right = True, left = True)

    ax.set_xlim([-0.01, 0.21])
    ax.set_ylim([100,3e5])


    # default multi-output nn
    
    nn_dir = './training_default_4output/'
    split_nn_dir = { 'top' : './training_default_1output_top/',
                'zll' : 'training_default_1output_zll/',
                'ztt' : 'training_default_1output_ztt/' }

    build_and_draw_roc_curve(ax, nn_dir, sample_dict, signal_name, bkg_names, label = 'totalbkg')
    for bkg_target in split_nn_dir :
        nn_dir = split_nn_dir[bkg_target]
        label = 'totalbkg_{}'.format(bkg_target)
        build_and_draw_roc_curve(ax, nn_dir, sample_dict, signal_name, bkg_names, label = label)


    ax.legend(loc = 'best', frameon = False)
    fig.savefig('roc_multi_vs_single_total_bkg.pdf', bbox_inches = 'tight', dpi = 200)

def get_total_bkg_roc_curves_with_ratio(sample_dict, signal_name, bkg_names) :


    fig = plt.figure(figsize = (7,8))
    grid = GridSpec(100,100)
    upper_pad = fig.add_subplot(grid[0:75,:])
    lower_pad = fig.add_subplot(grid[80:100,:], sharex = upper_pad)

    #upper_pad.set_xticklabels([])
    xlow, xhigh = 0.02, 0.26
    upper_pad.set_xlim([xlow, xhigh])
    lower_pad.set_xlim([xlow, xhigh])

    upper_pad.set_ylim([100,3e5])
    upper_pad.set_yscale('log')

    for ax in [upper_pad, lower_pad] :
        ax.tick_params(axis = 'both', which = 'both', labelsize = 16, direction = 'in',
            labelleft = True, bottom = True, top = True, left = True)
        ax.grid(color = 'k', which = 'both', linestyle = '-', lw = 1, alpha = 0.1)

    lower_pad.set_xlabel('Signal Efficiency, $\\varepsilon_{s}$', horizontalalignment = 'right', x = 1)
    upper_pad.set_ylabel('Total Background Rejection, $1/\\varepsilon_{B}$', horizontalalignment = 'right', y = 1)
    lower_pad.set_ylabel('Ratio', fontsize = 18)

    # default multi-output nn
    
    nn_dir = './training_default_4output/'
    split_nn_dir = { 'top' : './training_default_1output_top/',
                'zll' : 'training_default_1output_zll/',
                'ztt' : 'training_default_1output_ztt/' }

    sig_eff_sig, bkg_rej_sig = build_and_draw_roc_curve(upper_pad, nn_dir, sample_dict, signal_name, bkg_names, label = 'totalbkg')
    sig_eff_top, bkg_rej_top = None, None
    for bkg_target in split_nn_dir :
        nn_dir = split_nn_dir[bkg_target]
        label = 'totalbkg_{}'.format(bkg_target)
        sig_eff_bkg, bkg_rej_bkg = build_and_draw_roc_curve(upper_pad, nn_dir, sample_dict, signal_name, bkg_names, label = label)
        if bkg_target == 'top' :
            sig_eff_top, bkg_rej_top = sig_eff_bkg, bkg_rej_bkg

    

    # ratio of multi-output to ttbar+wt single output NN
    #lower_pad.set_ylim([0.84, 1.21])
    ratio_rej = np.divide(bkg_rej_sig, bkg_rej_top)

    size = sig_eff_sig.shape[0]


    x_sig = sig_eff_sig[::-1]
    y_sig = bkg_rej_sig[::-1]
    x_bkg = sig_eff_top[::-1]
    y_bkg = bkg_rej_top[::-1]

    print('x_sig BEFORE = {}'.format(x_sig))

    sig_idx = x_sig < 0.27
    x_sig = x_sig [sig_idx]
    y_sig = y_sig [sig_idx]

    print('x_sig AFTER = {}'.format(x_sig))

    bkg_idx = x_bkg < 0.27
    x_bkg = x_bkg [bkg_idx]
    y_bkg = y_bkg [ bkg_idx]
    

    x_new = np.linspace(x_sig.min(), x_sig.max(), 100)
    y_smooth = spline(x_sig, y_sig, x_new)
    sig_y_smooth = spline(x_sig, y_sig, x_new)
    bkg_y_smooth = spline(x_bkg, y_bkg, x_new)

    smooth_ratio = np.divide(sig_y_smooth, bkg_y_smooth)
    lower_pad.plot(x_new[1:-1], smooth_ratio[1:-1], 'k-')

    avg_ratio = np.mean(smooth_ratio)
    lower_pad.plot([xlow, xhigh], [avg_ratio, avg_ratio], 'b--')
    lower_pad.plot([xlow, xhigh], [1.0, 1.0], 'r-', zorder=0)

    upper_pad.legend(loc = 'best', frameon = False)
    fig.savefig('roc_multi_vs_single_total_bkg_ratio.pdf', bbox_inches = 'tight', dpi = 200)


def make_total_bkg_roc_curves(sample_dict) :

    signal_name = 'hh'
    bkg_names = [key for key in sample_dict if key is not 'hh']
    #get_total_bkg_roc_curves(sample_dict, signal_name, bkg_names)
    get_total_bkg_roc_curves_with_ratio(sample_dict, signal_name, bkg_names)

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

    #make_targetted_roc_curves(samples)
    make_total_bkg_roc_curves(samples)
    


#__________________________________________
if __name__ == '__main__' :
    main()
