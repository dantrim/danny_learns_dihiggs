#!/usr/bin/env python3

import os
import sys

import h5py
import numpy as np
import argparse

import matplotlib.pyplot as plt

from overlay_nn import Sample, chunk_generator, valid_idx

from train import DataScaler, floatify

nn_filedir="/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/ml_inputs_aug22/scores/"
hh_nn_file="{}/CENTRAL_123456_scores.h5".format(nn_filedir)
tt_nn_file="{}/CENTRAL_410009_scores.h5".format(nn_filedir)
wt_nn_file="{}/wt_bkg_scores.h5".format(nn_filedir)
z_nn_file="{}/sherpa_zjets_scores.h5".format(nn_filedir)

cut_filedir = "/Users/dantrim/workarea/physics_analysis/wwbb/ml_training/samples/"
hh_cut_file = "/Users/dantrim/workarea/physics_analysis/wwbb/ml_training/samples/CENTRAL_123456.h5"
tt_cut_file = "{}/CENTRAL_410009.h5".format(cut_filedir)
wt_cut_file = "{}/wt_bkg.h5".format(cut_filedir)
zll_cut_file = "{}/sherpa_zll.h5".format(cut_filedir)
ztt_cut_file = "{}/sherpa_ztt.h5".format(cut_filedir)

filedir = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/samples/"
hh_file = "{}/CENTRAL_123456_sep3.h5".format(filedir)
tt_file = "{}/CENTRAL_410009_sep3.h5".format(filedir)
wt_file = "{}/wt_bkg_sep3.h5".format(filedir)
zll_file = "{}/sherpa_zll_sep3.h5".format(filedir)
ztt_file = "{}/sherpa_ztt_sep3.h5".format(filedir)

def get_nn_data(sample) :

    histo_data = []
    weight_data = []
    weight2_data = []

    with h5py.File(sample.filename, 'r', libver = 'latest') as sample_file :
        for dataset_name in sample_file :
            dataset = sample_file[dataset_name]
            for chunk in chunk_generator(dataset) :
                if chunk.dtype.names[0] != 'eventweight' :
                    print('ERROR chunk first field is not the event weight for NN data')
                    sys.exit()

                data = chunk[ 'nn_score_0' ]
                weights = chunk[ 'eventweight']
                weights *= 36.1

                valid = valid_idx(data)
                data = data[valid]
                weights = weights[valid]

                histo_data.extend(data)
                weight_data.extend(weights)
                weight2_data.extend(weights**2)

    return histo_data, weight_data, weight2_data

def get_cut_data(sample) :

    histo_data = []
    weight_data = []
    weight2_data = []

    with h5py.File(sample.filename, 'r', libver = 'latest') as sample_file :

        if 'superNt' not in sample_file :
            print('ERROR superNt dataset not found in input cut data')
            sys.exit()

        dataset = sample_file['superNt']
        for chunk in chunk_generator(dataset) :
            sel_idx = (chunk['mbb']>100) & (chunk['mbb']<140) & (chunk['mt2_llbb']>100) & (chunk['mt2_llbb']<140) & (chunk['HT2Ratio']>0.8) & (chunk['dRll']<0.9) & (chunk['nBJets']==2) & (chunk['l1_pt']>20.) & (chunk['mll']>20.)# & (chunk['mt2_bb']>150.)
            chunk = chunk[sel_idx]
            weights = chunk['eventweight'] 
            weights *= 36.1

            if 'ttbar' in sample.name :
                weights *= 0.92
            elif 'wt' in sample.name :
                weights *= 1.1037

            data = chunk['mt2_bb']
            histo_data.extend(data)
            weight_data.extend(weights)
            weight2_data.extend(weights**2)

    return histo_data, weight_data, weight2_data

def get_histogram(samples = []) :

    histo_data = []
    weight_data = []
    weight2_data = []
    bins = []

    for sample in samples :
        print('get_histogram > {}'.format(sample.name))
        if '_nn' in sample.name :
            bw = 0.01
            bins = np.arange(0,1+bw,bw)

            data, weights, weights_squared = get_nn_data(sample)
            histo_data.extend( data )
            weight_data.extend( weights )
            weight2_data.extend( weights_squared )
        elif '_cut' in sample.name :
            bw = 1
            bins = np.arange(0,500+bw,bw)

            data, weights, weights_squared = get_cut_data(sample)
            histo_data.extend( data )
            weight_data.extend( weights )
            weight2_data.extend( weights_squared )

    hist, x_edges = np.histogram( histo_data, bins = bins, weights = weight_data )
    sumw2_hist, _ = np.histogram( histo_data, bins = bins, weights = weight2_data )

#    if len(hist) > 110 :
#        for ibin in range(len(hist)) :
#            sum_mt2 = np.sum(hist[ibin:])
#            cut = bins[ibin]
#            #val = hist[ibin]
#            print("JJJ MT2 > {} yield = {}".format(cut, sum_mt2))

    return hist, sumw2_hist

def draw_roc_curve(ax, h_sig, h_bkg, label = "", sig_sumw2 = [], bkg_sumw2 = []) :

    print('draw_roc_curve > {}'.format(label))

    #min_bkg = h_bkg > 0.15
    #h_bkg = h_bkg[min_bkg]
    #h_sig = h_sig[min_bkg]
    #sig_sumw2 = sig_sumw2[min_bkg]
    #bkg_sumw2 = bkg_sumw2[min_bkg]

    total_sig_yield = h_sig.sum()
    total_bkg_yield = h_bkg.sum()
    sig_eff = np.cumsum(h_sig[::-1])[::-1]
    bkg_eff = np.cumsum(h_bkg[::-1])[::-1]

    min_cum = bkg_eff > 1
    bkg_eff = bkg_eff[min_cum]
    sig_eff = sig_eff[min_cum]

  #  if 'cut' in label :
  #      n_cut = 151
  #      h_sig = h_sig[:n_cut]
  #      h_bkg = h_bkg[:n_cut]
  #      sig_sumw2 = sig_sumw2[:n_cut]
  #      bkg_sumw2 = bkg_sumw2[:n_cut]

  #      bkg_eff = bkg_eff[:n_cut]
  #      sig_eff = sig_eff[:n_cut]


    sig_eff = sig_eff / total_sig_yield
    bkg_eff = bkg_eff / total_bkg_yield

    bkg_rej = 1. / bkg_eff

    labels = {}
    labels['nn'] = 'Multi-output NN'
    labels['cut'] = 'Cut & Count'

    colors = {}
    colors['nn'] = 'b'
    colors['cut'] = 'r'
    ax.plot(sig_eff, bkg_rej, linewidth = 1, color = colors[label])

    if bkg_sumw2.any() :

        bkg_error = np.sqrt(bkg_sumw2)

        h_bkg_up = h_bkg + np.abs(bkg_error)
        h_bkg_dn = h_bkg - np.abs(bkg_error)

        #h_bkg_up = h_bkg + bkg_error
        total_bkg_yield_up = h_bkg_up.sum()
        bkg_eff_up = np.cumsum(h_bkg_up[::-1])[::-1]
        bkg_eff_up = bkg_eff_up[min_cum]
        bkg_eff_up = bkg_eff_up /  total_bkg_yield_up
        rej_up = 1. / bkg_eff_up

        # down - when we subtract bkg by the error, the bkg rejection will go UP, so these are the UP errors
        total_bkg_yield_dn = h_bkg_dn.sum()
        bkg_eff_dn = np.cumsum(h_bkg_dn[::-1])[::-1]
        bkg_eff_dn = bkg_eff_dn[min_cum]
        bkg_eff_dn = bkg_eff_dn / total_bkg_yield_dn
        rej_dn = 1. / bkg_eff_dn

        ### UP error bars
        rel_error_up = np.abs(np.abs(rej_dn - bkg_rej) / bkg_rej)
        rel_error_dn = np.abs(np.abs(rej_up - bkg_rej) / bkg_rej)

        for ibkg, bkg_yield in enumerate(bkg_rej) :
            if bkg_rej[ibkg] > 0 :
                print("YYY bin {0} : rej = {1} sig eff = {5}-> rel error up = {2}, upper error bar at = {3} (error {4})".format(ibkg, bkg_rej[ibkg], rel_error_up[ibkg], rel_error_dn[ibkg], bkg_error[ibkg], sig_eff[ibkg]))

        ax.fill_between(sig_eff, bkg_rej - bkg_rej * rel_error_dn , bkg_rej + bkg_rej * rel_error_up, alpha = 0.6, facecolor = colors[label], edgecolor = 'none', label = labels[label])

def make_roc_curves(signal_samples = [], bkg_samples = [], args = None) :

    signal_nn_histo, signal_nn_sumw2 = get_histogram([signal_samples[0]])
    signal_cut_histo, signal_cut_sumw2 = get_histogram([signal_samples[1]])

    bkg_nn_histo, bkg_nn_sumw2 = get_histogram(bkg_samples[0])
    bkg_cut_histo, bkg_cut_sumw2 = get_histogram(bkg_samples[1])

    fig, ax = plt.subplots(1,1)
    ax.set_xlim([-0.01,1.01])
    ax.set_yscale('log')
    ax.set_xlabel('Signal Efficiency, $\\varepsilon_{s}$', horizontalalignment = 'right', x = 1)
    ax.set_ylabel('Background Rejection, $1/\\varepsilon_{b}$', horizontalalignment = 'right', y = 1)

    draw_roc_curve(ax, h_sig = signal_nn_histo, h_bkg = bkg_nn_histo, label = "nn", sig_sumw2 = signal_nn_sumw2, bkg_sumw2 = bkg_nn_sumw2)
    draw_roc_curve(ax, h_sig = signal_cut_histo, h_bkg = bkg_cut_histo, label = "cut", sig_sumw2 = signal_cut_sumw2, bkg_sumw2 = bkg_cut_sumw2)
    ax.legend(loc = 'best', frameon = False)

    fig.savefig('roc_nn_cut.pdf', bbox_inches = 'tight', dpi = 200)

    fig, ax = plt.subplots(1,1)
    bkg_cut_error = np.sqrt(bkg_cut_sumw2)
    valid = bkg_cut_histo > 0
    bkg_cut_histo = bkg_cut_histo[valid]
    bkg_cut_error = bkg_cut_error[valid]
    rel_bkg_error = bkg_cut_error / bkg_cut_histo
    ax.plot(rel_bkg_error)

    fig.savefig("bkg_count_error.pdf", bbox_inches = 'tight', dpi = 200)

############################################
def get_data(sample, kind, scaler = None, model = None) :

    lcd = 0.0
    histo_data = []
    weight_data = []
    weight2_data = []

    use_stored_model = scaler and model
    total_read = 0.0

    with h5py.File(sample.filename, 'r', libver = 'latest') as sample_file :
        if 'superNt' not in sample_file :
            print("ERROR superNt dataset not found in input file (={})".format(sample.filename))
            sys.exit()
        dataset = sample_file['superNt']
        for chunk in chunk_generator(dataset) :
            total_read += chunk.size
            if total_read > 500000. : break
            print("TOTAL READ = {}".format(total_read))

            weights = chunk['eventweight']

            # count the total number of weighted events at the base, denominator selection
            #lcd_idx = (chunk['nBJets']>=1) & (chunk['mbb']>100) & (chunk['mbb']<140) & (chunk['mt2_llbb']>100) & (chunk['mt2_llbb']<140) & (chunk['dRll']<0.9)
            lcd_idx = (chunk['nBJets']>=2) #& (chunk['mt2_llbb']<140) #& (chunk['mbb']<140)# & (chunk['dRll']<0.9)
            #lcd_idx = chunk['nBJets'] >= 1
            weights_lcd = weights[lcd_idx] * 36.1
            lcd += np.sum(weights_lcd)

            # now get the disciminants we want to scan over
            if kind == 'nn' :
                chunk = chunk[lcd_idx]
                weights = weights[lcd_idx] * 36.1

                if use_stored_model :
                    input_features = chunk[scaler.feature_list()]
                    input_features = floatify(input_features, scaler.feature_list())
                    input_features = (input_features - scaler.mean()) / scaler.scale()
                    scores = model.predict(input_features)

                    num_data = scores[:,0] # get the HH score from NN output
                    den_data = scores[:,1]
                    den_data += scores[:,2]
                    den_data += scores[:,3]# * 0.1
                    #den_data += scores[:,0]

                    ok_idx = den_data != 0
                    num_data = num_data[ok_idx]
                    den_data = den_data[ok_idx]

                    #d_tt_num = scores[:,1]
                    #d_tt_den = scores[:,0]
                    #d_tt_den += scores[:,2]
                    #d_tt_den += scores[:,3]
                    #ok_idx = d_tt_den != 0

                    #num_data = num_data[ok_idx]
                    #den_data = den_data[ok_idx]
                    #d_tt_num = d_tt_num[ok_idx]
                    #d_tt_den = d_tt_den[ok_idx]

                    #d_tt = np.log(d_tt_num / d_tt_den)
#                    print("mean d_tt = {}".format(np.mean(d_tt)))
                    #ok_idx = d_tt < -2
                    
                    #num_data = num_data[ok_idx]
                    #den_data = den_data[ok_idx]

                    weights = weights[ok_idx]
                    data = np.log(num_data / den_data)
                    #print("MIN MAX = {} {}".format(np.min(data), np.max(data)))

                else :
                    data = chunk['nn_p_hh'] # target HH score from NN

                histo_data.extend(data)
                weight_data.extend(weights)
                weight2_data.extend(weights**2)

            elif kind == 'cut' :
                sel_idx = (chunk['mbb']>100) & (chunk['mbb']<140) & (chunk['mt2_llbb']>100) & (chunk['mt2_llbb']<140) & (chunk['dRll']<0.9) & (chunk['HT2Ratio']>0.8) & (chunk['nBJets']>=2) & (chunk['l1_pt']>20.) & (chunk['mll']>20.)
                data = chunk[sel_idx]
                weights = weights[sel_idx] * 36.1
                data = data['mt2_bb'] # we are going to scan over mt2_bb in the cut based strategy

                histo_data.extend(data)
                weight_data.extend(weights)
                weight2_data.extend(weights**2)

    return lcd, histo_data, weight_data, weight2_data

def load_stored_model(nn_dir) :

    import glob
    from keras.models import model_from_json

    if not os.path.isdir(nn_dir) :
        print("ERROR load_stored_model Provided NN dir (={}) was not found".format(nn_dir))
        sys.exit()

    scaling_dir = "{}/scaling/".format(nn_dir)
    if not os.path.isdir(scaling_dir) :
        print("ERROR could not find scaling directory dataset (looking for {})".format(scaling_dir))
        sys.exit()

    training_dir = "{}/training/".format(nn_dir)
    if not os.path.isdir(training_dir) :
        print("ERROR could not find training directory dataset (looking for {})".format(training_dir)) 
        sys.exit()

    scaling_dataset = glob.glob("{}/*.h5".format(scaling_dir))
    if len(scaling_dataset) != 1 :
        print("ERROR did not find a scaling dataset in the scaling dir (or found more than one)")
        sys.exit()
    arch_file = glob.glob("{}/*arch*.json".format(training_dir))
    if len(arch_file) != 1 :
        print("ERROR did not find an architecture file (or found more than one)")
        sys.exit()
    weights_file = glob.glob("{}/*weight*.h5".format(training_dir))
    if len(weights_file) != 1 :
        print("ERROR did not find a weights file (or found more than one)")
        sys.exit()

    scaling_dataset_filename = scaling_dataset[0]
    arch_filename = arch_file[0]
    weights_filename = weights_file[0]

    # load the datascaler
    data_scaler = None
    with h5py.File(scaling_dataset_filename, 'r', libver = 'latest') as input_file :
        if 'scaling' in input_file :
            scaling_group = input_file['scaling']
            scaling_dataset = scaling_group['scaling_data']
            data_scaler = DataScaler( scaling_dataset = scaling_dataset, ignore_features = ['eventweight'] )
            print("Loaded DataScaler found {} features as inputs".format(len(data_scaler.feature_list())))
        else :
            print("ERROR scaling dataset (={}) does not have a scaling top level node".format(scaling_dataset_name))
            sys.exit()

    # load the model
    with open(arch_filename, 'r') as model_file :
        model = model_from_json(model_file.read())
    model.load_weights(weights_filename)

    return data_scaler, model

def get_histogram(samples, kind = '', nn_dir = '') :

    if kind == '' :
        print("ERROR get_histogram received no kind")
        sys.exit()

    total_lcd = 0.0
    histo_data = []
    weight_data = []
    weight2_data = []
    bins = []

    data_scaler = None
    loaded_model = None
    if nn_dir != '' :
        data_scaler, loaded_model = load_stored_model(nn_dir)

    for sample in samples :
        print('get_histogram > {}'.format(sample.name))

        if kind.lower() == 'nn' :
            bw = 0.5
            bins = np.arange(-50, 50+bw, bw)
            lcd, data, weights, weights_squared = get_data(sample, kind, scaler = data_scaler, model = loaded_model)

            total_lcd += lcd
            histo_data.extend( data )
            weight_data.extend( weights )
            weight2_data.extend( weights_squared )

        elif kind.lower() == 'cut' :
            bw = 1
            bins = np.arange(0, 500+bw, bw)
            lcd, data, weights, weights_squared = get_data(sample, kind)

            total_lcd += lcd
            histo_data.extend( data )
            weight_data.extend( weights )
            weight2_data.extend( weights_squared )

    hist, _ = np.histogram( histo_data, bins = bins, weights = weight_data )
    sumw2_hist, _ = np.histogram( histo_data, bins = bins, weights = weight2_data )
    return total_lcd, hist, sumw2_hist

def draw_roc_curve(ax, lcds, sig_histos, bkg_histos, label) :

    total_sig_yield, total_bkg_yield = lcds
    h_sig, sig_sumw2 = sig_histos
    h_bkg, bkg_sumw2 = bkg_histos

    sig_eff = np.cumsum(h_sig[::-1])[::-1]
    bkg_eff = np.cumsum(h_bkg[::-1])[::-1]

    # require selections with at least 1 weighted bkg event
    min_cum_idx = bkg_eff > 1.5
    sig_eff = sig_eff[min_cum_idx]
    bkg_eff = bkg_eff[min_cum_idx]

    sig_eff = sig_eff / total_sig_yield
    bkg_eff = bkg_eff / total_bkg_yield
    bkg_rej = 1.0 / bkg_eff

    labels = {}
    labels['nn'] = 'Multi-output NN'
    labels['cut'] = 'Cut & Count'

    colors = {}
    colors['nn'] = 'b'
    colors['cut'] = 'r'

    # plot the nominal ROC curve
    ax.plot(sig_eff, bkg_rej, linewidth = 1, color = colors[label])

    # calculate the errors on the bkg rejection (stat error)
    if bkg_sumw2.any() :

        bkg_error = np.sqrt(bkg_sumw2)

        # re-build the bkg discriminants with varied bin yields (+/- in stat error)
        h_bkg_up = h_bkg + np.abs(bkg_error)
        h_bkg_dn = h_bkg - np.abs(bkg_error)

        delta_up = np.abs(h_bkg_up.sum() - total_bkg_yield)
        delta_dn = np.abs(h_bkg_dn.sum() - total_bkg_yield)

        total_bkg_yield_up = total_bkg_yield + delta_up
        total_bkg_yield_dn = total_bkg_yield - delta_dn

        # re-build the rejection values for the +/- stat error bkg histograms

        # with YIELDS UP (REJECTION DOWN)
        bkg_eff_up = np.cumsum(h_bkg_up[::-1])[::-1]
        bkg_eff_up = bkg_eff_up[min_cum_idx]
        bkg_eff_up = bkg_eff_up / total_bkg_yield_up
        rej_up = 1.0 / bkg_eff_up

        # with YIELDS DOWN (REJECTION UP)
        bkg_eff_dn = np.cumsum(h_bkg_dn[::-1])[::-1]
        bkg_eff_dn = bkg_eff_dn[min_cum_idx]
        bkg_eff_dn = bkg_eff_dn / total_bkg_yield_dn
        rej_dn = 1.0 / bkg_eff_dn

        # LOWER ERROR BARS
        rel_lower_error = np.abs( (rej_up - bkg_rej) / bkg_rej )

        # UPPER ERROR BARS
        rel_upper_error = np.abs( (rej_dn - bkg_rej) / bkg_rej )

        # now plot the error band
        ax.fill_between(sig_eff, bkg_rej - bkg_rej * rel_lower_error, bkg_rej + bkg_rej * rel_upper_error, alpha = 0.6, facecolor = colors[label], edgecolor = 'none', label = labels[label])


def make_roc_curves_lcd(signal_sample, bkg_samples, args) :

    signal_nn_lcd, signal_nn_histo, signal_nn_sumw2_histo = get_histogram(signal_sample, kind = 'nn', nn_dir = args.nn_dir)
    bkg_nn_lcd, bkg_nn_histo, bkg_nn_sumw2_histo = get_histogram(bkg_samples, kind = 'nn', nn_dir = args.nn_dir)

    signal_cut_lcd, signal_cut_histo, signal_cut_sumw2_histo = get_histogram(signal_sample, kind = 'cut')
    bkg_cut_lcd, bkg_cut_histo, bkg_cut_sumw2_histo = get_histogram(bkg_samples, kind = 'cut')

    if len(set([signal_nn_lcd, signal_cut_lcd])) != 1 :
        print('ERROR signal LCD are not equal between cut and NN')
        sys.exit()
    if len(set([bkg_nn_lcd, bkg_cut_lcd])) != 1 :
        print('ERROR bkg LCD are not equal between cut and NN')
        sys.exit()


    fig, ax = plt.subplots(1,1)
    ax.set_xlim([-0.01, 1.01])
    ax.set_yscale('log')
    ax.set_xlabel('Signal Efficiency, $\\varepsilon_{s}$', horizontalalignment = 'right', x = 1)
    ax.set_ylabel('Background Rejection, $\\varepsilon_{b}$', horizontalalignment = 'right', y = 1)

    lcds = [signal_nn_lcd, bkg_nn_lcd]
    sig_nn_histos = [signal_nn_histo, signal_nn_sumw2_histo]
    bkg_nn_histos = [bkg_nn_histo, bkg_nn_sumw2_histo]
    draw_roc_curve(ax, lcds = lcds, sig_histos = sig_nn_histos, bkg_histos = bkg_nn_histos, label = 'nn')

    lcds = [signal_cut_lcd, bkg_cut_lcd]
    sig_cut_histos = [signal_cut_histo, signal_cut_sumw2_histo]
    bkg_cut_histos = [bkg_cut_histo, bkg_cut_sumw2_histo]
    draw_roc_curve(ax, lcds = lcds, sig_histos = sig_cut_histos, bkg_histos = bkg_cut_histos, label = 'cut')

    ax.legend(loc = 'best', frameon = False)
    fig.savefig('roc_nn_cut_new.pdf', bbox_inches = 'tight', dpi = 200)
    
def main() :

    #hh_nn_sample = Sample("hh_nn", hh_nn_file, "")
    #tt_nn_sample = Sample("ttbar_nn", tt_nn_file, "")
    #wt_nn_sample = Sample("wt_nn", wt_nn_file, "")
    #z_nn_sample = Sample("zjets_nn", z_nn_file, "")
    #bkg_nn_samples = [tt_nn_sample, wt_nn_sample, z_nn_sample]

    #hh_cut_sample = Sample("hh_cut", hh_cut_file, "")
    #tt_cut_sample = Sample("ttbar_cut", tt_cut_file, "")
    #wt_cut_sample = Sample("wt_cut", wt_cut_file, "")
    #zll_cut_sample = Sample("zjets_ll_cut", zll_cut_file, "")
    #ztt_cut_sample = Sample("zjets_tt_cut", ztt_cut_file, "")
    #bkg_cut_samples = [tt_cut_sample, wt_cut_sample, zll_cut_sample, ztt_cut_sample]

    hh_sample = Sample("hh", hh_file, "")
    tt_sample = Sample("ttbar", tt_file, "")
    wt_sample = Sample("wt", wt_file, "")
    zll_sample = Sample("zjets_ll", zll_file, "")
    ztt_sample = Sample("zjets_tt", ztt_file, "")

    # this will be the group of samples we pass around
    signal_samples = [hh_sample]
    bkg_samples = [tt_sample, wt_sample, zll_sample, ztt_sample]

    bkg_names = ["_".join(s.name.split("_")[:-1]) for s in bkg_samples]
    bkg_names = list(set(bkg_names))

    parser = argparse.ArgumentParser(description = "Plot ROC curves with NN and cut-based stuff all in one")
    parser.add_argument("--bkg", default = "all", help = "Select a background")
    parser.add_argument("--nn-dir", default = "", help = "Provide a neural network dir and load this instead")
    args = parser.parse_args()

    if args.bkg != "all" and args.bkg not in bkg_names :
        print("ERROR requested bkg (={}) not in background list (={})".format(args.bkg, bkg_names))
        sys.exit()

    if args.bkg != "all" :
        tmp = []
        for bkg in bkg_samples :
            if bkg.name == args.bkg :
                tmp.append(bkg)
        bkg_samples = tmp
        bkg_names = [args.bkg]

    #make_roc_curves(signal_samples, bkg_samples, args)
    make_roc_curves_lcd(signal_samples, bkg_samples, args)

    

if __name__ == '__main__' :
    main()

