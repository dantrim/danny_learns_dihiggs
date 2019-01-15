#!/usr/bin/env python3

import os
import sys

import h5py
import numpy as np
import argparse

import significance
from scipy.special import betainc
from scipy.stats import norm

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

filedir="/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/ml_inputs_aug22/scores/"
hh_file="{}/CENTRAL_123456_scores.h5".format(filedir)
tt_file="{}/CENTRAL_410009_scores.h5".format(filedir)
wt_file="{}/wt_bkg_scores.h5".format(filedir)
z_file="{}/sherpa_zjets_scores.h5".format(filedir)

class Sample :
    def __init__(self, name = "", filename = "", color = "") :
        self.name = name
        self.filename = filename
        self.color = color

def chunk_generator(input_h5_dataset, chunksize = 100000) :
    for x in range(0, input_h5_dataset.size, chunksize) :
        yield input_h5_dataset[x:x+chunksize]

def valid_idx(input_array) :
    valid_lo = input_array > -np.inf
    valid_hi = input_array < np.inf
    return valid_lo & valid_hi

def make_plots(samples, args) :

    edges = {}
    edges['nn_disc_0'] = np.arange(-10, 14, 0.1)
    edges['nn_disc_1'] = np.arange(-30, 1, 0.1)
    edges['nn_disc_2'] = np.arange(-14,0,0.1)
    edges['nn_disc_3'] = np.arange(-40,3,0.1)
    edges['nn_score_0'] = np.arange(0,1,0.005)
    edges['nn_score_1'] = np.arange(0,1,0.005)
    edges['nn_score_2'] = np.arange(0,1,0.005)
    edges['nn_score_3'] = np.arange(0,1,0.005)

    edges['nn_disc_0'] = np.arange(-10, 14, 0.1)
    edges['nn_disc_1'] = np.arange(-30, 1, 0.1)
    edges['nn_disc_2'] = np.arange(-14,0,0.1)
    edges['nn_disc_3'] = np.arange(-40,3,0.1)
    edges['nn_score_0'] = np.arange(0,1,0.005)
    edges['nn_score_1'] = np.arange(0,1,0.005)
    edges['nn_score_2'] = np.arange(0,1,0.005)
    edges['nn_score_3'] = np.arange(0,1,0.005)

    names = {}
    names['nn_disc_0'] = '$d_{hh}$'
    names['nn_disc_1'] = '$d_{tt}$'
    names['nn_disc_2'] = '$d_{Wt}$'
    names['nn_disc_3'] = '$d_{Z+jets}$'
    names['nn_score_0'] = '$p_{hh}$'
    names['nn_score_1'] = '$p_{tt}$'
    names['nn_score_2'] = '$p_{Wt}$'
    names['nn_score_3'] = '$p_{Z+jets}$'

    x_data = {}
    y_data = {}
    w_data = {}

    for s in samples :
        x_data[s.name] = []
        y_data[s.name] = []
        w_data[s.name] = []

    score_dset_match = "nn_scores"

    for sample in samples :
        print(" > {}".format(sample.name))
        with h5py.File(sample.filename, 'r', libver = 'latest') as sample_file :
            for dataset_name in sample_file :
                dataset = sample_file[dataset_name]
                for chunk in chunk_generator(dataset) :
                    if chunk.dtype.names[0] != 'eventweight' :
                        print('ERROR chunk first field is not the eventweight')
                        sys.exit()

                    valid_x = valid_idx( chunk[args.varX] )
                    valid_y = valid_idx( chunk[args.varY] )
                    valid = valid_x & valid_y
                    chunk = chunk[valid]
                    weights = chunk['eventweight']
                    nn_score_0 = chunk['nn_score_0']
                    nn_score_1 = chunk['nn_score_1']
                    nn_score_2 = chunk['nn_score_2']

                    varx = nn_score_0[:] / nn_score_1[:]
                    vary = nn_score_0[:] / nn_score_2[:]

                    valid_x = valid_idx(varx)
                    valid_y = valid_idx(vary)
                    valid = valid_x & valid_y
                    varx = varx[valid]
                    vary = vary[valid]
                    weights = weights[valid]

                    x_data[sample.name].extend(varx)
                    y_data[sample.name].extend(vary)
                    w_data[sample.name].extend(weights)

    ##### 1D PLOT
    fig, ax = plt.subplots(1,1)
    xlims = [0, 7]
    ylims = [0, 1.3]
    ax.set_xlim(xlims)
    #ax.set_ylim(ylims)
    bw = 0.1
    bins = np.arange(0,5+bw,bw)
    bin_centers = (bins[0:-1] + bins[1:])/2
    normed = False
    h_tt_sig, _ = np.histogram(x_data[samples[0].name], bins = bins, normed = False)
    total_sig_yield = h_tt_sig.sum()
    h_tt_sig = h_tt_sig / total_sig_yield
    h_tt_sig = np.cumsum(h_tt_sig[::-1])[::-1]

    h_tt_tot, _ = np.histogram(x_data[samples[1].name], bins = bins, normed = False)
    for sample in samples[2:] :
        h_tt_tmp, _ = np.histogram(x_data[sample.name], bins = bins, normed = False)
        h_tt_tot += h_tt_tmp

    total_bkg_yield = h_tt_tot.sum()
    h_tt_tot = h_tt_tot / total_bkg_yield
    h_tt_tot = np.cumsum(h_tt_tot[::-1])[::-1]

    h_tt_ratio = h_tt_sig / h_tt_tot
    print("h_tt_ratio = {}".format(h_tt_ratio))
    ax.plot(bin_centers, h_tt_ratio, 'o', markersize = 4, label = '$p_{hh} / p_{Wt}$')
    #ax.plot(bin_centers, h_tt_sig, 'o', markersize = 4, label = '$p_{hh} / p_{Wt}$ sig')
    #ax.plot(bin_centers, h_tt_tot, 'o', markersize = 4, label = '$p_{hh} / p_{Wt}$ bkg')

    h_wt_sig, _ = np.histogram(y_data[samples[0].name], bins = bins, normed = False)
    total_sig_yield = h_wt_sig.sum()
    h_wt_sig = h_wt_sig / total_sig_yield
    h_wt_sig = np.cumsum(h_wt_sig[::-1])[::-1]

    h_wt_tot, _ = np.histogram(y_data[samples[1].name], bins = bins, normed = False)
    for sample in samples[2:] :
        h_wt_tmp, _ = np.histogram(y_data[sample.name], bins = bins, normed = False)
        h_wt_tot += h_wt_tmp
    total_bkg_yield = h_wt_tot.sum()
    h_wt_tot = h_wt_tot / total_bkg_yield
    h_wt_tot = np.cumsum(h_wt_tot[::-1])[::-1]

    h_wt_ratio = h_wt_sig / h_wt_tot
    ax.plot(bin_centers, h_wt_ratio, 'o', markersize = 4, label = '$p_{hh} / p_{Wt}$')

    # horizontal line
    ax.plot([min(xlims), max(xlims)], [1.0, 1.0], 'k--', lw = 1)

    # lables
    ax.set_ylabel("$hh$ Purity", horizontalalignment = 'right', y = 1)

    ax.legend(loc='best', frameon = False, numpoints = 1)

    fig.savefig("test_1d.pdf", bbox_inches = "tight", dpi = 200)


    ##### 2D PLOT
    fig, ax = plt.subplots(1,1)
    x_edges = np.arange(0,6, 0.08)
    y_edges = np.arange(0,6, 0.08)
    #x_edges = edges[args.varX]
    #y_edges = edges[args.varY]
    bins = [x_edges, y_edges]

    ax.set_xlabel(names[args.varX], horizontalalignment = 'right', x = 1)
    ax.set_ylabel(names[args.varY], horizontalalignment = 'right', y = 1)
    h_sig, x, y = np.histogram2d(x_data[samples[0].name], y_data[samples[0].name], bins = bins, normed = False)
    total_sig_yield = h_sig.sum()
    h_sig = h_sig / total_sig_yield

    h_total, x, y = np.histogram2d(x_data[samples[1].name], y_data[samples[1].name], bins = bins, normed = False)
    for sample in samples[2:] :
        h_tmp, x, y= np.histogram2d(x_data[sample.name], y_data[sample.name], bins = bins, normed = False)
        h_total += h_tmp
    total_bkg_yield = h_total.sum()
    h_total = h_total / total_bkg_yield

    h_ratio = h_sig / h_total
    h_ratio = h_ratio.T
    #ax.set_xscale('log')
    imextent = list( (min(x_edges), max(x_edges) ) ) + list( (min(y_edges), max(y_edges)))
    ax.set_facecolor('lightgrey')
    im = ax.imshow(h_ratio, origin = 'lower', cmap = 'coolwarm', aspect = 'auto', interpolation = 'nearest', extent = imextent, norm = LogNorm())
    cb = fig.colorbar(im)
    fig.savefig("test_2d.pdf", bbox_inches = 'tight', dpi = 200)

def zn(hist_s, hist_b, rel_b_unc = 0.3) :

    total = hist_s + hist_b
    tau = 1.0 / hist_b / (rel_b_unc**2)
    aux = hist_b * tau

    return -1.0 * norm.ppf( betainc( total, aux + 1, 1.0 / (1.0 + tau)) )
    

def make_2d_eff(samples, args) :

    x_data = {}
    y_data = {}
    w_data = {}

    for s in samples :
        x_data[s.name] = []
        y_data[s.name] = []
        w_data[s.name] = []

    dset_name = "nn_scores"
    for sample in samples :
        with h5py.File(sample.filename, 'r', libver = 'latest') as sample_file :
            for dataset_name in sample_file :
                dataset = sample_file[dataset_name]
                for chunk in chunk_generator(dataset) :
                    if chunk.dtype.names[0] != 'eventweight' :
                        print('ERROR first field is not the eventweight')
                        sys.exit()

                    hh_score = chunk['nn_score_0']
                    tt_score = chunk['nn_score_1']
                    wt_score = chunk['nn_score_2']
                    z_score = chunk['nn_score_3']
                    weights = chunk['eventweight']

                    varx = hh_score[:] / tt_score[:]
                    vary = hh_score[:] / wt_score[:]

                    valid_x = valid_idx(varx)
                    valid_y = valid_idx(vary)
                    valid = valid_x & valid_y

                    varx = varx[valid]
                    vary = vary[valid]
                    weights = weights[valid]

                    x_data[sample.name].extend(varx)
                    y_data[sample.name].extend(vary)
                    w_data[sample.name].extend(weights)

    x_edges = np.arange(0,5.1,0.1)
    y_edges = np.arange(0,5.1,0.1)
 #   x_edges = np.arange(-20, 10, 1)
 #   y_edges = np.arange(-20, 2, 1)
    bins = [x_edges,y_edges]

    # first make the signal 2D distribution
    h_sig, x_edges, y_edges = np.histogram2d(x_data[samples[0].name], y_data[samples[0].name], bins = bins, weights = np.array(w_data[samples[0].name]) * 36.1)
    total_sig_yield = h_sig.sum()
    #h_sig_eff = h_sig / total_sig_yield

    # now make the total background 2D distribution
    h_bkg, x_edges, y_edges = np.histogram2d(x_data[samples[1].name], y_data[samples[1].name], bins = bins, weights = np.array(w_data[samples[1].name]) * 36.1)
    for sample in samples[2:] :
        h, _, _ = np.histogram2d(x_data[sample.name], y_data[sample.name], bins = bins, weights = np.array(w_data[sample.name]) * 36.1)
        h_bkg += h
    total_bkg_yield = h_bkg.sum()
    #h_bkg_eff = h_bkg / total_bkg_yield

    # take the 2D integral in reverse fashion of each
    h_sig_eff = np.cumsum(h_sig[::-1,:], axis = 0)[::-1,:]
    h_sig_eff = np.cumsum(h_sig_eff[:,::-1], axis = 1)[:,::-1]
    h_sig_eff = h_sig_eff / total_sig_yield

    h_bkg_eff = np.cumsum(h_bkg[::-1,:], axis = 0)[::-1,:]
    h_bkg_eff = np.cumsum(h_bkg_eff[:,::-1], axis = 1)[:,::-1]
    h_bkg_eff = h_bkg_eff / total_bkg_yield

    # plot time
    # take Z to be s / sqrt(s+b)
#    h_sob = h_sig / np.sqrt(h_sig + h_bkg)
#    h_sob = h_sig_eff / np.sqrt(h_sig_eff + h_bkg_eff)
    h_sob = h_sig_eff / np.sqrt( h_bkg_eff )
#    h_sob = h_sig_eff / h_bkg_eff
#    h_sob = h_sig_eff / np.sqrt(h_bkg_eff**2 + 0.3**2)
#    s = h_sig
#    b = h_bkg
#    h_sob = np.sqrt( 2 * ( (s + b) * np.log(1 + s/b) - s) )

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('$p_{hh} / p_{tt}$', horizontalalignment = 'right', x = 1)
    ax.set_ylabel('$p_{hh} / p_{Wt}$', horizontalalignment = 'right', y = 1)
    imextent = list( (min(x_edges),max(x_edges)) ) + list( (min(y_edges),max(y_edges)) )
    im = ax.imshow(h_sob.T, origin = 'lower', cmap = 'coolwarm', aspect = 'auto', interpolation = 'nearest', extent = imextent)#, vmax = 3)#, norm = LogNorm())
    cb = fig.colorbar(im)
    cb.set_label('$\\varepsilon_{s}/\\sqrt{\\varepsilon_{b}}$', horizontalalignment = 'right', y = 1)
    #cb.set_label('$\\varepsilon_{s}/\\sqrt{\\varepsilon_{s} + \\varepsilon_{b}}$', horizontalalignment = 'right', y = 1)

    # add some contours
#    levels = np.arange(0.2, 1.0, 0.1)
#    levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#    levels = [0.5, 1.0, 1.25, 1.5]
    levels = [0.5, 1.0, 1.25, 1.5, 1.75, 2.0]
#    contours = ax.contour(h_sob.T, levels, extent = imextent, colors = 'k')
#    ax.clabel( contours, levels, inline = 1, fmt = '%1.2f', fontsize=12)

    # get the background contours
 #   fig_bkg, ax_bkg = plt.subplots(1,1)
 #   levels = [0.1, 0.5]
 #   im = ax_bkg.imshow(h_sob.T, origin = 'lower', cmap = 'coolwarm', aspect = 'auto', interpolation = 'nearest', extent = imextent)
 #   bkg_contours =  ax.contour(h_bkg_eff.T, levels, extent = imextent, colors = 'k', linestyles='--', linewidths = 1)
 #   ax.clabel( bkg_contours, [levels[0]], inline = 1, fmt = '%1.2f', fontsize = 10)


    # contour labels
    fig.savefig('test_sob.pdf', bbox_inches = 'tight', dpi = 200)

def make_1d_dist(samples, args) :

    hist_data = {}
    w_data = {}

    for s in samples :
        hist_data[s.name] = []
        w_data[s.name] = []

    dset_name = "nn_scores"
    for sample in samples :
        with h5py.File(sample.filename, 'r', libver = 'latest') as sample_file :
            for dataset_name in sample_file :
                dataset = sample_file[dataset_name]
                for chunk in chunk_generator(dataset) :
                    hh_score = chunk['nn_score_0']
                    tt_score = chunk['nn_score_1']
                    wt_score = chunk['nn_score_2']
                    z_score = chunk['nn_score_3']
                    weights = chunk['eventweight']
                    
                    varx = hh_score /  (wt_score)
#                    varx = chunk['nn_disc_0']
                    #varx = np.log(hh_score /  (z_score + tt_score + wt_score))
                    valid_x = valid_idx(varx)

                    varx = varx[valid_x]
                    weights = weights[valid_x]

                    hist_data[sample.name] = varx
                    w_data[sample.name] = weights

    x_edges = np.arange(0,60,1)

    histos = []
    weights = []
    names = []
    for sample in samples :
        histos.append( hist_data[sample.name] )
        weights.append( w_data[sample.name] )
        names.append(sample.name)

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("p_hh / p_X")
    ax.hist(histos, weights = weights, bins = x_edges, label = names, density = True, histtype = 'step')
    ax.set_yscale('log')
    ax.legend(loc='best', frameon = False)

    fig.savefig("test_varx.pdf", bbox_inches = 'tight', dpi = 200)

def main() :

    hh_sample = Sample("hh", hh_file, "cool")
    tt_sample = Sample("ttbar", tt_file, "Greys")
    wt_sample = Sample("wt", wt_file, "Greys")
    z_sample = Sample("zjets", z_file, "Greys")
    samples = [hh_sample, tt_sample, wt_sample, z_sample]

    parser = argparse.ArgumentParser(description = "Plot Stuff in 2D")
    parser.add_argument("-x", "--varX", help = "Provide variable to put on X-axis", default = "")
    parser.add_argument("-y", "--varY", help = "Provide variable to put on Y-axis", default = "")
    parser.add_argument("--eff", help = "Make 2D eff plot", action = 'store_true', default = False)
    args = parser.parse_args()

    for s in samples :
        if not os.path.isfile(s.filename) :
            print("ERROR File for sample {} (file={}) not found".format(s.name, s.filename))
            sys.exit()
    if not args.eff :
        make_plots(samples, args)
    else :
        make_1d_dist(samples,args)
        make_2d_eff(samples,args)

if __name__ == '__main__' :
    main()
