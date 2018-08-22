#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import argparse

# h5py
import h5py

# numpy
import numpy as np

# stats
import significance


filedir = "/Users/dantrim/workarea/physics_analysis/wwbb/ml_training/samples/"
bkg_filenames = ["sherpa_zll.h5", "sherpa_ztt.h5", "wt_bkg.h5", "CENTRAL_410009.h5"]
bkg_files = [ "{}/{}".format(filedir, fname) for fname in bkg_filenames ]

reco_sig = "/Users/dantrim/workarea/physics_analysis/wwbb/ml_training/samples/CENTRAL_342053.h5"
truth_sig = "/Users/dantrim/workarea/physics_analysis/wwbb/ml_training/samples/wwbb_truth_342053_aug6.h5"
#truth_sig = "/Users/dantrim/workarea/physics_analysis/wwbb/ml_training/samples/wwbb_truth_123456_aug6_custom.h5"

lumi_factor = 36.1

def chunks(input_file, chunksize = 100000, dataset_name = "superNt") :

    with h5py.File(input_file, 'r', libver= 'latest') as f :
        dataset = f[dataset_name]
        for x in range(0, dataset.size, chunksize) :
            yield dataset[x:x+chunksize]

def get_yields(filelist, dataset_name = 'superNt') :


    total_yields = 0.0
    total_w2 = 0.0

    is_bkg = False

    for input_file in filelist :

        yield_for_sample = 0.0
        w2 = 0.0

        for chunk in chunks(input_file = input_file, dataset_name = dataset_name) :

            # apply selection
            sel_idx = (chunk['mbb']>100) & (chunk['mbb']<140) & (chunk['mt2_llbb']>100) & (chunk['mt2_llbb']<140) & (chunk['HT2Ratio']>0.8) & (chunk['dRll']<0.9) & (chunk['mt2_bb']>150) & (chunk['nBJets']==2) & (chunk['l1_pt']>20.) & (chunk['mll']>20.)
            data = chunk[sel_idx]
            weights = data['eventweight']
            #data = chunk
            #weights = data['eventweight']

            # apply scalefactors based on previous R20.7 HistFitter background-only fit
            scalefactor = 1.0
            if "410009" in input_file :
                is_bkg = True
                scalefactor = 0.92
            elif "wt" in input_file :
                scalefactor = 1.1037
            yield_for_sample += np.sum(weights) * 36.1 * scalefactor
            w2 += np.sum(weights**2)
        total_yields += yield_for_sample
        total_w2 += w2

        print("yield for {0} = {1:.3f} +/- {2:.3f}".format(input_file.split("/")[-1], yield_for_sample, np.sqrt(w2)))

    # add dibosn
    if is_bkg :
        total_yields += 0.17
    print(" --> total yield = {0:.3f} +/- {1:.3f}".format(total_yields, np.sqrt(total_w2)))

    return total_yields

def calculate_upperlimit(bkg_counts) :

    sig = 0.5
    z = 0


    # scan until we hit the 95% confidence exlusion
    while True :
        z = significance.binomial_exp_z(sig, bkg_counts, 0.365)
        if z >= 1.64 : break
        if sig > 100 :
            sig = -1
            break
        sig += 0.01
    return sig

def main() :

    parser = argparse.ArgumentParser()
    parser.add_argument("--nbkg", help = "Provide total nbkg", default = "")
    args = parser.parse_args()

    total_bkg_yield = 0.0

    if args.nbkg != "" :
        total_bkg_yield = float(args.nbkg)
    else :
        # first find the upperlimit on allowed sig
        total_bkg_yield = get_yields(bkg_files)

    n_sig_ul = calculate_upperlimit(total_bkg_yield)
    print("UL on sig = {}".format(n_sig_ul))

    reco_sig_yields = get_yields([reco_sig])
    truth_sig_yields = get_yields([truth_sig], 'truth')

    #e_times_a = reco_sig_yields / truth_sig_yields
    acceptance = truth_sig_yields / (590.0 * 36.1)
    print("A  = {}".format(acceptance))
    efficiency = reco_sig_yields / truth_sig_yields
    print("E = {}".format(efficiency))
    e_times_a = acceptance * efficiency

    print(50 * "=")
    print("e x A = {}".format(e_times_a))

    xsec_ul = n_sig_ul / 36.1
    xsec_ul = xsec_ul / (e_times_a * 2.0 * 0.57 * 0.21)
    print("xsec UL = {}".format(xsec_ul))
    

if __name__ == "__main__" :
    main()

