#!/usr/bin/env python3

import os
import sys

import h5py
import numpy as np
import argparse

import numpy as np

from overlay_nn import Sample, chunk_generator, valid_idx
from train import DataScaler, floatify
from roc_nn_and_cut import load_stored_model

filedir = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/samples/bdef_for_ul_calc/"
reco_sig = "{}/CENTRAL_123456.h5".format(filedir)
truth_sig = "{}/wwbb_truth_123456_sep10.h5".format(filedir)
lumi_factor = 36.1

def get_yields(args, kind = '') :

    if not kind :
        print('did not provide kind')
        sys.exit()

    filename = {'reco' : reco_sig,
                'truth' : truth_sig}[kind]
    treename = {'reco' : 'superNt',
                'truth' : 'truth'}[kind]

    sample = Sample(kind, filename, '')
    data_scaler, model = load_stored_model(args.nn_dir)

    total_counts_raw = 0
    total_counts_weighted = 0.0

    with h5py.File(sample.filename, 'r', libver = 'latest') as sample_file :
        if treename not in sample_file :
            print('ERROR treename (={}) is not found in input file (={})'.format(treename, sample.filename))
            sys.exit()
        dataset = sample_file[treename]
        for chunk in chunk_generator(dataset) :

            weights = chunk['eventweight'] * lumi_factor

            if not args.cut_based :
                # calculate OTF
                input_features = chunk[data_scaler.feature_list()]
                input_features = floatify(input_features, data_scaler.feature_list())
                input_features = (input_features - data_scaler.mean()) / data_scaler.scale()
                scores = model.predict(input_features)

                nn_p_hh = scores[:,0]
                nn_p_tt = scores[:,1]
                nn_p_wt = scores[:,2]
                nn_p_zjets = scores[:,3]

                nn_d_hh = np.log( nn_p_hh / (nn_p_tt + nn_p_wt + nn_p_zjets) )
                ok_idx = (nn_d_hh > -np.inf) & (nn_d_hh < np.inf)
                weights = weights[ok_idx]
                chunk = chunk[ok_idx]

                selection_idx = (chunk['nBJets']>=2) & (chunk['mbb']>110) & (chunk['mbb']<140) & (chunk['mt2_bb']>65)
                nn_idx = nn_d_hh > 6.2
                #selection_idx = (chunk['nBJets']>=2) & (chunk['mbb']>100) & (chunk['mbb']<140) & (chunk['mt2_bb']>65)
                #nn_idx = nn_d_hh > 6.3

                print('nn_idx = {}'.format(nn_idx.any()))
                selection_idx = nn_idx & selection_idx

                weights = weights[selection_idx]
                data = chunk[selection_idx]

                total_counts_raw += data.size
                total_counts_weighted += np.sum(weights)
            else :
                selection_idx = (chunk['mll']>20.) & (chunk['l1_pt']>20.) & (chunk['nBJets']==2) & (chunk['dRll']<0.9) & (chunk['HT2Ratio']>0.8) & (chunk['mt2_bb']>150.) & (chunk['mbb']>100) & (chunk['mbb']<140) & (chunk['mt2_llbb']>100.) & (chunk['mt2_llbb']<140.)
                weights = weights[selection_idx]
                data = chunk[selection_idx]

                total_counts_raw += data.size
                total_counts_weighted += np.sum(weights)

    print('yields for {}: {} ({})'.format(kind, total_counts_weighted, total_counts_raw))
    return total_counts_weighted, total_counts_raw
            
        

def main() :

    parser = argparse.ArgumentParser(description = "Calculate the acc x efficiency for the signal to pass NN selection")
    parser.add_argument("--nn-dir", required = True, help = "Provide the directory where the model is stored")
    parser.add_argument("--vis-xsec", default = "", help = "Provide an UL on visible cross-section calculated via other means (e.g. HistFitter)")
    parser.add_argument("-v", "--verbose", default = False, action = "store_true", help = "Speak up")
    parser.add_argument("--cut-based", default = False, action = "store_true", help = "Calculate numbers for the cut-based selection")
    args = parser.parse_args()

    if not os.path.isdir(args.nn_dir) :
        print('ERROR Provided nn-dir (={}) cannot be located'.format(args.nn_dir))
        sys.exit()

    truth_sr_counts_weighted, truth_sr_counts_raw = get_yields(args, kind = 'truth')
    reco_sr_counts_weighted, reco_sr_counts_raw = get_yields(args, kind = 'reco')

    # total truth level events at EVNT level
    n_total_truth = 21298.8

    # filter stuff
    w_lnu_efficiency = 0.3272**2 # square of the BR for the leptonic decay of W bosons
    kinematic_efficiency = 0.23849 # GenFiltEfficiency from custom WWbb signal sample with pT > 12 GeV requirement
    kinematic_efficiency = kinematic_efficiency / 0.50 # remove effect of hBB and hWW filters to just get kinematic filter eff from MultiElecMuTau filter
    den_filter_factor = w_lnu_efficiency * kinematic_efficiency

    # hh -> WWbb branching ratio
    br_hbb = 0.5824
    br_hWW = 0.2137
    branching_ratio = 2 * br_hbb * br_hWW

    # den factor minus acceptance x efficiency
    den_factor = branching_ratio * den_filter_factor

    # truth acceptance
    acceptance = truth_sr_counts_weighted / n_total_truth

    # reco efficiency
    reco_efficiency = reco_sr_counts_weighted / truth_sr_counts_weighted

    # acceptance x efficiency
    a_x_e = acceptance * reco_efficiency

    # UL denominator
    ul_den = a_x_e * den_factor

    if args.verbose :
        print(' > den_filter_factor = {}'.format(den_filter_factor))
        print(' > a     = {}'.format(acceptance))
        print(' > e     = {}'.format(reco_efficiency))
        print(' > a_x_e = {}'.format(a_x_e))

    print('UL denominator: {0}'.format(ul_den))

    if args.vis_xsec != "" :
        vis_xsec_limit = float(args.vis_xsec)

        vis_xsec_limit = vis_xsec_limit / 1000. # convert from [fb] to [pb]
        ul_limit = vis_xsec_limit / ul_den
        print('UL limit      : {0:.6f}'.format(ul_limit))

    

if __name__ == "__main__" :
    main()
