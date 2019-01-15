#!/bin/env python

import sys, os
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import h5py

from train import DataScaler, floatify, Sample, build_combined_input
from roc_nn_and_cut import load_stored_model

# enforce reproducibility (this is the same seed used in training)
seed = 347
np.random.seed(seed)

training_dir = '/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/OVERTRAINCHECK/training_note/training_earlyStopping_Patience20/'

def adjust_axes(ax) :

    ax.tick_params(axis = 'both', which = 'both', direction = 'in',
        labelleft = True, bottom = True, top = True, left = True)
    ax.grid(color = 'k', which = 'both', linestyle = '-', lw = 1, alpha = 0.2, zorder = 0)
    ax.set_xlim([0,1])

def get_process_inputs(data_scaler) :

    idx = -1
    if training_dir.endswith('/') : idx = -2
    input_file_dir = '/'.join(training_dir.split('/')[:idx])
    input_file = '%s/wwbb_preprocessed.h5' % input_file_dir

    if not os.path.isdir(input_file_dir) :
        print('ERROR could not locate file dir (={})'.format(input_file_dir))
        sys.exit()
    if not os.path.isfile(input_file) :
        print('ERROR could not locate file (={})'.format(input_file))
        sys.exit()

    samples_group_name = 'samples'
    scaling_group_name = 'scaling'
    scaling_data_name = 'scaling_data'


    sample_dict = {}

    with h5py.File(input_file, 'r', libver = 'latest') as input_file :

        if samples_group_name in input_file :
            sample_group = input_file[samples_group_name]
            n_per_sample = -1
            for p in sample_group :

                if p == 'ttbar' or p == 'hh' :

                    sample_dict[p] = {}

                    process_group = sample_group[p]
                    class_label = process_group.attrs['training_label']

                    # get the "test" sample
                    test_sample = Sample(name = '%s_test' % p, class_label = int(class_label),
                            input_data = floatify( process_group['validation_features'][tuple(data_scaler.feature_list())], data_scaler.feature_list()))
                    test_sample.eventweights = floatify( process_group['validation_features'][tuple(['eventweight'])], ['eventweight'])

                    # get the "training" sample (which has our validation data in it)
                    training_data = floatify( process_group['train_features'][tuple(data_scaler.feature_list())], data_scaler.feature_list())
                    training_weights = floatify( process_group['train_features'][tuple(['eventweight'])], ['eventweight'])

                    # randomize
                    if p != 'hh' :
                        if n_per_sample < 0 :
                            print('ERROR Did not get number to split for train/validation from signal')
                            sys.exit()
                    elif p == 'hh' :
                        n_per_sample = int(training_data.shape[0])

                    randomize = np.arange(len(training_data))
                    np.random.shuffle(randomize)
                    shuffled_training_data = training_data[randomize]
                    shuffled_training_weights = training_weights[randomize]

                    fraction_for_validation = 0.2
                    total_n = len(shuffled_training_data)
                    n_for_validation = int(fraction_for_validation * total_n)

                    split_train_data = shuffled_training_data[n_for_validation:]
                    split_train_weights = shuffled_training_weights[n_for_validation:]
                    split_val_data = shuffled_training_data[:n_for_validation]
                    split_val_weights = shuffled_training_weights[:n_for_validation]

                    train_sample = Sample(name = '%s_train' % p, class_label = int(class_label), input_data = split_train_data)
                    train_sample.eventweights = split_train_weights

                    val_sample = Sample(name = '%s_val' % p, class_label = int(class_label), input_data = split_val_data)
                    val_sample.eventweights = split_val_weights

                    print('Loaded sample %s: n train = %d, n val = %d, n_test = %d' % (p, len(train_sample.data()), len(val_sample.data()), len(test_sample.data())))

                    sample_dict[p]['test'] = test_sample
                    sample_dict[p]['train'] = train_sample
                    sample_dict[p]['val'] = val_sample

    return sample_dict

def make_nn_output_plots(inputs_dict, data_scaler, model) :

    fig = plt.figure(figsize = (7,8))
    grid = GridSpec(100,100)
    upper_pad = fig.add_subplot( grid[0:60,:])
    middle_pad = fig.add_subplot( grid[64:78,:], sharex = upper_pad )
    lower_pad = fig.add_subplot( grid[82:96,:], sharex = upper_pad )


    upper_pad.set_ylabel('a.u.', horizontalalignment = 'right', y = 1)
    middle_pad.set_ylabel('X / test')
    lower_pad.set_ylabel('X / test')
    lower_pad.set_xlabel('$p_{hh}$', horizontalalignment = 'right', x = 1)
#    lower_pad.set_xlabel('$p_{t\\bar{t}}$', horizontalalignment = 'right', x = 1)

    upper_pad.set_ylim([0,18])
    middle_pad.set_ylim([0,2])
    lower_pad.set_ylim([0,2])


    colors = ['b','r']

    binning_dict = { 'p' : [0.05, 0, 1],
                        'd' : [2, -30, 15] }

    which = 'p'
    bin_bounds = binning_dict[which]
    bin_width = bin_bounds[0]
    x_low = bin_bounds[1]
    x_high = bin_bounds[2]
    bin_edges = np.arange(x_low, x_high + bin_width, bin_width)

    for pad in [upper_pad, middle_pad, lower_pad] :
        adjust_axes(pad)
        pad.set_xlim([x_low, x_high])
#    upper_pad.set_xticks([0])

    #x_ticks = np.arange(0, 1, 0.1)
    #x_ticks = np.arange(x_low, x_high, bin_width * 2) 
    #lower_pad.xaxis.set_visible(True)
    #lower_pad.set_xticks(x_ticks)

    sample_names = []
    handles = []
    for i, sample_name in enumerate(inputs_dict) :

        hist_data = []
        w2_data = []

        rel_errors = []
        alphas = [0.4, 0.7, 1.0]
        loc_samples = []
        for idata, data_type in enumerate(['train', 'val', 'test']) :

            sample = inputs_dict[sample_name][data_type]
            sample_names.append(sample.name())
            loc_samples.append(sample.name())
            input_features = sample.data()
            weights = sample.eventweights

            input_features_scaled = (input_features - data_scaler.mean()) / data_scaler.scale()
            scores = model.predict(input_features_scaled)

            p_hh = scores[:,0]
            p_tt = scores[:,1]
            p_wt = scores[:,2]
            p_z = scores[:,3]

            #d_num = p_hh
            #d_den = p_tt + p_wt + p_z
            #idx = d_den != 0
            #d_num = d_num[idx]
            #d_den = d_den[idx]
            #weights = weights[idx]
            #d_hh = np.log(d_num / d_den)

            #p_use = p_tt
            p_use = p_hh


            weights = np.ones(len(weights))

            h, _, _ = upper_pad.hist(p_use, weights = weights, bins = bin_edges, color = colors[i], label = sample.name(), density = True, histtype = 'step', lw = 1, alpha = alphas[idata])
            hist_data.append(h)


            # error bar
            weights = np.reshape(weights, (weights.shape[0],))
            w2 = weights ** 2
            w2 = np.reshape(w2, (w2.shape[0],))
            herr, _ = np.histogram(p_use, weights = weights, bins = bin_edges)
            hsumw2, _ = np.histogram(p_use, weights = w2, bins = bin_edges)
            err = np.sqrt(hsumw2)
            rel_err = np.divide(err, herr)
            rel_errors.append(rel_err)

#            w2 = weights ** 2
#            w2 = np.reshape(w2, (w2.shape[0],))
#            sqrt_sumw2 = np.sqrt(hsumw2)
#            delta = np.abs(sqrt_sumw2 - h)
#            rel_error = delta
##            rel_delta = np.divide(delta, h)
##            rel_error = rel_delta
#
#            print('h              = {}'.format(h))
#            print('sumw2            {}'.format(hsumw2))
#            print('sqrt suwm2       {}'.format(sqrt_sumw2))
#            print('delta            {}'.format(delta))
#            print('rel_error        {}'.format(rel_error))
#            rel_errors.append(rel_error)
#            #rel_errors.append(rel_delta)

            x_vals = bin_edges + 0.5 * bin_width
            x_vals = x_vals[:-1]
            x_offsets = [-0.2 * bin_width, 0.2 * bin_width, 0]
            x_vals += x_offsets[idata]
            upper_pad.errorbar(x_vals, h, yerr = rel_err,   fmt = 'none', ecolor = colors[i], alpha = alphas[idata])

        color_idx = 0
        if 'ttbar' in sample_name :
            color_idx = 1
        for isn, s in enumerate(loc_samples) :
            alpha = alphas[isn]
            color = colors[color_idx]
            handles.append(Line2D([0],[0], color = color, alpha = alpha))
        upper_pad.legend(handles, sample_names, loc = 'best')
#        upper_pad.legend(loc = 'best')

        train_idx = 0
        val_idx = 1
        test_idx = 2

        ratio_train = np.divide(hist_data[train_idx], hist_data[test_idx])
        ratio_val = np.divide(hist_data[val_idx], hist_data[test_idx])

        pad = None
        if 'hh' in sample_name :
            pad = middle_pad
        else :
            pad = lower_pad

        x_vals = bin_edges + 0.5 * bin_width
        x_vals = x_vals[:-1]
        x_offsets = [-0.2 * bin_width, 0.2 * bin_width]
        x_vals_ = x_vals + x_offsets[train_idx]
        pad.plot(x_vals_, ratio_train, linestyle = 'none', marker = 'o', color = colors[i], alpha = alphas[train_idx])
        x_vals_ = x_vals + x_offsets[val_idx]
        pad.plot(x_vals_, ratio_val, linestyle = 'none', marker = 'o', color = colors[i], alpha = alphas[val_idx])

        rel_err_den = rel_errors[test_idx]

        # train error bars
        rel_err_num = rel_errors[train_idx]
        x_vals_ = x_vals + x_offsets[train_idx]
        rel_error = np.sqrt( np.power( rel_err_num, 2) + np.power( rel_err_den, 2)) * ratio_train
#        print('rel_errors[test] =  {}'.format(rel_errors[test_idx]))
#        print('rel_errors[train] = {}'.format(rel_errors[train_idx]))
#        print('rel_error train = {}'.format(rel_error))
        pad.errorbar(x_vals_, ratio_train, yerr = rel_error, fmt = 'none', ecolor = colors[i], alpha = alphas[train_idx])

        # val error bars
        rel_err_num = rel_errors[val_idx]
        x_vals_ = x_vals + x_offsets[val_idx]
        rel_error = np.sqrt( np.power(rel_err_num, 2) + np.power( rel_err_den, 2)) * ratio_val
        pad.errorbar(x_vals_, ratio_val, yerr = rel_error, fmt = 'none', ecolor = colors[i], alpha = alphas[val_idx])
        

    fig.savefig('./plots/overtrain_check_nn_p.pdf', bbox_inches = 'tight', dpi = 200)

def main() :

    parser = argparse.ArgumentParser(
        description = 'Make plots of NN discriminants for training, validation, and test data'
    )

    data_scaler, model = load_stored_model(training_dir)
    inputs_dict = get_process_inputs(data_scaler)

    make_nn_output_plots(inputs_dict, data_scaler, model)

if __name__ == '__main__' :
    main()
