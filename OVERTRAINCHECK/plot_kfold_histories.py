#!/bin/env python

import sys, os, glob
import argparse
import matplotlib.pyplot as plt
import pickle
import numpy as np

no_do_dir = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/OVERTRAINCHECK/training_note/training_kfold_NoDO/"
with_do_dir = "/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/OVERTRAINCHECK/training_note/training_kfold_WithDO/"

class HistoryHolder :
    def __init__(self, name = '', files = []) :
        self.name = name
        self.files = []
        self.histories = self.load_histories(files)
        print('HistoryHolder {} loaded {} histories'.format(name, len(self.histories)))

    def load_histories(self, files) :
        out = []
        for f in files :
            out.append(pickle.load(open(f, 'rb')))
        return out

def adjust_axes(ax) :

    ax.tick_params(axis = 'both', which = 'both', direction = 'in',
        labelleft = True, bottom = True, top = True, left = True)
    ax.grid(color = 'k', which = 'both', linestyle = '-', lw = 1, alpha = 0.2, zorder = 0)
    ax.set_xlim([0,1])

def get_curves(hh, which) :

    train_key = { 'roc' : '',
                    'acc' : 'categorical_accuracy',
                    'loss' : 'loss' } [which]
    val_key = { 'roc' : 'val_roc_auc',
                'acc' : 'val_categorical_accuracy',
                'loss' : 'val_loss' } [which]

    train_info = []
    val_info = []

    if train_key != '' :

        y_train_folds = []
        n_ep = -1

        means = []
        errors = []

        for ih, h in enumerate(hh.histories) :
            history = h[train_key]
            y_train_folds.append(history)
            e = len(history)
            if n_ep < 0 : n_ep = e
            else :
                if e != n_ep :
                    print('WARNING Found TRAIN k-fold with different number of epochs (expect: {}, got: {})'.format(n_ep, e))
            y_train_folds.append(history)

        for epoch_num in range(n_ep) :
            # get data for this epoch across all folds
            vals = []
            n_folds = len(y_train_folds)
            for fold_num in range(n_folds) :
                fold = y_train_folds[fold_num]
                vals.append(fold[epoch_num])

            val_arr = np.array(vals)

            mean = np.mean(val_arr)
            err = np.std(val_arr)

            means.append(mean)
            errors.append(err)
        train_info = [means, errors]

    if val_key != '' :
        y_train_folds = []
        n_ep = -1
        means = []
        errors = []
        for ih, h in enumerate(hh.histories) :
            history = h[val_key]
            e = len(history)
            if n_ep < 0 : n_ep = e
            else :
                if e != n_ep :
                    print('WARNING Found VAL k-fold with different number of epochs (expect: {}, got: {})'.format(n_ep, e))
            y_train_folds.append(history)
        for epoch_num in range(n_ep) :
            vals = []
            n_folds = len(y_train_folds)
            for fold_num in range(n_folds) :
                fold = y_train_folds[fold_num]
                val = fold[epoch_num]
                if 'roc' in val_key :
                    val = val[epoch_num]
                vals.append(val)
            val_arr = np.array(vals)
            mean = np.mean(val_arr)
            err = np.std(val_arr)# / np.sqrt(2)
            means.append(mean)
            errors.append(err)
        val_info = [means, errors]

    return train_info, val_info

            

def make_roc_plot(hh_no, hh_with) :

    no_train_info, no_val_info = get_curves(hh_no, 'roc')
    with_train_info, with_val_info = get_curves(hh_with, 'roc')

    fig, ax = plt.subplots(1,1)
    adjust_axes(ax)
    ax.set_xlabel('Training Epoch (Normalized)', horizontalalignment = 'right', x = 1)
    ax.set_ylabel('Validation ROC AUC [%]', horizontalalignment = 'right', y = 1)

    ax.set_ylim([100 * 0.93, 100 * 0.942])

    # draw the mean curves
    no_mean_val = 100. * np.array(no_val_info[0])
    with_mean_val = 100. * np.array(with_val_info[0])

    x_vals = np.arange(len(no_mean_val))
    x_vals = [float(x) / len(x_vals) for x in x_vals]

    ax.plot(x_vals, no_mean_val, color = 'r', linestyle = '-') #, label = 'No Dropout')
    ax.plot(x_vals, with_mean_val, color = 'b', linestyle = '-')#, label = 'With Dropout')

    # error bands
    no_err = 100. * np.array(no_val_info[1])
    with_err = 100. * np.array(with_val_info[1])
    no_err_up, no_err_dn = no_mean_val + no_err, no_mean_val - no_err
    with_err_up, with_err_dn = with_mean_val + with_err, with_mean_val - with_err

    ax.fill_between(x_vals, no_err_dn, no_err_up, alpha = 0.6, facecolor = 'r', edgecolor = 'none', label = 'No Dropout')
    ax.fill_between(x_vals, with_err_dn, with_err_up, alpha = 0.6, facecolor = 'b', edgecolor = 'none', label = 'With Dropout')

    ax.legend(loc = 'best')
    fig.savefig('./plots/overtrain_check_DO_roc_auc.pdf', bbox_inches = 'tight', dpi = 200)

def make_acc_plot(hh_no, hh_with) :

    no_train_info, no_val_info = get_curves(hh_no, 'acc')
    with_train_info, with_val_info = get_curves(hh_with, 'acc')

    fig, ax = plt.subplots(1,1)
    adjust_axes(ax)
    ax.set_xlabel('Training Epoch (Normalized)', horizontalalignment = 'right', x = 1)
    ax.set_ylabel('Classification Accuracy [%]', horizontalalignment = 'right', y = 1)
    ax.set_ylim([75, 81.5])

    # means for training
    no_mean_train = 100. * np.array(no_train_info[0])
    with_mean_train = 100. * np.array(with_train_info[0])

    # means for validation
    no_mean_val = 100. * np.array(no_val_info[0])
    with_mean_val = 100. * np.array(with_val_info[0])

    x_vals = np.arange(len(no_mean_train))
    x_vals = [float(x) / len(x_vals) for x in x_vals]

    # plot mean vals
    ax.plot(x_vals, no_mean_train, color = 'r', linestyle = '--')
    ax.plot(x_vals, with_mean_train, color = 'b', linestyle = '--')
    ax.plot(x_vals, no_mean_val, color = 'r', linestyle = '-')
    ax.plot(x_vals, with_mean_val, color = 'b', linestyle = '-')

    # error bands
    no_err_train = 100. * np.array( no_train_info[1] )
    no_err_val = 100. * np.array( no_val_info[1] )
    with_err_train = 100. * np.array( with_train_info[1] )
    with_err_val = 100. * np.array( with_val_info[1] )

    no_err_train_up, no_err_train_dn = no_mean_train + no_err_train, no_mean_train - no_err_train
    no_err_val_up, no_err_val_dn = no_mean_val + no_err_val, no_mean_val - no_err_val
    with_err_train_up, with_err_train_dn = with_mean_train + with_err_train, with_mean_train - with_err_train
    with_err_val_up, with_err_val_dn = with_mean_val + with_err_val, with_mean_val - with_err_val

    ax.fill_between(x_vals, no_err_train_dn, no_err_train_up, alpha = 0.6, facecolor = 'r', edgecolor = 'none', label = 'No Dropout (Train)') 
    ax.fill_between(x_vals, no_err_val_dn, no_err_val_up, alpha = 0.6, facecolor = 'r', hatch = '\\\\\\', edgecolor = 'none', label = 'No Dropout (Valid)')
    ax.fill_between(x_vals, with_err_train_dn, with_err_train_up, alpha = 0.6, facecolor = 'b', edgecolor = 'none', label = 'With Dropout (Train)')
    ax.fill_between(x_vals, with_err_val_dn, with_err_val_up, alpha = 0.6, facecolor = 'b', hatch = '////', edgecolor = 'none', label = 'With Dropout (Valid)')

    ax.legend(loc = 'best')
    fig.savefig('./plots/overtrain_check_DO_acc.pdf', bbox_inches = 'tight', dpi = 200)
    
def make_loss_plot(hh_no, hh_with) :

    no_train_info, no_val_info = get_curves(hh_no, 'loss')
    with_train_info, with_val_info = get_curves(hh_with, 'loss')

    fig, ax = plt.subplots(1,1)
    adjust_axes(ax)
    ax.set_xlabel('Training Epoch (Normalized)', horizontalalignment = 'right', x = 1)
    ax.set_ylabel('Training Loss [a.u.]', horizontalalignment = 'right', y = 1)
    ax.set_ylim([0.42, 0.58])

    # means for training
    no_mean_train = np.array(no_train_info[0])
    with_mean_train = np.array(with_train_info[0])

    # means for validation
    no_mean_val = np.array(no_val_info[0])
    with_mean_val = np.array(with_val_info[0])

    x_vals = np.arange(len(no_mean_train))
    x_vals = [float(x) / len(x_vals) for x in x_vals]

    # plot mean vals
    ax.plot(x_vals, no_mean_train, color = 'r', linestyle = '--')
    ax.plot(x_vals, with_mean_train, color = 'b', linestyle = '--')
    ax.plot(x_vals, no_mean_val, color = 'r', linestyle = '-')
    ax.plot(x_vals, with_mean_val, color = 'b', linestyle = '-')

    # error bands
    no_err_train = np.array( no_train_info[1] )
    no_err_val = np.array( no_val_info[1] )
    with_err_train = np.array( with_train_info[1] )
    with_err_val = np.array( with_val_info[1] )

    no_err_train_up, no_err_train_dn = no_mean_train + no_err_train, no_mean_train - no_err_train
    no_err_val_up, no_err_val_dn = no_mean_val + no_err_val, no_mean_val - no_err_val
    with_err_train_up, with_err_train_dn = with_mean_train + with_err_train, with_mean_train - with_err_train
    with_err_val_up, with_err_val_dn = with_mean_val + with_err_val, with_mean_val - with_err_val

    ax.fill_between(x_vals, no_err_train_dn, no_err_train_up, alpha = 0.6, facecolor = 'r', edgecolor = 'none', label = 'No Dropout (Train)') 
    ax.fill_between(x_vals, no_err_val_dn, no_err_val_up, alpha = 0.6, facecolor = 'r', hatch = '\\\\\\', edgecolor = 'none', label = 'No Dropout (Valid)')
    ax.fill_between(x_vals, with_err_train_dn, with_err_train_up, alpha = 0.6, facecolor = 'b', edgecolor = 'none', label = 'With Dropout (Train)')
    ax.fill_between(x_vals, with_err_val_dn, with_err_val_up, alpha = 0.6, facecolor = 'b', hatch = '////', edgecolor = 'none', label = 'With Dropout (Valid)')

    ax.legend(loc = 'best')
    fig.savefig('./plots/overtrain_check_DO_loss.pdf', bbox_inches = 'tight', dpi = 200)

def main() :

    parser = argparse.ArgumentParser(
        description = 'Plot the fit history across k-folds'
    )
    parser.add_argument('--which', default = 'all',
        help = 'Which history plot do you want [all,roc,acc,loss] (default: all)'
    )
    args = parser.parse_args()

    do_roc = False
    do_acc = False
    do_loss = False
    if args.which.lower() == 'all' :
        do_roc = True
        do_acc = True
        do_loss = True
    elif args.which.lower() == 'roc' :
        do_roc = True
    elif args.which.lower() == 'acc' :
        do_acc = True
    elif args.which.lower() == 'loss' :
        do_loss = True

    no_do_files = glob.glob('{}/fit_history*.pkl'.format(no_do_dir))
    with_do_files = glob.glob('{}/fit_history*.pkl'.format(with_do_dir))
    n_no = len(no_do_files)
    n_with = len(with_do_files)
    if n_no != n_with :
        print('ERROR number of files with (={}) and without (={}) dropout differ'.format(n_with, n_no))
        sys.exit()

    hh_no = HistoryHolder(name = 'no_do', files = no_do_files)
    hh_with = HistoryHolder(name = 'with_do', files = with_do_files)

    if do_roc :
        make_roc_plot(hh_no, hh_with)
    if do_acc :
        make_acc_plot(hh_no, hh_with)
    if do_loss :
        make_loss_plot(hh_no, hh_with)



    
#__________________________
if __name__ == '__main__' :
    main()
