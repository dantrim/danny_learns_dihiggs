#!/bin/env python

import sys, os
import argparse
import matplotlib.pyplot as plt
import pickle
import numpy as np

history_nom = '/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/OVERTRAINCHECK/training_note/training_earlyStopping_Patience20/fit_history.pkl'
history_patience_40 = '/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/OVERTRAINCHECK/training_note/training_earlyStopping_Patience40/fit_history.pkl'
history_no_earlystop = '/Users/dantrim/workarea/physics_analysis/wwbb/danny_learns_dihiggs/OVERTRAINCHECK/training_note/training_noEarlyStop_200Epochs/fit_history.pkl'

class History :
    def __init__(self, name = '', filename = '', color = '') :
        self.filename = filename
        self.name = name
        self.color = color
        self.history = None

def nice_name(name) :

    return { 'pat20' : 'Early Stopping, Patience 20',
                'pat40' : 'Early Stopping, Patience 40',
                'noES' : 'No Early Stopping, 200 epochs' } [name]

def adjust_axes(ax) :

    ax.tick_params(axis = 'both', which = 'both', direction = 'in',
        labelleft = True, bottom = True, top = True, left = True)
    ax.grid(color = 'k', which = 'both', linestyle = '-', lw = 1, alpha = 0.2, zorder = 0)
    ax.set_xlim([0,1])

def make_roc_plots(histories, args) :

    fig, ax = plt.subplots(1,1)
    #ax.grid(True)
    adjust_axes(ax)

    ax.set_xlabel('Training Epoch (Normalized)', horizontalalignment = 'right', x = 1)
    ax.set_ylabel('Validation ROC AUC [%]', horizontalalignment = 'right', y = 1)

    ax.set_ylim([100*0.92, 100*0.94])

    data_key = 'val_roc_auc'

    offsets = [0, 0.0005, -0.0005]
    n_epochs = []
    for i, h in enumerate(histories) :
        y_vals = h.history[data_key][0]
        y_vals = [y + offsets[i] * y for y in y_vals]
        y_vals = [100. * y for y in y_vals]
        x_vals = np.arange(len(y_vals))
        n_e = len(x_vals)
        n_epochs.append(n_e)
        x_vals = [float(x) / len(x_vals) for x in x_vals]
        label = nice_name(h.name)
        if 'noES' not in h.name :
            label = '%s (# epochs = %d)' % (nice_name(h.name), n_e)
        ax.plot(x_vals, y_vals, color = h.color, label = label)
    ax.legend(loc = 'best')

    fig.savefig('./plots/overtrain_check_roc_auc.pdf', bbox_inches = 'tight', dpi = 200)

def make_acc_plots(histories, args) :

    fig, ax = plt.subplots(1,1)
    adjust_axes(ax)

    ax.set_xlabel('Training Epoch (Normalized)', horizontalalignment = 'right', x = 1)
    ax.set_ylabel('Classificiation Accuracy [%]', horizontalalignment = 'right', y = 1)
    ax.set_ylim([100 * 0.73, 100 * 0.9])

    train_loss_key = 'categorical_accuracy'
    val_loss_key = 'val_categorical_accuracy'

    offsets = [0, 0.0005, -0.0005]
    for i, h in enumerate(histories) :

        y_train = h.history[train_loss_key]
        y_train = [100. * y for y in y_train]
        y_val = h.history[val_loss_key]
        y_val = [100. * y for y in y_val]
        x_vals = np.arange(len(y_val))
        x_vals = [float(x) / len(x_vals) for x in x_vals]
        n_e = len(x_vals)

        base_label = nice_name(h.name)
        if 'noES' not in h.name :
            base_label = '%s (# epochs = %d)' % (nice_name(h.name), n_e)

        label = 'Validation : %s' % base_label
        ax.plot(x_vals, y_val, color = h.color, linestyle = '-', label = label)
        label = 'Train : %s' % base_label
        ax.plot(x_vals, y_train, color = h.color, linestyle = '--', label = label)

    ax.legend(loc = 'best')
    fig.savefig('./plots/overtrain_check_acc.pdf', bbox_inches = 'tight', dpi = 200)

def make_loss_plots(histories, args) :

    fig, ax = plt.subplots(1,1)
    adjust_axes(ax)
    ax.set_xlabel('Training Epoch (Normalized)', horizontalalignment = 'right', x = 1)
    ax.set_ylabel('Loss [a.u.]', horizontalalignment = 'right', y = 1)
    ax.set_ylim([0.32, 0.72])
 
    train_loss_key = 'loss'
    val_loss_key = 'val_loss'

    for i, h in enumerate(histories) :
        y_train = h.history[train_loss_key]
        y_val = h.history[val_loss_key]
        x_vals = np.arange(len(y_val))
        x_vals = [float(x) / len(x_vals) for x in x_vals]
        n_e = len(x_vals)

        base_label = nice_name(h.name)
        if 'noES' not in h.name :
            base_label = '%s (# epochs = %d)' % (nice_name(h.name), n_e)
        label = 'Validation : %s' % base_label
        ax.plot(x_vals, y_val, color = h.color, linestyle = '-', label = label)
        label = 'Train : %s' % base_label
        ax.plot(x_vals, y_train, color = h.color, linestyle = '--', label = label)

    ax.legend(loc = 'best')
    fig.savefig('./plots/overtrain_check_loss.pdf', bbox_inches = 'tight', dpi = 200)

def make_plots(histories, args) :

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

    for h in histories :
        h.history = pickle.load( open(h.filename, 'rb') )

    if do_roc :
        make_roc_plots(histories, args)
    if do_acc :
        make_acc_plots(histories, args)
    if do_loss :
        make_loss_plots(histories, args)

def main() :

    parser = argparse.ArgumentParser(
        description = 'Plot the fit history outputs from NN training'
    )
    parser.add_argument('--which', default = 'all',
        help = 'Which history plot do you want [all,roc,acc,loss] (default: all)'
    )
    args = parser.parse_args()

    hist_nom = History('pat20', history_nom, 'b')
    hist_40 = History('pat40', history_patience_40, 'g')
    hist_noES = History('noES', history_no_earlystop, 'r')
    histories = [hist_nom, hist_40, hist_noES]

    make_plots(histories, args)


#_______________________________________
if __name__ == '__main__' :
    main()
