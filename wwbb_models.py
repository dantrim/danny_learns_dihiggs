#!/usr/bin/env python

import sys
import os

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import SGD
from keras import regularizers
from keras import initializers
import keras
import pickle

import wwbb_callbacks

import numpy as np
seed = 347
np.random.seed(seed)

def get_layer_opts() :
    return dict( activation = 'relu', kernel_initializer = initializers.VarianceScaling(scale = 1.0, mode = 'fan_in', distribution = 'normal', seed = seed))

def get_model(model_name = "") :

    if model_name == "" :
        print("wwbb_models get_model received null model name")
        sys.exit()

    return {
            "WWbbNN" : WWbbNN
    }[model_name]()

class WWbbNN :
    def __init__(self) :
        self._name = "WWbbNN"
        self._model = None
        self._fit_history = None

    def name(self) :
        return self._name

    def model(self) :
        return self._model

    def build_model(self, n_inputs,  n_outputs) :

        #n_nodes = 275
        n_nodes = 250
        layer_opts = get_layer_opts()

        input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
        x = Dense( n_nodes, **layer_opts ) (input_layer)
        x = Dropout(0.5)(x)
        x = Dense( n_nodes, **layer_opts ) (x)
        predictions = Dense( n_outputs, activation = 'softmax', name = "OutputLayer" )(x)

        model = Model( inputs = input_layer, outputs = predictions )
        model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'] )
        self._model = model

    def fit(self, n_classes, input_features, targets, n_epochs = 400, batch_size = 2000) :

        # encode
        targets_encoded = keras.utils.to_categorical(targets, num_classes = n_classes)

        # randomize/shuffle the data
        n_per_sample = int(input_features.shape[0] / n_classes)
        randomize = np.arange(len(input_features))
        np.random.shuffle(randomize)
        shuffled_input_features = input_features[randomize]
        shuffled_targets = targets_encoded[randomize]

        fraction_for_validation = 0.2
        total_number_of_samples = len(shuffled_targets)
        n_for_validation = int(fraction_for_validation * total_number_of_samples)

        x_train, y_train = shuffled_input_features[n_for_validation:], shuffled_targets[n_for_validation:]
        x_val, y_val = shuffled_input_features[:n_for_validation], shuffled_targets[:n_for_validation]

        do_lr_sched = False
        do_early_stop = True
        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, verbose = True, min_delta = 0.001)

        auc_cb = wwbb_callbacks.roc_callback()

        lr_schedule = keras.callbacks.LearningRateScheduler(wwbb_callbacks.lr_step_decay)

        callbacks = []
        if do_lr_sched :
            callbacks.append(lr_schedule)
        if do_early_stop :
            callbacks.append(early_stop)
        callbacks.append(auc_cb)

        self._fit_history = self._model.fit(x_train, y_train, epochs = n_epochs, batch_size = batch_size, shuffle = True, validation_data = (x_val, y_val), callbacks = callbacks)

    def fit_kfold(self, n_classes, input_features, targets, n_epochs = 100, batch_size = 10000) :

        from sklearn.model_selection import StratifiedKFold

        # encode
        # randomize/shuffle the data
        n_per_sample = int(input_features.shape[0] / n_classes)
        randomize = np.arange(len(input_features))
        np.random.shuffle(randomize)
        shuffled_input_features = input_features[randomize]
        shuffled_targets = targets[randomize]

        auc_cb = wwbb_callbacks.roc_callback()

        kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)

        kfold_idx = 0

        fit_histories = []
        for train_data, val_data in kfold.split(shuffled_input_features, shuffled_targets) :

            tx = shuffled_input_features[train_data]
            ty = shuffled_targets[train_data]

            vx = shuffled_input_features[val_data]
            vy = shuffled_targets[val_data]

            # encode
            ty = keras.utils.to_categorical(ty, num_classes = n_classes)
            vy = keras.utils.to_categorical(vy, num_classes = n_classes)

            self.build_model(30, 4)
            fit_history = self._model.fit(tx, ty, epochs = n_epochs, batch_size = batch_size, shuffle = True, validation_data = (vx,vy), callbacks = [auc_cb])
            fit_histories.append(fit_history)

            history_name = "fit_history_kfold_{}.pkl".format(kfold_idx)
            with open(history_name, 'wb') as pickle_history :
                pickle.dump( fit_history.history, pickle_history )
            kfold_idx+=1

        self._fit_history = fit_histories[0] # just use the last one

    def fit_history(self) :
        return self._fit_history
