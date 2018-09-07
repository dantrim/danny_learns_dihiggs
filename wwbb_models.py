#!/usr/bin/env python

import sys
import os

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import SGD
from keras import regularizers
from keras import initializers
import keras

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

    return {"NNHighLevel" : NNHighLevel,
            "TTOnlyTarget": TTOnlyTarget,
            "NNTest" : NNTest,
            "NNSimpleGoodForG2B" : NNSimpleGoodForG2B,
            "NNSimpleGoodForG2B2" : NNSimpleGoodForG2B2,
            "NNTest1B" : NNTest1B,
            "NNSimpleGoodForG2B3" : NNSimpleGoodForG2B3,
            "NNTestCallback" : NNTestCallback }[model_name]()

class TTOnlyTarget :
    def __init__(self) :
        self._model = None
        self._fit_history = None

    def model(self) :
        return self._model

    def build_model(self, n_inputs, n_outputs) :

        n_nodes = 50
        layer_opts = get_layer_opts()

        input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
        x = Dense( n_nodes, **layer_opts )(input_layer)
        x = Dropout(0.1)(x)
        x = Dense( n_nodes, **layer_opts )(x)
        x = Dropout(0.1)(x)
        x = Dense( n_nodes, **layer_opts )(x)
        predictions = Dense( n_outputs, activation = 'softmax', name = 'OutputLayer' )(x)
        model = Model( inputs = input_layer, outputs = predictions )
        model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.08, momentum = 0.00,decay=0.00001, nesterov = True), metrics = ['categorical_accuracy'] )
        #model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'])
        self._model = model

    def fit(self, n_classes, input_features, targets, n_epochs = 100, batch_size = 10000) :

        batch_size = 15000

        # encode
        targets_encoded = keras.utils.to_categorical(targets, num_classes = n_classes)

        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_categorical_accuracy', patience = 10, verbose = True)

        self._fit_history = self._model.fit(input_features, targets_encoded, epochs = n_epochs, validation_split = 0.2, shuffle = True, batch_size = batch_size, callbacks = [early_stop])

    def fit_history(self) :
        return self._fit_history

class NNHighLevel :
    def __init__(self) :
        self._model = None
        self._fit_history = None

    def model(self) :
        return self._model

    def build_model(self, n_inputs, n_outputs) :

        n_nodes = 1000
        layer_opts = get_layer_opts()

        input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
        x = Dense( n_nodes, **layer_opts ) (input_layer)
        x = Dropout(0.8)(x)
        x = Dense( n_nodes, **layer_opts )(x)
        x = Dropout(0.8)(x)
        x = Dense( n_nodes, **layer_opts )(x)
        predictions = Dense( n_outputs, activation = 'softmax', name = "OutputLayer" )(x)

        model = Model( inputs = input_layer, outputs = predictions )
        model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.1, momentum = 0.00,decay=0.0001, nesterov = True), metrics = ['categorical_accuracy'] )
        self._model = model

    def fit(self, n_classes, input_features, targets, n_epochs = 100, batch_size = 10000) :

        # encode
        targets_encoded = keras.utils.to_categorical(targets, num_classes = n_classes)

        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_categorical_accuracy', patience = 5, verbose = True)

        self._fit_history = self._model.fit(input_features, targets_encoded, epochs = n_epochs, validation_split = 0.2, shuffle = True, batch_size = batch_size, callbacks = [early_stop])

    def fit_history(self) :
        return self._fit_history

class NNSimpleGoodForG2B : # Sep4 : this model, wtih data at >=2 bjets only gave ~equivalent ROC curve with cut & count using the log(hh / all-bkg) discriminant
    def __init__(self) :
        self._model = None
        self._fit_history = None

    def model(self) :
        return self._model

    def build_model(self, n_inputs, n_outputs) :

        n_nodes = 60
        layer_opts = get_layer_opts()

        input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
        x = Dense( n_nodes, **layer_opts ) (input_layer)
        x = Dense( n_nodes, **layer_opts )(x)
        x = Dense( n_nodes*2, **layer_opts )(x)
        x = Dense( n_nodes*2, **layer_opts )(x)
        x = Dropout(0.2)(x)
        predictions = Dense( n_outputs, activation = 'softmax', name = "OutputLayer" )(x)

        model = Model( inputs = input_layer, outputs = predictions )
        model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.5, momentum = 0.02,decay=0.00001, nesterov = True), metrics = ['categorical_accuracy'] )
        self._model = model

    def fit(self, n_classes, input_features, targets, n_epochs = 100, batch_size = 10000) :

        # encode
        targets_encoded = keras.utils.to_categorical(targets, num_classes = n_classes)

        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_categorical_accuracy', patience = 20, verbose = True)

        self._fit_history = self._model.fit(input_features, targets_encoded, epochs = n_epochs, validation_split = 0.2, shuffle = True, batch_size = batch_size, callbacks = [early_stop])

    def fit_history(self) :
        return self._fit_history

class NNSimpleGoodForG2B2 :
    def __init__(self) :
        self._model = None
        self._fit_history = None

    def model(self) :
        return self._model

    def build_model(self, n_inputs, n_outputs) :

        n_nodes = 60
        layer_opts = get_layer_opts()

        input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
        x = Dense( n_nodes, **layer_opts ) (input_layer)
        x = Dropout(0.2)(x)
        x = Dense( n_nodes, **layer_opts )(x)
        x = Dense( n_nodes, **layer_opts )(x)
        x = Dense( n_nodes, **layer_opts )(x)
        x = Dropout(0.2)(x)
        predictions = Dense( n_outputs, activation = 'softmax', name = "OutputLayer" )(x)

        model = Model( inputs = input_layer, outputs = predictions )
        model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.2, momentum = 0.02,decay=0.0001, nesterov = True), metrics = ['categorical_accuracy'] )
        self._model = model

    def fit(self, n_classes, input_features, targets, n_epochs = 100, batch_size = 10000) :

        # encode
        targets_encoded = keras.utils.to_categorical(targets, num_classes = n_classes)

        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_categorical_accuracy', patience = 20, verbose = True)

        self._fit_history = self._model.fit(input_features, targets_encoded, epochs = n_epochs, validation_split = 0.2, shuffle = True, batch_size = batch_size, callbacks = [early_stop])

    def fit_history(self) :
        return self._fit_history

class NNTest1B :
    def __init__(self) :
        self._model = None
        self._fit_history = None

    def model(self) :
        return self._model

    def build_model(self, n_inputs, n_outputs) :

        n_nodes = 34
        layer_opts = get_layer_opts()

        input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
        x = Dense( n_nodes, **layer_opts ) (input_layer)
        x = Dense( n_nodes, **layer_opts )(x)
        #x = Dense( n_nodes, **layer_opts )(x)
        #x = Dense( n_nodes, **layer_opts )(x)
        #x = Dropout(0.1)(x)
        #x = Dense( n_nodes, **layer_opts )(x)
        #x = Dense( n_nodes, **layer_opts )(x)
        predictions = Dense( n_outputs, activation = 'softmax', name = "OutputLayer" )(x)

        model = Model( inputs = input_layer, outputs = predictions )
        model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.1, momentum = 0.02,decay=0.0001, nesterov = True), metrics = ['categorical_accuracy'] )
        self._model = model

    def fit(self, n_classes, input_features, targets, n_epochs = 100, batch_size = 10000) :
        n_epochs = 200

        # encode
        targets_encoded = keras.utils.to_categorical(targets, num_classes = n_classes)

        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, verbose = True, min_delta = 0.01)

        self._fit_history = self._model.fit(input_features, targets_encoded, epochs = n_epochs, validation_split = 0.2, shuffle = True, batch_size = batch_size, callbacks = [early_stop])

    def fit_history(self) :
        return self._fit_history


class NNTest :
    def __init__(self) :
        self._model = None
        self._fit_history = None

    def model(self) :
        return self._model

    def build_model(self, n_inputs, n_outputs) :

        n_nodes = 50
        layer_opts = get_layer_opts()

        input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
        x = Dense( n_nodes, **layer_opts ) (input_layer)
        x = Dense( n_nodes, **layer_opts )(x)
       # x = Dense( n_nodes, **layer_opts )(x)
       # x = Dense( n_nodes, **layer_opts )(x)
       # x = Dropout(0.5)(x)
       # x = Dense( n_nodes, **layer_opts )(x)
        predictions = Dense( n_outputs, activation = 'softmax', name = "OutputLayer" )(x)

        model = Model( inputs = input_layer, outputs = predictions )
        model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.05, momentum = 0.0,decay=0.0000, nesterov = True), metrics = ['categorical_accuracy'] )
        #model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'])
        self._model = model

    def fit(self, n_classes, input_features, targets, n_epochs = 100, batch_size = 1000) :

        # encode
        targets_encoded = keras.utils.to_categorical(targets, num_classes = n_classes)

        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, verbose = True)

        self._fit_history = self._model.fit(input_features, targets_encoded, epochs = n_epochs, validation_split = 0.2, shuffle = True, batch_size = batch_size, callbacks = [early_stop])

    def fit_history(self) :
        return self._fit_history

class NNSimpleGoodForG2B3 : # Sep5 : this model, wtih data at >=2 bjets only gave ~equivalent ROC curve with cut & count using the log(hh / all-bkg) discriminant
    def __init__(self) :
        self._name = "NNSimpleGoodForG2B3"
        self._model = None
        self._fit_history = None

    def name(self) :
        return self._name

    def model(self) :
        return self._model

    def build_model(self, n_inputs, n_outputs) :

        n_nodes = 50
        layer_opts = get_layer_opts()

        input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
        x = Dense( n_nodes, **layer_opts ) (input_layer)
        x = Dense( n_nodes, **layer_opts )(x)
        x = Dense( n_nodes*2, **layer_opts )(x)
        #x = Dense( n_nodes*2, **layer_opts )(x)
        x = Dropout(0.1)(x)
        predictions = Dense( n_outputs, activation = 'softmax', name = "OutputLayer" )(x)

        model = Model( inputs = input_layer, outputs = predictions )
        model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.2, momentum = 0.02,decay=0.00001, nesterov = True), metrics = ['categorical_accuracy'] )
        self._model = model

    def fit(self, n_classes, input_features, targets, n_epochs = 100, batch_size = 10000) :

        # encode
        targets_encoded = keras.utils.to_categorical(targets, num_classes = n_classes)

        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_categorical_accuracy', patience = 20, verbose = True)

        self._fit_history = self._model.fit(input_features, targets_encoded, epochs = n_epochs, validation_split = 0.2, shuffle = True, batch_size = batch_size, callbacks = [early_stop])

    def fit_history(self) :
        return self._fit_history

class NNTestCallback :
    def __init__(self) :
        self._name = "NNTestCallback"
        self._model = None
        self._fit_history = None

    def name(self) :
        return self._name

    def model(self) :
        return self._model

    def build_model(self, n_inputs, n_outputs) :

        n_nodes = 275
        layer_opts = get_layer_opts()

        input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
        x = Dense( n_nodes, **layer_opts ) (input_layer)
        x = Dense( n_nodes, **layer_opts ) (x)
        x = Dropout(0.5)(x)
        x = Dense( n_nodes, **layer_opts ) (x)
        x = Dense( n_nodes, **layer_opts ) (x)
        x = Dropout(0.1)(x)
        x = Dense( n_nodes, **layer_opts ) (x)
        #x = Dense( 100, **layer_opts )(x)
        #x = Dense( 100, **layer_opts )(x)
        #x = Dense( n_nodes, **layer_opts )(x)
 #       x = Dropout(0.5)(x)
 #       x = Dense( n_nodes*2, **layer_opts )(x)
 #       x = Dense( n_nodes*2, **layer_opts )(x)
        #x = Dense( n_nodes*2, **layer_opts )(x)
        predictions = Dense( n_outputs, activation = 'softmax', name = "OutputLayer" )(x)

        model = Model( inputs = input_layer, outputs = predictions )
        model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adadelta(lr=1.0, rho = 0.95, epsilon = 1e-08, decay = 0.0), metrics = ['categorical_accuracy'] )
        #model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.2, momentum = 0.02,decay=0.00001, nesterov = True), metrics = ['categorical_accuracy'] )
        #model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adagrad(lr = 0.03, decay=0.15), metrics = ['categorical_accuracy'] )
        #model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(amsgrad=True, lr = 0.001, decay=0.05), metrics = ['categorical_accuracy'] )
        #model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'] )
        self._model = model

    def fit(self, n_classes, input_features, targets, n_epochs = 100, batch_size = 10000) :

        n_epochs = 400

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
        #self._fit_history = self._model.fit(x_train, y_train, epochs = n_epochs, batch_size = batch_size, validation_data = (x_val, y_val), callbacks = callbacks)

    def fit_history(self) :
        return self._fit_history
