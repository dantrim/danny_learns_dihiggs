#!/usr/bin/env python

import sys
import os

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import SGD
from keras import regularizers
from keras import initializers
import keras

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
            "NNSimpleGoodForG2B2" : NNSimpleGoodForG2B2 }[model_name]()

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


class NNTest :
    def __init__(self) :
        self._model = None
        self._fit_history = None

    def model(self) :
        return self._model

    def build_model(self, n_inputs, n_outputs) :

        n_nodes = 200
        layer_opts = get_layer_opts()

        input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
        x = Dense( n_nodes, **layer_opts ) (input_layer)
        x = Dropout(0.5)(x)
        x = Dense( n_nodes, **layer_opts )(x)
        x = Dense( n_nodes, **layer_opts )(x)
        x = Dense( n_nodes, **layer_opts )(x)
        x = Dropout(0.5)(x)
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

