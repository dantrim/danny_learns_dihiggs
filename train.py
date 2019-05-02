#!/usr/bin/env python

import argparse
import sys
import os
from time import time
from preprocess import mkdir_p, unique_filename
import pickle

# h5py
import h5py

# keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import SGD
from keras import regularizers
from keras import initializers
import keras

import wwbb_models

# numpy
import numpy as np
seed = 347
np.random.seed(seed)

class Sample :

    """
    Sample

    This class will hold the feature data for a given sample.
    """

    def __init__(self, name = "", class_label = -1, input_data = None) :
        """
        Sample constructor

        Args :
            name : descriptive name of the sample (obtained from the input
                pre-processed file)
            input_data : numpy array of the data from the pre-processed file
                (expects an array of dtype = np.float64, not a structured array!)
            class_label : input class label as found in the input pre-processed
                file
        """

        if input_data.dtype != np.float64 :
            raise Exception("ERROR Sample input data must be type 'np.float64', input is '{}'".format(input_data.dtype))

        if class_label < 0 :
            raise ValueError("ERROR Sample (={})class label is not set (<0)".format(name, class_label))

        print("Creating sample {} (label = {})".format(name, class_label))

        self._name = name
        self._class_label = class_label
        self._input_data = input_data
        self._regression_inputs = None

    def name(self) :
        return self._name
    def class_label(self) :
        return self._class_label
    def data(self) :
        return self._input_data
    @property
    def regression_inputs(self) :
        return self._regression_inputs
    @regression_inputs.setter
    def regression_inputs(self, data) :
        self._regression_inputs = data
        


class DataScaler :

    """
    DataScaler

    This class will hold the scaling information needed for the training
    features (variables) contained in the input, pre-processed file.
    Its constructor takes as input the scaling data dataset object
    contained in the pre-processed file and it builds the associated
    feature-list and an associated dictionary to store the scaling
    parameters for each of the input features.
    """

    def __init__(self, scaling_dataset = None, ignore_features = []) :

        """
        ScalingData constructor

        Args:
            scaling_dataset : input HDF5 dataset object which contains the
                scaling data and feature-list
        """

        self._raw_feature_list = []
        self._feature_list = []
        self._scaling_dict = {}
        self._mean = []
        self._scale = []
        self._var = []
        self.load(scaling_dataset, ignore_features)

    def load(self, scaling_dataset = None, ignore_features = []) :

        self._raw_feature_list = list( scaling_dataset['name'] )
        self._feature_list = list( filter( lambda x : x not in ignore_features, self._raw_feature_list ) )

        #self._mean = scaling_dataset['mean']
        #self._scale = scaling_dataset['scale']
        #self._var = scaling_dataset['var']


        for x in scaling_dataset :
            name, mean, scale, var = x['name'], x['mean'], x['scale'], x['var']
            if name in ignore_features : continue
            self._scaling_dict[name] = { 'mean' : mean, 'scale' : scale, 'var' : var }
            self._mean.append(mean)
            self._scale.append(scale)
            self._var.append(var)

        self._mean = np.array(self._mean, dtype = np.float64)
        self._scale = np.array(self._scale, dtype = np.float64)
        self._var = np.array(self._var, dtype = np.float64)

    def raw_feature_list(self) :
        return self._raw_feature_list

    def feature_list(self) :
        return self._feature_list

    def scaling_dict(self) :
        return self._scaling_dict

    def get_params(self, feature = "") :
        if feature in self._scaling_dict :
            return self._scaling_dict[feature]
        raise KeyError("requested feature (={}) not found in set of scaling features".format(feature))

    def mean(self) :
        return self._mean
    def scale(self) :
        return self._scale
    def var(self) :
        return self._var

def floatify(input_array, feature_list) :
    ftype = [(name, float) for name in feature_list]
    return input_array.astype(ftype).view(float).reshape(input_array.shape + (-1,))

def load_input_file(args) :

    """
    Check that the provided input HDF5 file is of the expected form
    as defined by the pre-processing. Exits if this is not the case.
    Returns a list of the sample names found in the file.

    Args :
        args : user input to the executable
    """

    # check that the file can be found
    if not os.path.isfile(args.input) :
        print("ERROR provided input file (={}) is not found or is not a regular file".format(args.input))
        sys.exit()

    samples_group_name = "samples"
    scaling_group_name = "scaling"
    scaling_data_name = "scaling_data"

    found_samples = False
    found_scalings = False
    samples = []
    data_scaler = None

    features_to_ignore = ["eventweight", "eventNumber"]
    if args.regress != "" :
        features_to_ignore.append(args.regress)

    with h5py.File(args.input, 'r') as input_file :

        # look up the scalings first, in order to build the feature list used for the Sample creation
        if scaling_group_name in input_file :
            found_scalings = True
            scaling_group = input_file[scaling_group_name]
            scaling_dataset = scaling_group[scaling_data_name]
            data_scaler = DataScaler( scaling_dataset = scaling_dataset, ignore_features = features_to_ignore )
            print("DataScaler found {} features to train on (there were {} total features in the input)".format( len(data_scaler.feature_list()), len(data_scaler.raw_feature_list() )))
        else :
            print("scaling group (={}) not found in file".format(scaling_group_name))
            sys.exit()

        # now build the samples
        if samples_group_name in input_file :
            found_samples = True
            sample_group = input_file[samples_group_name]
            for p in sample_group :
                process_group = sample_group[p]
                class_label = process_group.attrs['training_label']
                s = Sample(name = p, class_label = int(class_label),
                    input_data = floatify( process_group['train_features'][tuple(data_scaler.feature_list())], data_scaler.feature_list() ) )
                if args.regress :
                    s.regression_inputs = floatify( process_group['train_features'][tuple( [args.regress] )], [args.regress] )
                samples.append(s)

        else :
            print("samples group (={}) not found in file".format(samples_group_name))
            sys.exit()

    samples = sorted(samples, key = lambda x: x.class_label())

    return samples, data_scaler

def build_combined_input(training_samples, data_scaler = None, scale = True, regress_var = "") :

    targets = []
    # used extended slicing to partition arbitrary number of samples
    sample0, sample1, *other = training_samples

    targets.extend( np.ones( sample0.data().shape[0] ) * sample0.class_label() )
    targets.extend( np.ones( sample1.data().shape[0] ) * sample1.class_label() )

    inputs = np.concatenate( (sample0.data(), sample1.data()), axis = 0)
    for sample in other :
        inputs = np.concatenate( (inputs, sample.data()) , axis = 0 )
        targets.extend( np.ones( sample.data().shape[0] ) * sample.class_label() )

    # perform scaling
    if scale :
        inputs = (inputs - data_scaler.mean()) / data_scaler.scale()

    targets = np.array(targets, dtype = int )

    regress_targets = []
    if regress_var != "" :
        regress_targets.extend( sample0.regression_inputs )
        regress_targets.extend( sample1.regression_inputs )
        for sample in other :
            regress_targets.extend( sample.regression_inputs )
        regress_targets = np.array( regress_targets, dtype = np.float64 )

    return inputs, targets, regress_targets

def build_and_train(wwbb_model, n_inputs, n_outputs, input_features, targets) :

    wwbb_model.build_model(n_inputs, n_outputs)
    #wwbb_model.fit_kfold(n_outputs, input_features, targets, n_epochs = 100, batch_size = 10000)
    wwbb_model.fit(n_outputs, input_features, targets, n_epochs = 100, batch_size = 10000)
    
    return wwbb_model.model(), wwbb_model.fit_history()

#def build_keras_model( n_inputs, n_outputs ) :
#
#    my_model = wwbb_models.NNHighLevel()
#    my_model.build_model(n_inputs, n_outputs)
#    return n
#    
#
#    n_nodes = 800
#    layer_opts = dict( activation = 'relu', kernel_initializer = initializers.VarianceScaling(scale = 1.0, mode = 'fan_in', distribution = 'normal', seed = seed))
#
#    input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
#    x = Dense( n_nodes, **layer_opts ) (input_layer)
#    x = Dropout(0.8)(x)
#    x = Dense( n_nodes , **layer_opts ) (x)
#    x = Dropout(0.8)(x)
#    x = Dense( n_nodes , **layer_opts ) (x)
#    predictions = Dense( n_outputs, activation = 'softmax', name = "OutputLayer")(x)
#
#    model = Model(inputs = input_layer, outputs = predictions)
#    model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.1, momentum = 0.00,decay=0.0001, nesterov = True), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adagrad(lr = 0.03, decay=0.15), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(amsgrad=True, lr = 0.001, decay=0.05), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'] )
#
#    return model
#
#def train(n_classes, input_features, targets, model, regression_targets = []) :
#
#    # encode the targets
#    targets_encoded = keras.utils.to_categorical(targets, num_classes = n_classes)
#
#    # number of training epochs
#    n_epochs = 100
#
#    # batch size for training
#    batch_size = 10000
#
#    # early stopping callback
#    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_categorical_accuracy', patience = 5, verbose = True)
#
#    # learning rate schedular
#    #lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule, verbose = 1)
#        
#
#    # fit
#    fit_history = None
#    if len(regression_targets) == 0 :
#        fit_history = model.fit(input_features, targets_encoded, epochs = n_epochs, validation_split = 0.2, shuffle = True, batch_size = batch_size, callbacks = [early_stop])
#    else :
#        fit_history = model.fit(input_features, [targets_encoded, regression_targets], epochs = 100, validation_split = 0.2, shuffle = True, batch_size = 9000)
#
#    return model, fit_history
#def build_keras_model( n_inputs, n_outputs ) :
#
#    n_nodes = 1000
#    do_frac = 0.5
#    layer_opts = dict( activation = 'relu', kernel_initializer = initializers.VarianceScaling(scale = 1.0, mode = 'fan_in', distribution = 'normal', seed = seed))
#
#    input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
#    x = Dense( n_nodes, **layer_opts ) (input_layer)
#    x = Dropout(0.8)(x)
#    x = Dense( n_nodes, **layer_opts ) (x)
#    x = Dropout(0.5)(x)
#    x = Dense( n_nodes, **layer_opts ) (x)
#    predictions = Dense( n_outputs, activation = 'softmax', name = "OutputLayer")(x)
#
#    model = Model(inputs = input_layer, outputs = predictions)
#    model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.15, momentum = 0.02,decay=0.0005, nesterov = True), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adagrad(lr = 0.03, decay=0.15), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(amsgrad=False, lr = 0.004, decay=0.05), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'] )
#
#    return model
#
#def train(n_classes, input_features, targets, model, regression_targets = []) :
#
#    # encode the targets
#    targets_encoded = keras.utils.to_categorical(targets, num_classes = n_classes)
#
#    # fit
#    fit_history = None
#    if len(regression_targets) == 0 :
#        fit_history = model.fit(input_features, targets_encoded, epochs = 40, validation_split = 0.2, shuffle = True, batch_size = 15000)
#    else :
#        fit_history = model.fit(input_features, [targets_encoded, regression_targets], epochs = 100, validation_split = 0.2, shuffle = True, batch_size = 9000)
#
#    return model, fit_history

def build_keras_model_regression( n_inputs, n_outputs ) :

    n_nodes = 200
    layer_opts = dict( activation = 'relu', kernel_initializer = initializers.VarianceScaling(scale = 1.0, mode = 'fan_in', distribution = 'normal', seed = seed))

    input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
    d0 = Dense( n_nodes, **layer_opts ) (input_layer)
    d0 = Dropout(0.5)(d0)
    d1 = Dense( n_nodes, **layer_opts )(d0)
    #d1 = Dropout(0.1)(d1)
    d2 = Dense( n_nodes, **layer_opts )(d1)
    #d2 = Dropout(0.1)(d2)
    d3 = Dense( n_nodes, **layer_opts )(d2)

    r0 = Dense(10, **layer_opts) (d2)
    #r1 = Dense(10, **layer_opts) (r0)

    classifier_predictions = Dense( n_outputs, activation = 'softmax', name = "ClassifierOutputLayer" )(d2)
    regression_predictions = Dense(1) (r0)

    model = Model( inputs = input_layer, outputs = [classifier_predictions, regression_predictions] )
    #model.compile( loss = ['categorical_crossentropy', 'mse'], optimizer = keras.optimizers.SGD(lr=0.1, momentum = 0.05,decay=0.01, nesterov = True), metrics = ['categorical_accuracy', 'mae'] )
    #model.compile( loss = ['categorical_crossentropy','mse'], optimizer = keras.optimizers.Adam(amsgrad=False, lr = 0.004, decay=0.01), metrics = ['categorical_accuracy', 'mae'] )
    model.compile(loss = ['categorical_crossentropy', 'mse'], optimizer = 'adam', metrics = ['categorical_accuracy', 'mae'])

    return model

#def build_keras_model( n_inputs, n_outputs ) :
#
#    n_nodes = 200
#    do_frac = 0.5
#    layer_opts = dict( activation = 'relu', kernel_initializer = initializers.VarianceScaling(scale = 1.0, mode = 'fan_in', distribution = 'normal', seed = seed))
#
#    input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
#    x = Dense( n_nodes, **layer_opts ) (input_layer)
#    x = Dropout(0.5)(x)
#    x = Dense( n_nodes, **layer_opts ) (x)
#    x = Dense( n_nodes, **layer_opts ) (x)
#    x = Dense( n_nodes, **layer_opts ) (x)
#   # x = Dense( n_nodes, **layer_opts ) (x)
#    x = Dropout(0.1)(x)
#   # x = Dense( n_nodes, **layer_opts ) (x)
#   # x = Dense( n_nodes, **layer_opts ) (x)
#   # x = Dropout(0.1)(x)
#   # x = Dense( 30, **layer_opts ) (x)
#    #x = Dense( 10, **layer_opts ) (x)
#    #x = Dense( n_nodes, **layer_opts ) (x)
#    #x = Dropout(0.2)(x)
#    #x = Dense( n_nodes, **layer_opts ) (x)
#    #x = Dense( n_nodes, **layer_opts ) (x)
#    #x = Dense( n_nodes, **layer_opts ) (x)
#    predictions = Dense( n_outputs, activation = 'softmax', name = "OutputLayer")(x)
#
#    model = Model(inputs = input_layer, outputs = predictions)
#    model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.1, momentum = 0.05,decay=0.001, nesterov = True), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adagrad(lr = 0.03, decay=0.15), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(amsgrad=False, lr = 0.004, decay=0.05), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'] )
#
#    return model
#
#def train(n_classes, input_features, targets, model) :
#
#    # encode the targets
#    targets_encoded = keras.utils.to_categorical(targets, num_classes = n_classes)
#
#    # fit
#    fit_history = model.fit(input_features, targets_encoded, epochs = 40, validation_split = 0.2, shuffle = True, batch_size = 9000)
#    #fit_history = model.fit(input_features, targets_encoded, epochs = 30, validation_split = 0.2, shuffle = True, batch_size = 4750)
#
#    return model, fit_history

#def build_keras_model( n_inputs, n_outputs ) :
#
#    n_nodes = 100
#    do_frac = 0.5
#    layer_opts = dict( activation = 'relu', kernel_initializer = initializers.VarianceScaling(scale = 1.0, mode = 'fan_in', distribution = 'normal', seed = seed))
#
#    input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
#    x = Dense( n_nodes, **layer_opts ) (input_layer)
#    x = Dropout(0.1)(x)
#    x = Dense( n_nodes, **layer_opts ) (x)
#    x = Dropout(0.1)(x)
#    x = Dense( 30, **layer_opts ) (x)
#    #x = Dense( 10, **layer_opts ) (x)
#    #x = Dense( n_nodes, **layer_opts ) (x)
#    #x = Dropout(0.2)(x)
#    #x = Dense( n_nodes, **layer_opts ) (x)
#    #x = Dense( n_nodes, **layer_opts ) (x)
#    #x = Dense( n_nodes, **layer_opts ) (x)
#    predictions = Dense( n_outputs, activation = 'softmax', name = "OutputLayer")(x)
#
#    model = Model(inputs = input_layer, outputs = predictions)
#    #model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.4, momentum = 0.05,decay=0.001, nesterov = True), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adagrad(lr = 0.03, decay=0.15), metrics = ['categorical_accuracy'] )
#    model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(amsgrad=False, lr = 0.004, decay=0.05), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'] )
#
#    return model
#
#def train(n_classes, input_features, targets, model) :
#
#    # encode the targets
#    targets_encoded = keras.utils.to_categorical(targets, num_classes = n_classes)
#
#    # fit
#    fit_history = model.fit(input_features, targets_encoded, epochs = 200, validation_split = 0.2, shuffle = True, batch_size = 4000)
#    #fit_history = model.fit(input_features, targets_encoded, epochs = 30, validation_split = 0.2, shuffle = True, batch_size = 4750)
#
#    return model, fit_history

#def build_keras_model( n_inputs, n_outputs ) :
#
#    n_nodes = 100
#    do_frac = 0.5
#    layer_opts = dict( activation = 'relu', kernel_initializer = initializers.VarianceScaling(scale = 1.0, mode = 'fan_in', distribution = 'normal', seed = seed))
#
#    input_layer = Input( name = "InputLayer", shape = (n_inputs,) )
#    x = Dense( n_nodes*2, **layer_opts ) (input_layer)
#    x = Dense( n_nodes*2, **layer_opts ) (input_layer)
#    x = Dropout(0.5)(x)
#    x = Dense( n_nodes, **layer_opts ) (x)
#    predictions = Dense( n_outputs, activation = 'softmax', name = "OutputLayer")(x)
#
#    model = Model(inputs = input_layer, outputs = predictions)
#    #model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.SGD(lr=0.2, momentum = 0.02,decay=0.01, nesterov = True), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['categorical_accuracy'] )
#    model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adagrad(lr = 0.03, decay=0.15), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(amsgrad=False, lr = 0.002, decay=0.1), metrics = ['categorical_accuracy'] )
#    #model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'] )
#
#    return model
#
#def train(n_classes, input_features, targets, model) :
#
#    # encode the targets
#    targets_encoded = keras.utils.to_categorical(targets, num_classes = n_classes)
#
#    # fit
#    fit_history = model.fit(input_features, targets_encoded, epochs = 40, validation_split = 0.2, shuffle = True, batch_size = 4000)
#    #fit_history = model.fit(input_features, targets_encoded, epochs = 30, validation_split = 0.2, shuffle = True, batch_size = 4750)
#
#    return model, fit_history

def main() :

    parser = argparse.ArgumentParser(description = "Train a Keras model over you pre-processed files")
    parser.add_argument("-i", "--input",
        help = "Provide input, pre-processed HDF5 file with training, validation, and scaling data",
        required = True)
    parser.add_argument("-m", "--model-name", help = "Provide the name of the model to build and train", required = True)
    parser.add_argument("-o", "--outdir", help = "Provide an output directory do dump files [default: ./]", default = "./")
    parser.add_argument("-n", "--name", help = "Provide output filename descriptor", default = "test")
    parser.add_argument("-v", "--verbose", action = "store_true", default = False,
        help = "Be loud about it")
    parser.add_argument("--regress", help = "Provide a variable to regress on", default = "")
    args = parser.parse_args()

    training_samples, data_scaler = load_input_file(args)
    if len(training_samples) < 2 :
        print("ERROR there are not enough training samples loaded to perform a training")
        sys.exit()
    print("Pre-processed file contained {} samples: {}, {}".format(len(training_samples), [s.name() for s in training_samples], [s.class_label() for s in training_samples]))

    input_features, targets, regression_targets = build_combined_input(training_samples, data_scaler = data_scaler, scale = True, regress_var = args.regress)

    if len(regression_targets) == 0 :
        regression_targets = []

    #model = None
    #if args.regress == "" :
    #    model = build_keras_model( len(data_scaler.feature_list()), len(training_samples) )
    #else :
    #    model = build_keras_model_regression( len(data_scaler.feature_list()), len(training_samples) )

    ## TODO : save the fit_history object for later use
    #model, fit_history = train(len(training_samples), input_features, targets, model, regression_targets = regression_targets)
    n_inputs = len(data_scaler.feature_list())
    n_outputs = len(training_samples)
    my_model = wwbb_models.get_model(args.model_name)
    model, fit_history = build_and_train(my_model, n_inputs, n_outputs, input_features, targets)

    # dump the fit history to file for later use
    #with open("./ml_training_apr3_split_4/fit_history_{}.pkl".format(args.name), 'wb') as pickle_history :
    with open("fit_history_{}.pkl".format(args.name), 'wb') as pickle_history :
        pickle.dump( fit_history.history, pickle_history )

    # save
    job_suff = "_{}".format(args.name) if args.name else ""
    arch_name = "architecture{}.json".format(job_suff)
    weights_name = "weights{}.h5".format(job_suff)

    if args.outdir != "" :
        mkdir_p(args.outdir)
    arch_name = "{}/{}".format(args.outdir, arch_name)
    weights_name = "{}/{}".format(args.outdir, weights_name)

    print("Saving architecture to: {}".format(os.path.abspath(arch_name)))
    print("Saving weights to     : {}".format(os.path.abspath(weights_name)))
    with open(arch_name, 'w') as arch_file :
        arch_file.write(model.to_json())
    model.save_weights(weights_name)

if __name__ == "__main__" :
    main()
