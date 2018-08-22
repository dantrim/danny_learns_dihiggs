#!/usr/bin/env python

from __future__ import print_function
import sys
import os

import argparse
from preprocess import mkdir_p

# h5py
import h5py

# numpy
import numpy as np
from numpy.lib import recfunctions

# nn
from train import DataScaler, build_combined_input, floatify

def inputs_from_text_file(args) :

    if not os.path.isfile(args.input) :
        print("ERROR input text file (={}) is not found!".format(args.input))
        return []

    lines = [l.strip() for l in open(args.input).readlines()]

    out_files = []
    for line in lines :
        if not os.path.isfile(line) :
            print("WARNING file from input text file not found (={})".format(line))
        else :
            out_files.append(line)
    return out_files

def get_input_files(args) :

    """
    Parse the user input to determine the input files
    to open

    Args:
        args : command line arguments
    """

    if args.input.endswith(".txt") :
        return inputs_from_text_file(args)
    else :
        return args.input.split(",")

def load_scaler(args) :

    """
    Load the scaling information from the user-input location
    for the scaling file.

    Args:
        args : command line arguments
    """

    data_scaler = None
    with h5py.File(args.scaling, 'r', libver = 'latest') as input_file :
        if "scaling" in input_file :
            scaling_group = input_file['scaling']
            scaling_dataset = scaling_group['scaling_data']
            data_scaler = DataScaler( scaling_dataset = scaling_dataset, ignore_features = ['eventweight'] )
    return data_scaler

def load_model(args) :

    """
    Load the NN model from the user-provided weights and
    architecture files output by Keras.

    Args:
        args : command line arguments
    """

    from keras.models import model_from_json

    arch = args.arch_file
    weights = args.weights

    json_file = open(arch, 'r')
    loaded_model = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model)
    loaded_model.load_weights(weights)
    #loaded_model.compile

    return loaded_model

def chunk_generator(input_file, chunksize = 100000, dataset_name = "") :

    """
    Construct a generator for iterating over chunks
    of an input HDF5 file.

    Args :
        input_file : input HDF5 file to iterate through
        chunksize : size of chunks to readout during each iteration
        dataset_name : name of input dataset to process
    """


    with h5py.File(input_file, 'r', libver = 'latest') as f :
        dataset = f[dataset_name]

        if "CENTRAL" in input_file and "123456" in input_file :
            n_skip = 19504
            print("FOR {} ONLY CONSIDERING EVENTS AFTER THE {}'th EVENT".format(input_file, n_skip))
            yield dataset[n_skip:]
        else :
            for x in range(0, dataset.size, chunksize) :
                yield dataset[x:x+chunksize]
        

def build_discriminant_array(scores, n_labels) :

    """
    For each of the labels, construct the log ratio discriminant
    of the NN scores for this sample.

    Args :
        scores : NN output scores, shape = (N, n_labels)
        n_labels : number of NN output scores

    Returns :
        A python list of numpy arrays, each numpy array being
        the discriminant for class label i where i is its index
        in the output list.

        Note: the output discriminants are not striped of NaN
        or inf values.
    """

    discriminants = []
    for inum in range(n_labels) :
        numerator = scores[:,inum]
        denominator = np.empty( (scores.shape[0], ), dtype = np.float64 )
        for iden in range(n_labels) :
            if iden == inum : continue
            denominator += scores[:,iden]
        discriminants.append( np.log( numerator / denominator ) )
    return discriminants

def dump_scores(input_file, model, data_scaler, args) :

    """
    From the input HDF5 file, go through it and get the NN output
    for the features, storing them to a single output file whose
    name is based on the input filename.

    Args :
        input_file : input filename for HDF5 file to be opened and processed
        model : loaded Keras model
        data_scaler : loaded DataScaler object used to scale the input features
            prior to network evaluation
        args : command line inputs
    """

    outname = input_file.split("/")[-1].replace(".h5","").replace(".hdf5","")
    outname += "_scores.h5"
    if args.outdir != "" :
        mkdir_p(args.outdir)
    outname = "{}/{}".format(args.outdir, outname)

    #out_ds_created = False
    #out_ds = None

    #gen = chunk_generator(input_file, dataset_name = args.dataset)
    #chunk = next(gen)
    #chunk = chunk [ (chunk['nBJets']==2) ]
    #row_count = chunk.shape[0]

    #weights = chunk['eventweight']
    #input_features = chunk[data_scaler.feature_list()]
    #input_features = floatify(chunk[data_scaler.feature_list()], data_scaler.feature_list())
    #input_features = (input_features - data_scaler.mean()) / data_scaler.scale()
    #scores = model.predict(input_features)
    #n_outputs = scores.shape[1]

    #ds = np.array( list(weights), dtype = [('eventweight', float)])
    #for io in range(n_outputs) :
    #    ds = recfunctions.append_fields( ds , names = 'nn_score_{}'.format(io), data = scores[:,io], dtypes = float ) 
    #dtype = ds.dtype
    #row_count = ds.shape[0]

    dataset_id = 0
    with h5py.File(outname, 'w', libver = 'latest') as outfile :

        for chunk in chunk_generator(input_file, dataset_name = args.dataset) :

            # apply the selection here
            chunk = chunk[ (chunk['nBJets'] >= 1) & (chunk['mt2_bb'] > 65) ]
            if chunk.size == 0 : continue

            weights = chunk['eventweight']
            input_features = chunk[data_scaler.feature_list()]
            input_features = floatify(input_features, data_scaler.feature_list())
            input_features = (input_features - data_scaler.mean()) / data_scaler.scale()
            scores = model.predict(input_features)
            n_outputs = scores.shape[1]
            discriminants = build_discriminant_array(scores, n_outputs)

            ds = np.array( list(weights), dtype = [('eventweight', float)])
            for io in range(n_outputs) :
                ds = recfunctions.append_fields( ds , names = 'nn_score_{}'.format(io), data = scores[:,io], dtypes = np.float64 ) 
            for io in range(n_outputs) :
                ds = recfunctions.append_fields( ds, names = 'nn_disc_{}'.format(io), data = discriminants[io], dtypes = np.float64 )
            maxshape = (None,) + ds.shape[1:]

            dsname = "nn_scores_{}".format(dataset_id)
            out_ds = outfile.create_dataset(dsname, shape = ds.shape, maxshape = maxshape, chunks = ds.shape, dtype = ds.dtype)
            out_ds[:] = ds
            dataset_id += 1

    print(" > output saved : {}".format(os.path.abspath(outname)))

def main() :

    parser = argparse.ArgumentParser(description = "Produce an output file with NN output scores")
    parser.add_argument("-i", "--input", help = "Provide input file(s) or text file", required = True)
    parser.add_argument("--scaling", help = "Provide input scaling file", required = True)
    parser.add_argument("-w", "--weights", help = "Provide NN weights file", required = True)
    parser.add_argument("-a", "--arch-file", help = "Provide NN architecture file", required = True)
    parser.add_argument("-o", "--outdir", help = "Provide an output directory to dump outputs",
        default = "./")
    parser.add_argument("-d", "--dataset", help = "Provide name of input dataset in the input file",
        default = "superNt")

    args = parser.parse_args()

    input_files = get_input_files(args)
    data_scaler = load_scaler(args)
    if data_scaler == None :
        print("error loading data scaler from provided scaling file (={})".format(args.scaling))
        sys.exit()
    model = load_model(args)

    for ifile, f in enumerate(input_files) :
        print("[{}/{}] {}".format(ifile+1, len(input_files), f))
        dump_scores(f, model, data_scaler, args)

if __name__ == "__main__" :
    main()
