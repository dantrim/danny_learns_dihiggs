#!/usr/bin/env python

import argparse

import sys
import os
import glob
from time import time

# h5py
import h5py

def inputs_from_text_file(text_file = "") :

    """
    Return a list of *.hdf5 or *.h5 files contained in a text
    file (each line is a separate file)

    Arguments:
        text_file : input *.txt file
    """

    if text_file == "" :
        raise Exception("input text file is an empty string")

    lines = [l.strip() for l in open(text_file).readlines()]
    out = []
    for l in lines :
        if not l : continue
        if l.startswith("#") : continue
        if l.endswith(".h5") or l.endswith(".hdf5") :
            out.append(l)
    return out

def inputs_from_dir(filedir) :

    """
    Return a list of *.hdf5 or *.h5 files contained in
    the provided directory

    Arguments:
        filedir : directory with *.hdf5 or *.hf files inside
    """

    dir_files = glob.glob("{}/*.hdf5".format(filedir))
    dir_files += glob.glob("{}/*.h5".format(filedir))

    return dir_files

def fields_represented(required_fields = [], sample_fields = []) :

    """
    Determine if all entries in an input list of required fields
    are in the list of fields built from an input file

    Arguments:
        required_fields : the list of fields (variables) that we want
        sample_fields : the list of fields (variables) that are currently in the file
    """

    return set(required_fields).issubset(sample_fields)

def datasets_with_name(input_files = [], dataset_name = "", req_fields = None) :

    """
    From an input list of HDF5 files, return a list of such files
    that contain a given top-level dataset node name and 
    a set of fields

    Arguments:
        input_files : input list of HDF5 files
        dataset_name : name of dataset that is required to be in the files
        req_fields : HDF5 fields that must be in the dataset (if passing all, then
                        this string is expected to be a string "ALL_FIELDS"
    """

    out = []
    for ifile in input_files :
        with h5py.File(ifile, 'r', libver = 'latest') as sample_file :
            if dataset_name in sample_file :    
                dtype = sample_file[dataset_name].dtype.names
                sample_fields = list(sample_file[dataset_name].dtype.names)
                if req_fields != "ALL_FIELDS" and len(req_fields) > 0 :
                    if not fields_represented(req_fields, sample_fields) :
                        print("WARNING dataset (={}) does not have all of the required fields, missing {}".format(ifile, set(sample_fields) - set(req_fields)))
                        continue
                print("file {} : {}".format(ifile, sample_file[dataset_name]))
                out.append(ifile)
            else :
                print("WARNING dataset (={}) not found in input file {}".format( dataset_name, ifile ))
    return out

def get_inputs(args) :

    """
    Return a list of *.hdf5 or *.h5 files that are located
    in the user-provided input. Only builds a list of those
    files that contain the user-provided top-level group.
    """

    user_input = args.input
    dsname = args.dataset_name
    requested_fields = get_features(args)
    if requested_fields == "all" :
        requested_fields = "ALL_FIELDS"

    # text file with multiple files
    if os.path.isfile(user_input) and user_input.endswith(".txt") :
        return datasets_with_name(input_files = inputs_from_text_file(user_input), dataset_name = dsname, req_fields = requested_fields)
    # assume a single file
    elif os.path.isfile(user_input) and user_input.endswith(".h5") \
            or user_input.endswith(".hdf5") :
        return datasets_with_name(input_files = [user_input], dataset_name = dsname, req_fields = requested_fields)
    # assume a directory of files
    elif os.path.isdir(user_input) : 
        return datasets_with_name(input_files = inputs_from_dir(user_input), dataset_name = dsname, req_fields = requested_fields)

def features_from_file(input_file) :

    out = []
    lines = [l.strip() for l in open(input_file)]
    for l in lines :
        if not l : continue
        if l.startswith("#") : continue
        out.append(l)
    return out

def get_features(args) :

    """
    Return a list of the features (variables) to slim on.
    """

    if args.feature_list == "all" :
        return []

    if os.path.isfile(args.feature_list) :
        return features_from_file(args.feature_list)
    else :
        return args.feature_list.split(",")

def main() :

    parser = argparse.ArgumentParser(description = "Pre-process your inputs")
    parser.add_argument("-i", "--input",
        help = "Provide input HDF5 files [HDF5 file, text filelist, or directory of files]",
        required = True)
    parser.add_argument("-f", "--feature-list", help = "Provide list of features to slim on [comma-separated list, text file]",
        default = "all")
    parser.add_argument("-d", "--dataset-name", help = "Common dataset name in files",
                required = True)
    parser.add_argument("-v", "--verbose", help = "Be loud about it", default = False,
        action = 'store_true')
    args = parser.parse_args()

    inputs = get_inputs(args)
    print("Found {} input files".format(len(inputs)))


#_________________________________
if __name__ == "__main__" :
    main()
