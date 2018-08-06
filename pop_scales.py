#!/usr/bin/env python3

import sys
import argparse
from preprocess import mkdir_p, unique_filename

# h5py
import h5py

def extract_scale_dataset(args, ignore_features = ['eventweight']) :

    """
    From the user-provided path to the input HDF5 file produced
    by 'preprocess.py', extract the scaling dataset and store
    in a new output file. Having the scaling data contained
    in the file made by 'preprocess.py' is useful as it keeps
    it near the data that produced it (makes it easier for training
    and validation) but when expanding out and using the network
    on general datasets, it should be nicer to have this separate.

    Currently this does not worry about handling "features to ignore", like
    event weights. The DataScaler which loads this should handle this.

    Args:
        args : command-line user-input
    """

    scaling_group_name = "scaling"
    scaling_dataset_name = "scaling_data"

    scaling_dataset = None

    with h5py.File(args.input, 'r', libver = 'latest') as input_file :

        if scaling_group_name in input_file :
            scaling_group = input_file[scaling_group_name]
            scaling_dataset = scaling_group[scaling_dataset_name]
        else :
            raise Exception("Input file (={}) does not contained the expected \
                scaling dataset".format(args.input))

        output_name = args.input.split("/")[-1].replace(".h5","").replace(".hdf5","")
        output_name += "_scaling_data.h5"
        if args.outdir != "" :
            mkdir_p(args.outdir)
        output_name = "{}/{}".format(args.outdir, output_name)
        
#        if args.output != "" :
#            output_name = args.output

        with h5py.File(output_name, 'w', libver = 'latest') as output_file :
            scaling_group = output_file.create_group(scaling_group_name)

            # store the name of the original input file so that we can (loosely)
            # correlated it with this output file if this output file's
            # name drastically differs
            scaling_group.attrs['original_input_file'] = args.input

            out_ds = scaling_group.create_dataset(scaling_dataset_name, shape = scaling_dataset.shape, \
                dtype = scaling_dataset.dtype, data = scaling_dataset, maxshape = (None,))

def main() :

    parser = argparse.ArgumentParser(description = "A way to extract the scaling dataset\
        from the HDF5 files produced by 'preprocess.py'")
    parser.add_argument("-i", "--input", help = "Provide an HDF5 file produced by\
        'preprocess.py'", required = True)
    parser.add_argument("-n", "--name", help = "Provide an output filename (default is\
        based on input filename)", default = "")
    parser.add_argument("--outdir", help = "Provide an output directory to store the output", 
        default = "./")
    args = parser.parse_args()

    extract_scale_dataset(args)

    

if __name__ == "__main__" :
    main()
