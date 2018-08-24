#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import argparse
import math

# h5py
import h5py

# numpy
import numpy as np

# stats
import significance

def get_s95(bkg_yield = 0.0) :

    if np.isnan(bkg_yield) :
        print("invalid bkg value")
        sys.exit()
    sig = 0.5
    z = 0

    while True :
        z = significance.binomial_exp_z(sig, bkg_yield, 0.33)
        if z > 1.64 :
            break
        if sig > 500 :
            sig = -1
            break
        sig += 0.1
    return sig

def get_upperlimits(args) :

    outputname = args.input.split("/")[-1].replace(".txt","")
    outputname += "_s95"
    if args.suffix != "" :
        outputname += "_{}".format(args.suffix)
    outputname += ".txt"

    with open(outputname, 'w') as output_file :
        header = "#CUT\tS95\n"
        output_file.write(header)

        with open(args.input) as input_file :
            for line in input_file :
                data = line.strip().split()
                if not data : continue
                if "CUT" in data : continue
                cutval = data[0]
                if args.start != "" :
                    if float(cutval) < float(args.start) : continue
                if args.end != "" :
                    if float(cutval) > float(args.end) : continue
                cutyield = data[1]
                cuteff = data[2]
                total = data[3]

                s95 = get_s95(float(cutyield))
                print(" > S95 for {} : {}".format(cutval, s95))
                sline = "{}\t{}\n".format(cutval, s95)
                output_file.write(sline)

def main() :

    parser = argparse.ArgumentParser(description = "Take in yields files for backgrounds and calculate S95 using ZBinomial")
    parser.add_argument("-i", "--input", required = True, help = "Input text file with background yields")
    parser.add_argument("-s", "--suffix", default = "", help = "Append a suffix to any of the outputs")
    parser.add_argument("--start", default = "", help = "Provide a starting point for the S95 cut scan")
    parser.add_argument("--end", default = "", help = "Provide an end point for the S95 cut scan")
    args = parser.parse_args()

    get_upperlimits(args)

if __name__ == "__main__" :
    main()
