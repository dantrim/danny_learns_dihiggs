#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import argparse
import math

import numpy as np

class S95 :
    def __init__(self) :
        self._cutval = -1.0
        self._s95 = -1.0
        self._type = None

    @property
    def cutval(self) :
        return self._cutval
    @cutval.setter
    def cutval(self, val) :
        self._cutval = float(val)

    @property
    def s95(self) :
        return self._s95
    @s95.setter
    def s95(self, val) :
        self._s95 = float(val)

    @property
    def type(self) :
        return type
    @type.setter
    def type(self, val) :
        self._type = str(val)

class Count :
    def __init__(self) :
        self._cutval = -1.0
        self._yield = -1.0
        self._eff = -1.0
        self._total_yield = -1.0

    @property
    def cutval(self) :
        return self._cutval
    @cutval.setter
    def cutval(self, val) :
        self._cutval = float(val)

    @property
    def cutyield(self) :
        return self._yield
    @cutyield.setter
    def cutyield(self, val) :
        self._yield = float(val)

    @property
    def eff(self) :
        return self._eff
    @eff.setter
    def eff(self, val) :
        self._eff = float(val)

    @property
    def total_yield(self) :
        return self._total_yield
    @total_yield.setter
    def total_yield(self, val) :
        self._total_yield = float(val)

def load_s95(args) :

    out = {}
    with open(args.s95) as input_file :

        for line in input_file :
            if "CUT" in line : continue
            line = line.strip().split()
            if not line : continue
            cutval = float(line[0])
            s95 = float(line[1])
            if s95 < 0. : continue
            s = S95()
            s.cutval = cutval
            s.s95 = s95
            s.type = args.type
            out[cutval] = s
    return out

def load_acc(input_filename) :

    out = {}
    with open(input_filename) as input_file :
        for line in input_file :
            if "CUT" in line : continue
            line = line.strip().split()
            if not line : continue
            cutval = float(line[0])
            cutyield = float(line[1])
            cuteff = float(line[2])
            total_yield = float(line[3])
            counts = Count()
            counts.cutval = cutval
            counts.cutyield = cutyield
            counts.eff = cuteff
            counts.total_yield = total_yield
            out[cutval] = counts
    return out


def main() :
    parser = argparse.ArgumentParser(description="From input text files calculate the xsec upperlimits")
    parser.add_argument("--s95", required = True, help = "Input S95 text file")
    parser.add_argument("--eff", required = True, help = "Input reco signal acceptance text file")
    parser.add_argument("--acc", required = True, help = "input truth signal acceptance text file")
    parser.add_argument("-t", "--type", required = True, help = "disc or score?")
    args = parser.parse_args()

    s95_dict = load_s95(args)
    truth_dict = load_acc(args.acc)
    reco_dict = load_acc(args.eff)

    lumi_value = 36.1

    truth_evnt_counts = 21298.8 # yield at EVNT/TRUTH stage 

    lowest_UL = 999999
    best_stuff = Count()

    for valid_cut_val in s95_dict :
        s95 = s95_dict[valid_cut_val]
        truth_counts = truth_dict[valid_cut_val]
        reco_counts = reco_dict[valid_cut_val]

        vis_xsec_UL = s95.s95 / lumi_value
        vis_xsec_UL /= 1000. # convert from [fb] to [pb]

        reco_efficiency = reco_counts.cutyield / truth_counts.cutyield
        truth_acceptance = truth_counts.cutyield / truth_evnt_counts
        acceptance_times_efficiency = reco_efficiency * truth_acceptance

        kinematic_efficiency = 0.23849 # GenFiltEfficiency from custom WWbb signal sample with pT > 12 GeV requirement
        kinematic_efficiency = kinematic_efficiency / 0.50 # remove effect of hBB and hWW filters to just get kinematic filter eff from MultiElecMuTau filter

        w_lnu_efficiency = 0.3272**2 # square of the BR for the leptonic decay of W bosons

        den_filter_factor = w_lnu_efficiency * kinematic_efficiency

        denominator = acceptance_times_efficiency
        denominator *= w_lnu_efficiency * kinematic_efficiency
        denominator *= 2.0 * 0.57 * 0.21

        numerator = vis_xsec_UL

        if denominator != 0 and denominator > 0 :
#            print("numerator = {}".format(numerator))
#            print("denominator = {}".format(denominator))
            xsec_ul = numerator / denominator
            #print("CUT {} -> {}".format(valid_cut_val, xsec_ul))
            print("CUT {}: UL {}, S95 {}, A {}, E = {}, AxE = {}, AxExK = {}".format(valid_cut_val, xsec_ul, s95.s95, truth_acceptance, reco_efficiency, acceptance_times_efficiency, acceptance_times_efficiency * den_filter_factor * 2.0 * 0.5809 * 0.2152))
            if xsec_ul < lowest_UL :
                lowest_UL = xsec_ul

                best_stuff.lowest_UL = xsec_ul
                best_stuff.cutval = valid_cut_val
                best_stuff.s95 = s95.s95
                best_stuff.acc = truth_acceptance
                best_stuff.eff = reco_efficiency
                best_stuff.a_times_e = acceptance_times_efficiency

    #print("lowest UL = {}".format(lowest_UL))

    print(60 * "=")
    print("BEST: CUTVAL = {}, UL = {}, S95 = {}, A = {}, E = {}, AxE = {}".format(best_stuff.cutval, best_stuff.lowest_UL, best_stuff.s95, best_stuff.acc, best_stuff.eff, best_stuff.a_times_e))

        

    

if __name__ == "__main__" :
    main()
