#!/bin/bash

# call this script inside of the danny_learns_dihiggs/ directory

export PYTHONPATH=${PWD}:${PYTHONPATH}
export PYTHONPATH=${PWD}/analyze:${PYTHONPATH}
lsetup "lcgenv -p LCG_94python3 x86_64-slc6-gcc62-opt tensorflow" > /dev/null  # TF 1.8.0 and python3
