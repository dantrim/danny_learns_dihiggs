#!/bin/bash

#SBATCH -o logs/preprocess.%A.%a_%j.%N.log
#SBATCH -p nemo_vm_atlsch
#SBATCH -t 0:20:00
#SBATCH -n 1
#SBATCH --mem=10G
#SBATCH --get-user-env


python preprocess.py -i ./filelists/filelist_bfg.txt --outdir ml_train_test -f ./feature_lists/feature_list.txt -d superNt -v
