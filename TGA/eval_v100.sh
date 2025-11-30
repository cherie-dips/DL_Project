#!/bin/bash
#PBS -N tga_eval
#PBS -P hpc.plaksha.px117
#PBS -q high
#PBS -lselect=1:ncpus=2:ngpus=1:centos=skylake
#PBS -lwalltime=00:30:00
#PBS -o eval_out.txt
#PBS -e eval_err.txt

cd $PBS_O_WORKDIR
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tga

python eval_tga.py

