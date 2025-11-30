#!/bin/bash
#PBS -N check_gpu
#PBS -P hpc.plaksha.px117
#PBS -q high
#PBS -lselect=1:ncpus=2:ngpus=1:centos=skylake
#PBS -lwalltime=00:05:00
#PBS -o gpu_out.txt
#PBS -e gpu_err.txt

cd $PBS_O_WORKDIR
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tga

python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
EOF

