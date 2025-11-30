#!/bin/bash
#PBS -N tga_train
#PBS -P hpc.plaksha.px117
#PBS -q high
#PBS -lselect=1:ncpus=4:ngpus=1:centos=skylake
#PBS -lwalltime=06:00:00
#PBS -o /home/hpc/visitor/px117.visitor/scratch/tga_project/out.txt
#PBS -e /home/hpc/visitor/px117.visitor/scratch/tga_project/err.txt

# Move to submission directory
cd $PBS_O_WORKDIR

echo "WORKDIR: $PBS_O_WORKDIR"
hostname

# No CUDA modules on this cluster — remove module load lines

# Load conda properly
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tga
echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python --version

# Safety thread limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Starting training..."
python train_tga.py \
  --train-ann hiertext/train_annotations.json \
  --val-ann hiertext/val_annotations.json \
  --test-ann hiertext/test_annotations.json \
  --img-train hiertext/train \
  --img-val hiertext/validation \
  --img-test hiertext/test \
  --epochs 10 \
  --batch-size 1 \
  --num-workers 1 \
  --resize 1024 \
  --weights maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth



echo "Training complete."

