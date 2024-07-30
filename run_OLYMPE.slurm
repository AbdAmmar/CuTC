#!/bin/bash
#SBATCH -J tc_int
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00


module purge
module load gcc/7.3.0
module load cuda/11.7

#nvcc --version
#nvidia-smi

time ./bin/tc_int
#time ./bin/deb_binding

#nvidia-smi
