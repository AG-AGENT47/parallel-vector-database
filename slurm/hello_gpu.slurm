#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:05:00
#SBATCH -J hello_gpu
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH --gres=gpu:1 -c 1

module load nvidia/cuda/13.0.0

cd $HOME/parallel-vector-database

mkdir -p build
cd build
cmake ..
make hello_gpu

./hello_gpu