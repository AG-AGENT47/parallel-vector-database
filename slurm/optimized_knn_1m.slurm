#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:15:00
#SBATCH -J optimized_knn_1m
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH --gres=gpu:1 -c 1

module load nvidia/cuda/13.0

cd $HOME/parallel-vector-database

mkdir -p build benchmarks/results
cd build
cmake ..
make optimized_knn

cd ..

./build/optimized_knn 1000000 128 10 1000
