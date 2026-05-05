#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:10:00
#SBATCH -J optimized_knn
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH -c 1
#SBATCH --gres=gpu:1

module load nvidia/cuda/13.0.0

cd $HOME/parallel-vector-database/FinalProject

mkdir -p build
cd build
cmake ..
make optimized_knn

cd $HOME/parallel-vector-database/FinalProject
mkdir -p benchmarks/results

./build/optimized_knn 1000000 128 10 100
