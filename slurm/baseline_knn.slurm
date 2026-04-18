#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:10:00
#SBATCH -J baseline_knn
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH -c 1

# No GPU needed for the CPU serial baseline.
# Stage 2+ will add: --gres=gpu:1

module load nvidia/cuda/13.0.0

cd $HOME/parallel-vector-database

mkdir -p build
cd build
cmake ..
make baseline_knn

cd $HOME/parallel-vector-database
mkdir -p benchmarks/results

# Run: 100K vectors, 128-dim (SIFT1M scale), k=10, 1000 queries
./build/baseline_knn 100000 128 10 1000
