#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:10:00
#SBATCH -J baseline_knn
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH -c 1

# No GPU needed for the CPU serial baseline.
# Stage 2+ will add: --gres=gpu:1

module load gcc/11.3.0

cd /home/hagarwal23/parallel-vector-database

mkdir -p build
cd build
cmake ..
make baseline_knn

cd /home/hagarwal23/parallel-vector-database
mkdir -p benchmarks/results

# Run: 100K vectors, 128-dim (SIFT1M scale), k=10, 1000 queries
./build/baseline_knn 100000 128 10 1000
