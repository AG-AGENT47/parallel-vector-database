#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J ivf_pq_basic
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH --gres=gpu:1 -c 1

module load nvidia/cuda/13.0

cd /home/hagarwal23/parallel-vector-database

mkdir -p build benchmarks/results
cd build
cmake ..
make ivf_pq_basic

cd ..

# Run with 100K vectors, 128 dims, k=10, 1000 queries
./build/ivf_pq_basic 100000 128 10 1000
