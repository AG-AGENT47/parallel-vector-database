#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:45:00
#SBATCH -J ivf_pq_basic
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH --gres=gpu:1 -c 1

module load nvidia/cuda/13.0

cd $HOME/parallel-vector-database

mkdir -p build benchmarks/results
cd build
cmake ..
make ivf_pq_basic

cd ..

# Full 1M run — use SIFT official 10K queries
./build/ivf_pq_basic 1000000 128 10 100
