#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:10:00
#SBATCH -J naive_knn
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH -c 1
#SBATCH --gres=gpu:1

module load nvidia/cuda/13.0.0

cd $HOME/parallel-vector-database

mkdir -p build
cd build
cmake ..
make naive_knn

cd $HOME/parallel-vector-database
mkdir -p benchmarks/results

./build/naive_knn 1000000 128 10 100
