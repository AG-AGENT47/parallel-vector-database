#!/usr/bin/env zsh
                                                                                                                                                                                                            
#SBATCH -p instruction                                                                                                                                                                                      
#SBATCH -t 0-00:45:00
#SBATCH -J ivf_pq_optimized                                                                                                                                                                                 
#SBATCH -o output-%j.out -e output-%j.err                                                                                                                                                                   
#SBATCH --gres=gpu:1 -c 8
                                                                                                                                                                                                            
module load nvidia/cuda/13.0.0
                                                                                                                                                                                                            
cd $HOME/parallel-vector-database/FinalProject

export OMP_NUM_THREADS=8                                                                                                                                                                                    
                
mkdir -p build benchmarks/results                                                                                                                                                                           
cd build
cmake ..                                                                                                                                                                                                    
make ivf_pq_optimized

cd ..                                                                                                                                                                                                       

./build/ivf_pq_optimized 1000000 128 10 100
                
