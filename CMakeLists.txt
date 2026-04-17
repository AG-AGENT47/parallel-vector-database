cmake_minimum_required(VERSION 3.18)
project(ParallelVectorDB LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# RTX 4000 Ada = compute capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

# ─── Target 1: Hello GPU (infrastructure check) ───
add_executable(hello_gpu
    src/gpu/hello_gpu.cu
)

# ─── Target 2: CPU Baseline (uncomment on Day 3) ───
# add_executable(baseline_knn
#     src/cpu/baseline_knn.cpp
# )