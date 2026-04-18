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

# ─── Target 2: CPU Baseline ───
add_executable(baseline_knn
    src/cpu/baseline_knn.cpp
)
target_include_directories(baseline_knn PRIVATE src)

# ─── Target 3: GPU Naive k-NN ───
add_executable(naive_knn
    src/gpu/naive_knn.cu
)
target_include_directories(naive_knn PRIVATE src)

# ─── Target 4: GPU Optimized k-NN (Stage 3) ───
add_executable(optimized_knn
    src/gpu/optimized_knn.cu
)
target_include_directories(optimized_knn PRIVATE src)