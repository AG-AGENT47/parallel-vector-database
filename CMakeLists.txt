cmake_minimum_required(VERSION 3.18)
project(ParallelVectorDB LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)

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

# ─── Target 4: GPU Optimized k-NN ───
add_executable(optimized_knn
    src/gpu/optimized_knn.cu
)
target_include_directories(optimized_knn PRIVATE src)

# ─── Target 5: IVF-PQ Basic ───
add_executable(ivf_pq_basic
    src/gpu/ivf_pq_basic.cu
)
target_include_directories(ivf_pq_basic PRIVATE src)

# ─── Target 6: IVF-PQ Optimized ───
add_executable(ivf_pq_optimized
    src/gpu/ivf_pq_optimized.cu
)
target_compile_options(ivf_pq_optimized PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(ivf_pq_optimized PRIVATE ${OpenMP_CXX_LIBRARIES})
target_include_directories(ivf_pq_optimized PRIVATE src)