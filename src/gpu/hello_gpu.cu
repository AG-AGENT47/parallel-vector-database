#include <iostream>
#include <cuda_runtime.h>

// This is a GPU "kernel" — a function that runs on the GPU
// __global__ means: "called from CPU, runs on GPU"
__global__ void helloFromGPU() {
    // threadIdx.x = which thread am I? (0, 1, 2, ...)
    // blockIdx.x  = which block am I in?
    printf("Hello from GPU! Block %d, Thread %d\n",
           blockIdx.x, threadIdx.x);
}

int main() {
    // Print CPU info
    std::cout << "Hello from CPU!" << std::endl;

    // Check what GPU is available
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " GPU(s)" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU 0: " << prop.name << std::endl;
    std::cout << "Compute capability: "
              << prop.major << "." << prop.minor << std::endl;

    // Launch the kernel: 2 blocks, 4 threads each = 8 total threads
    // Syntax: functionName<<<blocks, threads_per_block>>>(args)
    helloFromGPU<<<2, 4>>>();

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    std::cout << "Done!" << std::endl;
    return 0;
}