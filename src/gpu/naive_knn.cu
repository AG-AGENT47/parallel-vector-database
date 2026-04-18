#include "common/utils.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " — " << cudaGetErrorString(err) << "\n";             \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

__device__ float l2_sq(const float* a, const float* b, int dim) {
    float d = 0.f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

// One thread per query. Each thread scans all N db vectors and maintains a
// max-heap of size k to find the k nearest neighbors.
// top_dist[0] is always the largest distance in the current top-k set.
__global__ void knn_kernel(
    const float* __restrict__ db,
    const float* __restrict__ queries,
    int* __restrict__ results,
    int N, int Q, int dim, int k)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Q) return;

    const float* qvec = queries + (long long)q * dim;

    // Stack-allocated top-k arrays (k <= 256 expected; k=10 in practice).
    float top_dist[256];
    int   top_idx[256];
    int   heap_size = 0;

    for (int i = 0; i < N; i++) {
        float dist = l2_sq(qvec, db + (long long)i * dim, dim);

        if (heap_size < k) {
            // Fill phase: just insert.
            top_dist[heap_size] = dist;
            top_idx[heap_size]  = i;
            heap_size++;

            // Once full, heapify so top_dist[0] = max.
            if (heap_size == k) {
                for (int h = k / 2 - 1; h >= 0; h--) {
                    // Sift-down h.
                    int root = h;
                    while (true) {
                        int largest = root;
                        int l = 2 * root + 1, r = 2 * root + 2;
                        if (l < k && top_dist[l] > top_dist[largest]) largest = l;
                        if (r < k && top_dist[r] > top_dist[largest]) largest = r;
                        if (largest == root) break;
                        float td = top_dist[root]; top_dist[root] = top_dist[largest]; top_dist[largest] = td;
                        int   ti = top_idx[root];  top_idx[root]  = top_idx[largest];  top_idx[largest]  = ti;
                        root = largest;
                    }
                }
            }
        } else if (dist < top_dist[0]) {
            // Replace root (current max) and sift down.
            top_dist[0] = dist;
            top_idx[0]  = i;
            int root = 0;
            while (true) {
                int largest = root;
                int l = 2 * root + 1, r = 2 * root + 2;
                if (l < k && top_dist[l] > top_dist[largest]) largest = l;
                if (r < k && top_dist[r] > top_dist[largest]) largest = r;
                if (largest == root) break;
                float td = top_dist[root]; top_dist[root] = top_dist[largest]; top_dist[largest] = td;
                int   ti = top_idx[root];  top_idx[root]  = top_idx[largest];  top_idx[largest]  = ti;
                root = largest;
            }
        }
    }

    // Write results (order within top-k doesn't matter for recall@k).
    int base = q * k;
    for (int j = 0; j < k; j++)
        results[base + j] = top_idx[j];
}

// Load ground truth written by Stage 1: Q*k int32 values, flat row-major.
bool load_ground_truth(const std::string& path,
                       std::vector<std::vector<int>>& gt,
                       int Q, int k)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    for (int q = 0; q < Q; q++) {
        gt[q].resize(k);
        f.read(reinterpret_cast<char*>(gt[q].data()), k * sizeof(int));
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <n_vectors> <dim> <k> <n_queries>\n";
        return 1;
    }

    const int N   = std::atoi(argv[1]);
    const int dim = std::atoi(argv[2]);
    const int k   = std::atoi(argv[3]);
    const int Q   = std::atoi(argv[4]);

    if (k > N || k > 256) {
        std::cerr << "Error: k must be <= n_vectors and <= 256\n";
        return 1;
    }

    // ── Data generation (identical to Stage 1 for reproducibility) ────────────
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    std::vector<float> db((long long)N * dim);
    std::vector<float> queries((long long)Q * dim);
    for (auto& v : db)      v = dist(gen);
    for (auto& v : queries) v = dist(gen);

    // ── Device memory allocation + H2D transfer ───────────────────────────────
    float* d_db      = nullptr;
    float* d_queries = nullptr;
    int*   d_results = nullptr;

    CUDA_CHECK(cudaMalloc(&d_db,      (long long)N * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_queries, (long long)Q * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_results, (long long)Q * k   * sizeof(int)));

    Timer t_total;
    t_total.start();

    CUDA_CHECK(cudaMemcpy(d_db,      db.data(),      (long long)N * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, queries.data(), (long long)Q * dim * sizeof(float), cudaMemcpyHostToDevice));

    // ── Kernel launch — timed separately ──────────────────────────────────────
    const int threads = 256;
    const int blocks  = (Q + threads - 1) / threads;

    CUDA_CHECK(cudaDeviceSynchronize());
    Timer t_kernel;
    t_kernel.start();

    knn_kernel<<<blocks, threads>>>(d_db, d_queries, d_results, N, Q, dim, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double kernel_ms = t_kernel.stop_ms();

    // ── D2H transfer ──────────────────────────────────────────────────────────
    std::vector<int> flat_results((long long)Q * k);
    CUDA_CHECK(cudaMemcpy(flat_results.data(), d_results,
                          (long long)Q * k * sizeof(int), cudaMemcpyDeviceToHost));

    double total_ms = t_total.stop_ms();

    // Reshape flat results to vector<vector<int>> for compute_recall.
    std::vector<std::vector<int>> results(Q, std::vector<int>(k));
    for (int q = 0; q < Q; q++)
        for (int j = 0; j < k; j++)
            results[q][j] = flat_results[q * k + j];

    // ── Load ground truth and compute recall ──────────────────────────────────
    std::vector<std::vector<int>> ground_truth(Q, std::vector<int>(k));
    float recall = 0.f;
    if (load_ground_truth("benchmarks/results/ground_truth.bin", ground_truth, Q, k)) {
        recall = compute_recall(ground_truth, results, k);
    } else {
        std::cerr << "Warning: could not load ground_truth.bin — recall set to 0\n";
    }

    // ── Log and print ─────────────────────────────────────────────────────────
    log_result("benchmarks/results/naive_knn.csv",
               "gpu_naive", N, dim, k, kernel_ms, recall);

    double qps = Q / (kernel_ms / 1000.0);
    std::cout << "Stage      : gpu_naive\n"
              << "N          : " << N          << "\n"
              << "dim        : " << dim        << "\n"
              << "k          : " << k          << "\n"
              << "queries    : " << Q          << "\n"
              << "kernel_ms  : " << kernel_ms  << "\n"
              << "total_ms   : " << total_ms   << "\n"
              << "QPS        : " << qps        << "\n"
              << "recall@k   : " << recall     << "\n";

    // ── Cleanup ───────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_db));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_results));

    return 0;
}
