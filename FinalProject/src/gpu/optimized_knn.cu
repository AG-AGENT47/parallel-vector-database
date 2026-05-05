#include "common/utils.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <cstdlib>
#include <sstream>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " — " << cudaGetErrorString(err) << "\n";             \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// Compile-time constants — enables #pragma unroll and __shared__ sizing.
#define TILE_SIZE  32   // DB vectors loaded per tile into shared memory
#define BLOCK_SIZE 32   // threads per block (one warp; no intra-warp divergence)
#define DIM        128  // vector dimensionality; must match runtime dim arg

// One thread per query. Threads in a block cooperatively load a tile of
// TILE_SIZE DB vectors into shared memory, then each thread computes distances
// against that tile. This gives TILE_SIZE-fold reuse of each DB vector load
// and replaces the uncoalesced per-thread global reads of the naive kernel.
__global__ void knn_tiled_kernel(
    const float* __restrict__ db,
    const float* __restrict__ queries,
    int* __restrict__ results,
    int N, int Q, int k)
{
    int q = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Load this thread's query vector into registers (accessed N times total).
    float qvec[DIM];
    if (q < Q) {
        const float* qptr = queries + (long long)q * DIM;
        #pragma unroll
        for (int d = 0; d < DIM; d++) qvec[d] = qptr[d];
    }

    // Max-heap of size k: top_dist[0] is always the current maximum distance.
    float top_dist[256];
    int   top_idx[256];
    int   heap_size = 0;

    // Shared memory tile: TILE_SIZE DB vectors, each DIM floats.
    // 32 * 128 * 4 = 16 KB per block — well within the 49 KB SM limit.
    __shared__ float s_db[TILE_SIZE][DIM];

    for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
        int actual_tile = min(TILE_SIZE, N - tile_start);

        // ── Cooperative coalesced load ────────────────────────────────────────
        // Consecutive threads read consecutive addresses → 128-byte coalesced
        // transactions. Each thread handles TILE_SIZE*DIM/BLOCK_SIZE = 128
        // elements per tile (for default params).
        for (int idx = threadIdx.x; idx < TILE_SIZE * DIM; idx += BLOCK_SIZE) {
            int t = idx / DIM;
            int d = idx % DIM;
            if (t < actual_tile)
                s_db[t][d] = db[(long long)(tile_start + t) * DIM + d];
        }
        // All threads must reach this barrier before any thread reads s_db.
        __syncthreads();

        // ── Distance computation against shared-memory tile ───────────────────
        if (q < Q) {
            for (int t = 0; t < actual_tile; t++) {
                float dist = 0.f;
                #pragma unroll
                for (int d = 0; d < DIM; d++) {
                    float diff = qvec[d] - s_db[t][d];
                    dist += diff * diff;
                }

                int db_idx = tile_start + t;

                if (heap_size < k) {
                    top_dist[heap_size] = dist;
                    top_idx[heap_size]  = db_idx;
                    heap_size++;

                    if (heap_size == k) {
                        for (int h = k / 2 - 1; h >= 0; h--) {
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
                    top_dist[0] = dist;
                    top_idx[0]  = db_idx;
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
        }
        // Guard: prevent threads from overwriting s_db on the next tile load
        // before all threads finish reading it for distance computation.
        __syncthreads();
    }

    if (q < Q) {
        int base = q * k;
        for (int j = 0; j < k; j++)
            results[base + j] = top_idx[j];
    }
}

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

    if (dim != DIM) {
        std::cerr << "Error: compiled for DIM=" << DIM
                  << " but got dim=" << dim << "\n";
        return 1;
    }
    if (k > N || k > 256) {
        std::cerr << "Error: k must be <= n_vectors and <= 256\n";
        return 1;
    }

    // ── Data generation (identical to Stage 1 & 2 for reproducibility) ────────
    int actual_N, actual_dim, actual_Q, qd;
    std::vector<float> db      = load_fvecs("data/sift/sift_base.fvecs",
                                            actual_N, actual_dim, N);
    std::vector<float> queries = load_fvecs("data/sift/sift_query.fvecs",
                                            actual_Q, qd, Q);

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

    // ── Kernel launch ─────────────────────────────────────────────────────────
    const int blocks = (Q + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaDeviceSynchronize());
    Timer t_kernel;
    t_kernel.start();

    knn_tiled_kernel<<<blocks, BLOCK_SIZE>>>(d_db, d_queries, d_results, N, Q, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double kernel_ms = t_kernel.stop_ms();

    // ── D2H transfer ──────────────────────────────────────────────────────────
    std::vector<int> flat_results((long long)Q * k);
    CUDA_CHECK(cudaMemcpy(flat_results.data(), d_results,
                          (long long)Q * k * sizeof(int), cudaMemcpyDeviceToHost));

    double total_ms = t_total.stop_ms();

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
    log_result("benchmarks/results/optimized_knn.csv",
               "gpu_optimized", N, dim, k, kernel_ms, recall);

    double qps     = Q / (kernel_ms / 1000.0);
    double naive_ms = 0.0;
    std::ifstream csv("benchmarks/results/naive_knn.csv");
    if (csv) {
        std::string line;
        while (std::getline(csv, line)) {
            // format: stage,N,dim,k,time_ms,recall
            std::istringstream ss(line);
            std::string stage; int n, d, k2; double t; float r;
            char comma;
            if (std::getline(ss, stage, ',')) {
                ss >> n >> comma >> d >> comma >> k2 >> comma >> t;
                naive_ms = t;  // takes the last matching run
            }
        }
    }
    double speedup = (naive_ms > 0) ? naive_ms / kernel_ms : 0.0;

    std::cout << "Stage      : gpu_optimized\n"
              << "N          : " << N          << "\n"
              << "dim        : " << dim        << "\n"
              << "k          : " << k          << "\n"
              << "queries    : " << Q          << "\n"
              << "kernel_ms  : " << kernel_ms  << "\n"
              << "total_ms   : " << total_ms   << "\n"
              << "QPS        : " << qps        << "\n"
              << "recall@k   : " << recall     << "\n"
              << "speedup    : " << speedup    << "x vs Stage 2 gpu_naive\n";

    // ── Cleanup ───────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_db));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_results));

    return 0;
}
