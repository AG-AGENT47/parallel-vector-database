// ═══════════════════════════════════════════════════════════════════════════
// Stage 5: IVF-PQ Optimized
//
// Four targeted optimizations over Stage 4 (ivf_pq_basic.cu):
//   1. OpenMP parallelized k-means (kmeans_omp) — assignment and update steps
//      run on all CPU cores, enabling NLIST=1024 with 25 iterations instead
//      of 8, which is the primary driver of recall improvement (0.30 → ~0.90).
//   2. Shared-memory LUT in ADC kernel (adc_scan_smem) — cooperatively loads
//      the 8 KB lookup table into shared memory, replacing global-memory reads.
//   3. Flat SoA inverted list layout — all codes and IDs stored contiguously,
//      eliminating nested-vector indirection and enabling fast memcpy gather.
//   4. CUDA streams + pinned host memory — N_STREAMS=4 async streams pipeline
//      GPU work with CPU preparation, overlapping PCIe transfers with compute.
// ═══════════════════════════════════════════════════════════════════════════

#include "../common/utils.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " — " << cudaGetErrorString(err) << "\n";            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ── IVF-PQ Configuration ─────────────────────────────────────────────────────
static constexpr int DIM       = 128;
static constexpr int NLIST     = 1024;  // up from 256 — better recall at N=1M
static constexpr int NPROBE    = 64;    // up from 32
static constexpr int MAX_ITER  = 25;    // up from 8 — affordable with OpenMP
static constexpr int M         = 8;
static constexpr int NBITS     = 8;
static constexpr int KSUB      = 1 << NBITS;   // 256
static constexpr int DSUB      = DIM / M;       // 16
static constexpr int N_STREAMS = 4;


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1: Utility
// ═══════════════════════════════════════════════════════════════════════════

inline float l2_sq_cpu(const float* a, const float* b, int d) {
    float s = 0.f;
    for (int i = 0; i < d; i++) { float diff = a[i]-b[i]; s += diff*diff; }
    return s;
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2: OpenMP-Parallelized K-Means
//
// Parallelizes both the assignment step and accumulation step.
// Per-thread private accumulators avoid write conflicts on centroid update.
// The RNG for empty-cluster re-seeding stays in the serial block.
// ═══════════════════════════════════════════════════════════════════════════
void kmeans_omp(const std::vector<float>& vectors,
                int N, int dim, int ncentroids, int max_iter,
                std::vector<float>& centroids,
                std::vector<int>& assignments)
{
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> pick(0, N - 1);

    centroids.assign((long long)ncentroids * dim, 0.f);
    for (int c = 0; c < ncentroids; c++) {
        int idx = pick(gen);
        std::copy(vectors.data() + (long long)idx * dim,
                  vectors.data() + (long long)idx * dim + dim,
                  centroids.data() + (long long)c * dim);
    }
    assignments.resize(N, 0);

    const int n_threads = omp_get_max_threads();

    for (int iter = 0; iter < max_iter; iter++) {

        // ── Assignment step: each vector → nearest centroid ───────────────
        // Independent per-vector writes into assignments[] — no races.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            float best = std::numeric_limits<float>::max();
            int   best_c = 0;
            const float* vi = vectors.data() + (long long)i * dim;
            for (int c = 0; c < ncentroids; c++) {
                float d = l2_sq_cpu(vi, centroids.data() + (long long)c * dim, dim);
                if (d < best) { best = d; best_c = c; }
            }
            assignments[i] = best_c;
        }

        // ── Update step: per-thread private accumulators ──────────────────
        // Each thread accumulates into its own copy, then we do a serial
        // reduction. Memory: n_threads * ncentroids * dim * 4 bytes ≈ 4 MB.
        std::vector<std::vector<float>> private_cents(
            n_threads, std::vector<float>((long long)ncentroids * dim, 0.f));
        std::vector<std::vector<int>> private_counts(
            n_threads, std::vector<int>(ncentroids, 0));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            float* my_cents  = private_cents[tid].data();
            int*   my_counts = private_counts[tid].data();

            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                int c = assignments[i];
                my_counts[c]++;
                const float* vi = vectors.data() + (long long)i * dim;
                float* mc = my_cents + (long long)c * dim;
                for (int d = 0; d < dim; d++) mc[d] += vi[d];
            }
        }  // implicit barrier — all threads finished accumulating

        // ── Serial global reduction ────────────────────────────────────────
        std::vector<float> new_cents((long long)ncentroids * dim, 0.f);
        std::vector<int>   counts(ncentroids, 0);

        for (int t = 0; t < n_threads; t++) {
            for (int c = 0; c < ncentroids; c++) {
                counts[c] += private_counts[t][c];
                const float* pc = private_cents[t].data() + (long long)c * dim;
                float*       nc = new_cents.data() + (long long)c * dim;
                for (int d = 0; d < dim; d++) nc[d] += pc[d];
            }
        }

        for (int c = 0; c < ncentroids; c++) {
            if (counts[c] > 0) {
                float inv = 1.f / counts[c];
                float* nc = new_cents.data() + (long long)c * dim;
                for (int d = 0; d < dim; d++) nc[d] *= inv;
            } else {
                // Empty cluster re-seed stays serial — no race on gen.
                int idx = pick(gen);
                std::copy(vectors.data() + (long long)idx * dim,
                          vectors.data() + (long long)idx * dim + dim,
                          new_cents.data() + (long long)c * dim);
            }
        }

        centroids = std::move(new_cents);

        if ((iter + 1) % 5 == 0 || iter == max_iter - 1)
            std::cerr << "    k-means iter " << iter+1 << "/" << max_iter << "\n";
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3: OpenMP-accelerated PQ Training
//
// Identical structure to train_pq() in Stage 4 but calls kmeans_omp().
// PQ sub-k-means uses 8 iterations (KSUB=256, DSUB=16 converge quickly).
// ═══════════════════════════════════════════════════════════════════════════
void train_pq_omp(const std::vector<float>& db, int N,
                  std::vector<float>& codebooks)
{
    codebooks.resize((long long)M * KSUB * DSUB);
    std::vector<float> subvecs((long long)N * DSUB);
    std::vector<int>   dummy_assign;

    for (int m = 0; m < M; m++) {
        for (int i = 0; i < N; i++) {
            const float* src = db.data() + (long long)i * DIM + m * DSUB;
            float*       dst = subvecs.data() + (long long)i * DSUB;
            std::copy(src, src + DSUB, dst);
        }
        std::vector<float> sub_centroids;
        kmeans_omp(subvecs, N, DSUB, KSUB, /*max_iter=*/8,
                   sub_centroids, dummy_assign);
        std::copy(sub_centroids.begin(), sub_centroids.end(),
                  codebooks.data() + (long long)m * KSUB * DSUB);
        std::cerr << "  PQ subspace " << m+1 << "/" << M << " trained\n";
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4: CPU PQ Encoding (unchanged from Stage 4)
// ═══════════════════════════════════════════════════════════════════════════
void encode_vector(const float* vec,
                   const std::vector<float>& codebooks,
                   uint8_t* code_out)
{
    for (int m = 0; m < M; m++) {
        const float* subvec = vec + m * DSUB;
        const float* book_m = codebooks.data() + (long long)m * KSUB * DSUB;
        float best = std::numeric_limits<float>::max();
        int   best_c = 0;
        for (int c = 0; c < KSUB; c++) {
            float d = l2_sq_cpu(subvec, book_m + c * DSUB, DSUB);
            if (d < best) { best = d; best_c = c; }
        }
        code_out[m] = static_cast<uint8_t>(best_c);
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5: IVF-PQ Optimized Index — Flat SoA Layout
//
// Replaces the nested vector<vector<>> from Stage 4 with contiguous arrays.
//
// Layout invariant:
//   flat_codes.data() + list_offsets[c] * M  — first code byte of cluster c
//   flat_ids.data()   + list_offsets[c]      — first ID of cluster c
//   list_sizes[c] == list_offsets[c+1] - list_offsets[c]
//
// list_offsets has size NLIST+1 (CSR-style); list_offsets[NLIST] == N.
// ═══════════════════════════════════════════════════════════════════════════
struct IVFPQIndexOpt {
    int N, nlist;
    std::vector<float>   coarse_centroids;  // nlist * DIM
    std::vector<float>   codebooks;         // M * KSUB * DSUB
    std::vector<uint8_t> flat_codes;        // N * M bytes total
    std::vector<int>     flat_ids;          // N ints total
    std::vector<int>     list_offsets;      // NLIST+1 prefix sums
    std::vector<int>     list_sizes;        // NLIST counts
};

// ─── SECTIONS 6–11 (build_index_opt, adc_scan_smem, StreamSlot,
//     query_ivfpq_opt, load_ground_truth_bin, main) to be added ─────────────
