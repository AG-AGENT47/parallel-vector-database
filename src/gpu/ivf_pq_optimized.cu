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
static constexpr int NLIST     = 256;  // up from 256 — better recall at N=1M
static constexpr int NPROBE    = 32;   // increased: 128/1024 = 12.5% probe rate, matching Stage 4
static constexpr int MAX_ITER  = 25;    // up from 8 — affordable with OpenMP
static constexpr int M         = 32;
static constexpr int NBITS     = 8;
static constexpr int KSUB      = 1 << NBITS;   // 256
static constexpr int DSUB      = DIM / M;       // 16
static constexpr int N_STREAMS = 8;


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

                                                                                                                                                                                                             
// ═══════════════════════════════════════════════════════════════════════════                                                                                                                              
// SECTION 6: Index Building (Optimized)                                                                                                                                                                    
// ═══════════════════════════════════════════════════════════════════════════                                                                                                                              
IVFPQIndexOpt build_index_opt(const std::vector<float>& db,                                                                                                                                                 
                                const std::vector<float>& db_train,                                                                                                                                          
                                int N, int N_train)                                                                                                                                                          
{                                                                                                                                                                                                           
    IVFPQIndexOpt idx;                                                                                                                                                                                      
    idx.N     = N;                                                                                                                                                                                          
    idx.nlist = NLIST;                                                                                                                                                                                      
                                                                                                                                                                                                            
    // ── Phase 1: Coarse quantizer (OpenMP k-means) ────────────────────────                                                                                                                               
    std::cerr << "[1/3] Training coarse quantizer ("
            << NLIST << " clusters, " << MAX_ITER << " iters, "                                                                                                                                           
            << omp_get_max_threads() << " OMP threads)...\n";                                                                                                                                             
    std::vector<int> coarse_assign;                                                                                                                                                                         
    kmeans_omp(db_train, N_train, DIM, NLIST, MAX_ITER,                                                                                                                                                     
                idx.coarse_centroids, coarse_assign);                                                                                                                                                      
    std::cerr << "      Done.\n";                                                                                                                                                                           
                                                                                                                                                                                                        
    // ── Phase 2: PQ codebook training (OpenMP) ────────────────────────────                                                                                                                               
    std::cerr << "[2/3] Training PQ codebooks (M=" << M                                                                                                                                                   
            << ", KSUB=" << KSUB << ")...\n";                                                                                                                                                             
    train_pq_omp(db_train, N_train, idx.codebooks);                                                                                                                                                       
    std::cerr << "      Done.\n";                                                                                                                                                                           
                                                                                                                                                                                                            
    // ── Phase 3: Build flat SoA inverted lists ────────────────────────────
    std::cerr << "[3/3] Assigning " << N << " vectors and building flat lists...\n";                                                                                                                        
                                                                                                                                                                                                            
    std::vector<int> full_assign(N);
    #pragma omp parallel for schedule(static)                                                                                                                                                               
    for (int i = 0; i < N; i++) {                                                                                                                                                                           
        float best = std::numeric_limits<float>::max();
        int   best_c = 0;                                                                                                                                                                                   
        const float* vi = db.data() + (long long)i * DIM;
        for (int c = 0; c < NLIST; c++) {                                                                                                                                                                   
            float d = l2_sq_cpu(vi, idx.coarse_centroids.data() + (long long)c * DIM, DIM);
            if (d < best) { best = d; best_c = c; }                                                                                                                                                         
        }       
        full_assign[i] = best_c;                                                                                                                                                                            
    }                                                                                                                                                                                                       

    idx.list_sizes.assign(NLIST, 0);                                                                                                                                                                        
    for (int i = 0; i < N; i++) idx.list_sizes[full_assign[i]]++;
                                                                                                                                                                                                            
    idx.list_offsets.resize(NLIST + 1, 0);
    for (int c = 0; c < NLIST; c++)                                                                                                                                                                         
        idx.list_offsets[c + 1] = idx.list_offsets[c] + idx.list_sizes[c];                                                                                                                                  
    assert(idx.list_offsets[NLIST] == N);
                                                                                                                                                                                                            
    idx.flat_codes.resize((long long)N * M);
    idx.flat_ids.resize(N);                                                                                                                                                                                 
                
    std::vector<int> cursor(idx.list_offsets.begin(),                                                                                                                                                       
                            idx.list_offsets.begin() + NLIST);
                                                                                                                                                                                                            
    for (int i = 0; i < N; i++) {
        int c   = full_assign[i];                                                                                                                                                                           
        int pos = cursor[c]++;
        idx.flat_ids[pos] = i;                                                                                                                                                                              
        encode_vector(db.data() + (long long)i * DIM, idx.codebooks,
                    idx.flat_codes.data() + (long long)pos * M);                                                                                                                                          
    }           
    std::cerr << "      Done.\n";                                                                                                                                                                           
                
    int min_sz = N, max_sz = 0;                                                                                                                                                                             
    for (int c = 0; c < NLIST; c++) {
        min_sz = std::min(min_sz, idx.list_sizes[c]);                                                                                                                                                       
        max_sz = std::max(max_sz, idx.list_sizes[c]);                                                                                                                                                       
    }
    std::cerr << "      List sizes — min: " << min_sz                                                                                                                                                       
            << "  max: " << max_sz                                                                                                                                                                        
            << "  avg: " << N / NLIST << "\n";
                                                                                                                                                                                                            
    return idx;                                                                                                                                                                                             
}
                                                                                                                                                                                                            
                
// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7: GPU ADC Scan Kernel — Shared Memory LUT
//                                                                                                                                                                                                          
// Cooperatively loads the M*KSUB=2048 float LUT (8 KB) into shared memory.
// Bounds check MUST come after __syncthreads() so no thread exits early and                                                                                                                                
// leaves shared memory partially uninitialized.                                                                                                                                                            
// ═══════════════════════════════════════════════════════════════════════════                                                                                                                              
__global__ void adc_scan_smem(                                                                                                                                                                              
    const float*   __restrict__ lut,                                                                                                                                                                        
    const uint8_t* __restrict__ codes,
    float*         __restrict__ dists_out,                                                                                                                                                                  
    int n_cands)                                                                                                                                                                                            
{                                                                                                                                                                                                           
    __shared__ float s_lut[M * KSUB];   // 2048 floats = 8 KB per block                                                                                                                                     
                                                                                                                                                                                                            
    for (int idx = threadIdx.x; idx < M * KSUB; idx += blockDim.x)                                                                                                                                          
        s_lut[idx] = lut[idx];                                                                                                                                                                              
    __syncthreads();   // all threads finish loading before any reads s_lut                                                                                                                                 
                                                                                                                                                                                                            
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cands) return;   // bounds check AFTER syncthreads                                                                                                                                           
                                                                                                                                                                                                            
    float dist = 0.f;
    #pragma unroll                                                                                                                                                                                          
    for (int m = 0; m < M; m++) {
        uint8_t c = codes[i * M + m];                                                                                                                                                                       
        dist += s_lut[m * KSUB + c];
    }                                                                                                                                                                                                       
    dists_out[i] = dist;
}                                                                                                                                                                                                           

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7b: Batched Coarse Quantizer Kernel
//
// For a batch of B queries: compute distance from each query to all NLIST
// centroids, then find top NPROBE nearest clusters per query.
//
// Grid: B blocks, one per query.
// Block: NLIST threads (256 — one per centroid).
//
// Each block:
//   1. Each thread computes distance from its query to one centroid.
//   2. Distances stored in shared memory.
//   3. NPROBE-pass selection of smallest distances (each pass picks min, marks it).
//
// This avoids the CPU's O(NLIST) sort by trading correctness for simplicity:
// a partial selection of NPROBE elements from NLIST is O(NPROBE * NLIST)
// which at NPROBE=32, NLIST=256 = 8192 comparisons — fast in shared mem.
// ═══════════════════════════════════════════════════════════════════════════
__global__ void coarse_quantizer_kernel(
    const float* __restrict__ queries,           // B * DIM
    const float* __restrict__ centroids,         // NLIST * DIM
    int*         __restrict__ probe_orders_out,  // B * NPROBE
    int B)
{
    int query_idx = blockIdx.x;
    int tid       = threadIdx.x;
    if (query_idx >= B) return;

    __shared__ float s_dists[NLIST];
    __shared__ int   s_picked[NLIST];  // 0 = available, 1 = already picked

    const float* qvec = queries + (long long)query_idx * DIM;

    // Compute distance from this query to centroid `tid`
    if (tid < NLIST) {
        float d = 0.f;
        const float* cvec = centroids + (long long)tid * DIM;
        for (int i = 0; i < DIM; i++) {
            float diff = qvec[i] - cvec[i];
            d += diff * diff;
        }
        s_dists[tid]  = d;
        s_picked[tid] = 0;
    }
    __syncthreads();

    // NPROBE passes of "find argmin among unpicked"
    // Done by thread 0 only (selection is inherently serial here).
    if (tid == 0) {
        for (int p = 0; p < NPROBE; p++) {
            float best = 3.4e38f;
            int   best_c = 0;
            for (int c = 0; c < NLIST; c++) {
                if (s_picked[c] == 0 && s_dists[c] < best) {
                    best = s_dists[c];
                    best_c = c;
                }
            }
            s_picked[best_c] = 1;
            probe_orders_out[query_idx * NPROBE + p] = best_c;
        }
    }
}
// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7c: Batched LUT Construction Kernel
//
// For a batch of B queries: compute the M*KSUB lookup table per query.
// LUT[m][c] = ||query_subvec_m - codebook[m][c]||^2
//
// Grid: (B, M) — one block per (query, subspace) pair.
// Block: KSUB threads — one per codeword.
//
// Each thread computes one (query, subspace, codeword) distance entry.
// Codebooks are loaded into shared memory (DSUB*KSUB floats = 4KB per block).
// ═══════════════════════════════════════════════════════════════════════════
__global__ void build_lut_kernel(
    const float* __restrict__ queries,    // B * DIM
    const float* __restrict__ codebooks,  // M * KSUB * DSUB
    float*       __restrict__ luts_out)   // B * M * KSUB
{
    int query_idx = blockIdx.x;
    int m         = blockIdx.y;
    int c         = threadIdx.x;
    if (c >= KSUB) return;

    const float* qsub   = queries + (long long)query_idx * DIM + m * DSUB;
    const float* cw     = codebooks + (long long)m * KSUB * DSUB + (long long)c * DSUB;

    float d = 0.f;
    #pragma unroll
    for (int i = 0; i < DSUB; i++) {
        float diff = qsub[i] - cw[i];
        d += diff * diff;
    }

    luts_out[(long long)query_idx * M * KSUB + m * KSUB + c] = d;
}
                                                                                                                                                                                                            
// ═══════════════════════════════════════════════════════════════════════════
// SECTION 8: CUDA Stream Slot
// ═══════════════════════════════════════════════════════════════════════════                                                                                                                              
struct StreamSlot {
    cudaStream_t     stream       = nullptr;                                                                                                                                                                
    float*           d_lut        = nullptr;
    uint8_t*         d_codes      = nullptr;                                                                                                                                                                
    float*           d_dists      = nullptr;
    float*           h_lut        = nullptr;                                                                                                                                                                
    uint8_t*         h_codes      = nullptr;                                                                                                                                                                
    float*           h_dists      = nullptr;
    int              pending_q    = -1;                                                                                                                                                                     
    int              pending_n    = 0;                                                                                                                                                                      
    std::vector<int> pending_ids;
};                                                                                                                                                                                                          
                
                                                                                                                                                                                                            
// ═══════════════════════════════════════════════════════════════════════════
// SECTION 9: Query Function (Optimized)                                                                                                                                                                    
// ═══════════════════════════════════════════════════════════════════════════
void query_ivfpq_opt(
    const IVFPQIndexOpt&      index,
    const std::vector<float>& queries,
    int Q, int k,
    std::vector<std::vector<int>>& results)
{
    results.assign(Q, std::vector<int>(k, -1));

    const int avg_list  = (index.N + NLIST - 1) / NLIST;
    const int MAX_CANDS = NPROBE * avg_list * 4;

    // One stream slot per OMP thread — each thread owns its slot exclusively.
    // No synchronization needed on slot access since threads never share slots.
    std::vector<StreamSlot> slots(N_STREAMS);
    for (int s = 0; s < N_STREAMS; s++) {
        CUDA_CHECK(cudaStreamCreate(&slots[s].stream));
        CUDA_CHECK(cudaMalloc(&slots[s].d_lut,
                              (long long)M * KSUB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&slots[s].d_codes,
                              (long long)MAX_CANDS * M * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&slots[s].d_dists,
                              (long long)MAX_CANDS * sizeof(float)));
        CUDA_CHECK(cudaHostAlloc(&slots[s].h_lut,
                                 (long long)M * KSUB * sizeof(float),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&slots[s].h_codes,
                                 (long long)MAX_CANDS * M * sizeof(uint8_t),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&slots[s].h_dists,
                                 (long long)MAX_CANDS * sizeof(float),
                                 cudaHostAllocDefault));
    }

    // Parallel query loop — each OMP thread handles a subset of queries.
    // Thread tid exclusively owns slots[tid], so no races on slot fields.
    // results[q] writes are safe since each q is processed by exactly one thread.
    #pragma omp parallel for schedule(dynamic, 1) num_threads(N_STREAMS)
    for (int q = 0; q < Q; q++) {
        int tid = omp_get_thread_num();
        StreamSlot& slot = slots[tid];

        // Wait for previous query on this stream to fully complete
        // before overwriting h_codes/h_lut pinned buffers
        CUDA_CHECK(cudaStreamSynchronize(slot.stream));

        const float* qvec = queries.data() + (long long)q * DIM;

        // ── Step 1: Coarse quantizer ──────────────────────────────────────
        std::vector<float> cdists(NLIST);
        for (int c = 0; c < NLIST; c++)
            cdists[c] = l2_sq_cpu(qvec,
                                  index.coarse_centroids.data() + (long long)c * DIM,
                                  DIM);
        std::vector<int> probe_order(NLIST);
        std::iota(probe_order.begin(), probe_order.end(), 0);
        std::partial_sort(probe_order.begin(), probe_order.begin() + NPROBE,
                          probe_order.end(),
                          [&cdists](int a, int b){ return cdists[a] < cdists[b]; });

        // ── Step 2: Build LUT ─────────────────────────────────────────────
        for (int m = 0; m < M; m++) {
            const float* qsub   = qvec + m * DSUB;
            const float* book_m = index.codebooks.data() + (long long)m * KSUB * DSUB;
            for (int c = 0; c < KSUB; c++)
                slot.h_lut[m * KSUB + c] = l2_sq_cpu(qsub, book_m + c * DSUB, DSUB);
        }

        // ── Step 3: Gather candidates ─────────────────────────────────────
        int n_cands = 0;
        std::vector<int> local_ids;
        for (int p = 0; p < NPROBE; p++) {
            int c  = probe_order[p];
            int sz = index.list_sizes[c];
            if (sz == 0) continue;
            if (n_cands + sz > MAX_CANDS) {
                std::cerr << "Warning: MAX_CANDS exceeded, truncating\n";
                break;
            }
            std::memcpy(slot.h_codes + (long long)n_cands * M,
                        index.flat_codes.data() + (long long)index.list_offsets[c] * M,
                        (long long)sz * M * sizeof(uint8_t));
            const int* ids_ptr = index.flat_ids.data() + index.list_offsets[c];
            local_ids.insert(local_ids.end(), ids_ptr, ids_ptr + sz);
            n_cands += sz;
        }

        if (n_cands == 0) continue;

        // ── Step 4: GPU ADC scan (async on this thread's stream) ──────────
        CUDA_CHECK(cudaMemcpyAsync(slot.d_lut, slot.h_lut,
                                   (long long)M * KSUB * sizeof(float),
                                   cudaMemcpyHostToDevice, slot.stream));
        CUDA_CHECK(cudaMemcpyAsync(slot.d_codes, slot.h_codes,
                                   (long long)n_cands * M * sizeof(uint8_t),
                                   cudaMemcpyHostToDevice, slot.stream));

        const int threads = 256;
        const int blocks  = (n_cands + threads - 1) / threads;
        adc_scan_smem<<<blocks, threads, 0, slot.stream>>>(
            slot.d_lut, slot.d_codes, slot.d_dists, n_cands);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(slot.h_dists, slot.d_dists,
                                   (long long)n_cands * sizeof(float),
                                   cudaMemcpyDeviceToHost, slot.stream));

        // ── Step 5: Synchronize and top-k (inline, no shared state) ──────
        CUDA_CHECK(cudaStreamSynchronize(slot.stream));

        int actual_k = std::min(k, n_cands);
        std::vector<int> idx_sort(n_cands);
        std::iota(idx_sort.begin(), idx_sort.end(), 0);
        std::partial_sort(idx_sort.begin(), idx_sort.begin() + actual_k,
                          idx_sort.end(),
                          [&slot](int a, int b){
                              return slot.h_dists[a] < slot.h_dists[b];
                          });
        for (int j = 0; j < actual_k; j++)
            results[q][j] = local_ids[idx_sort[j]];
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    for (int s = 0; s < N_STREAMS; s++) {
        CUDA_CHECK(cudaFree(slots[s].d_lut));
        CUDA_CHECK(cudaFree(slots[s].d_codes));
        CUDA_CHECK(cudaFree(slots[s].d_dists));
        CUDA_CHECK(cudaFreeHost(slots[s].h_lut));
        CUDA_CHECK(cudaFreeHost(slots[s].h_codes));
        CUDA_CHECK(cudaFreeHost(slots[s].h_dists));
        CUDA_CHECK(cudaStreamDestroy(slots[s].stream));
    }
}     

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 9b: Batched ADC Kernel
//
// Processes a batch of queries simultaneously. Each block handles one query
// from the batch. Within a block, threads cooperatively load that query's
// LUT into shared memory, then scan the candidate codes for that query.
//
// Layout:
//   batch_luts:    BATCH_SIZE * M * KSUB floats (one LUT per query)
//   batch_codes:   total_codes * M bytes (concatenated codes from all queries)
//   batch_offsets: BATCH_SIZE+1 ints (CSR offsets into batch_codes)
//   batch_dists:   total_codes floats (output distances per code per query)
// ═══════════════════════════════════════════════════════════════════════════
__global__ void adc_scan_batched(
    const float*   __restrict__ batch_luts,     // BATCH * M * KSUB
    const uint8_t* __restrict__ batch_codes,    // total_codes * M
    const int*     __restrict__ batch_offsets,  // BATCH+1
    float*         __restrict__ batch_dists)    // total_codes
{
    int query_idx = blockIdx.y;  // which query in the batch
    int code_idx  = blockIdx.x * blockDim.x + threadIdx.x;  // which code

    int start = batch_offsets[query_idx];
    int end   = batch_offsets[query_idx + 1];
    int n_cands_this_query = end - start;

    // Load this query's LUT into shared memory
    __shared__ float s_lut[M * KSUB];
    const float* my_lut = batch_luts + (long long)query_idx * M * KSUB;
    for (int idx = threadIdx.x; idx < M * KSUB; idx += blockDim.x)
        s_lut[idx] = my_lut[idx];
    __syncthreads();

    if (code_idx >= n_cands_this_query) return;

    int absolute_code = start + code_idx;
    float dist = 0.f;
    #pragma unroll
    for (int m = 0; m < M; m++) {
        uint8_t c = batch_codes[absolute_code * M + m];
        dist += s_lut[m * KSUB + c];
    }
    batch_dists[absolute_code] = dist;
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 9c: Batched Query Function
//
// Processes queries in batches of BATCH_SIZE. For each batch:
//   1. CPU (parallel): coarse quantizer + LUT construction + candidate gather
//      for all BATCH_SIZE queries simultaneously using OpenMP
//   2. GPU: single batched ADC kernel launch processes the entire batch
//   3. CPU (parallel): top-k selection per query
// ═══════════════════════════════════════════════════════════════════════════
void query_ivfpq_batched(
    const IVFPQIndexOpt&      index,
    const std::vector<float>& queries,
    int Q, int k,
    std::vector<std::vector<int>>& results)
{
    results.assign(Q, std::vector<int>(k, -1));

    constexpr int BATCH_SIZE = 64;
    const int avg_list  = (index.N + NLIST - 1) / NLIST;
    const int MAX_CANDS_PER_QUERY = NPROBE * avg_list * 4;
    const long long MAX_BATCH_CANDS = (long long)BATCH_SIZE * MAX_CANDS_PER_QUERY;

    // ── Allocate device buffers (reused across all batches) ───────────────
    float*   d_luts   = nullptr;
    uint8_t* d_codes  = nullptr;
    int*     d_offs   = nullptr;
    float*   d_dists  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_luts,
                          (long long)BATCH_SIZE * M * KSUB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_codes, MAX_BATCH_CANDS * M * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_offs,  (BATCH_SIZE + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dists, MAX_BATCH_CANDS * sizeof(float)));

    // Pinned host buffers
    float*   h_luts   = nullptr;
    uint8_t* h_codes  = nullptr;
    int*     h_offs   = nullptr;
    float*   h_dists  = nullptr;

    CUDA_CHECK(cudaHostAlloc(&h_luts,
                             (long long)BATCH_SIZE * M * KSUB * sizeof(float),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_codes, MAX_BATCH_CANDS * M * sizeof(uint8_t),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_offs,  (BATCH_SIZE + 1) * sizeof(int),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_dists, MAX_BATCH_CANDS * sizeof(float),
                             cudaHostAllocDefault));

    // Per-query metadata
    std::vector<std::vector<int>> batch_local_ids(BATCH_SIZE);
    std::vector<std::vector<int>> batch_probe_order(BATCH_SIZE);
    std::vector<int>              batch_n_cands(BATCH_SIZE, 0);

    // ── GPU buffers for coarse quantizer (allocated once, reused across batches) ──
    static float* d_q_batch      = nullptr;
    static float* d_centroids    = nullptr;
    static int*   d_probe_orders = nullptr;
    static float* d_codebooks    = nullptr;
    static bool   coarse_init    = false;
    if (!coarse_init) {
        CUDA_CHECK(cudaMalloc(&d_q_batch,
                              (long long)BATCH_SIZE * DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_centroids,
                              (long long)NLIST * DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_probe_orders,
                              (long long)BATCH_SIZE * NPROBE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_codebooks,
                              (long long)M * KSUB * DSUB * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_centroids, index.coarse_centroids.data(),
                              (long long)NLIST * DIM * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_codebooks, index.codebooks.data(),
                              (long long)M * KSUB * DSUB * sizeof(float),
                              cudaMemcpyHostToDevice));
        coarse_init = true;
    }

    for (int batch_start = 0; batch_start < Q; batch_start += BATCH_SIZE) {
        int batch_end = std::min(batch_start + BATCH_SIZE, Q);
        int batch_n   = batch_end - batch_start;

        for (int i = 0; i < BATCH_SIZE; i++) {
            batch_local_ids[i].clear();
            batch_probe_order[i].clear();
            batch_n_cands[i] = 0;
        }

        // ── GPU Coarse Quantizer for this batch ───────────────────────────
        CUDA_CHECK(cudaMemcpy(d_q_batch,
                              queries.data() + (long long)batch_start * DIM,
                              (long long)batch_n * DIM * sizeof(float),
                              cudaMemcpyHostToDevice));

        coarse_quantizer_kernel<<<batch_n, NLIST>>>(
            d_q_batch, d_centroids, d_probe_orders, batch_n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> h_probe_orders((size_t)batch_n * NPROBE);
        CUDA_CHECK(cudaMemcpy(h_probe_orders.data(), d_probe_orders,
                              (long long)batch_n * NPROBE * sizeof(int),
                              cudaMemcpyDeviceToHost));
                        
        // ── GPU LUT Construction for this batch ───────────────────────────
        // Launches into d_luts directly (no h_luts needed for this stage).
        dim3 lut_grid(batch_n, M);
        build_lut_kernel<<<lut_grid, KSUB>>>(d_q_batch, d_codebooks, d_luts);
        CUDA_CHECK(cudaGetLastError());
        // No sync needed — we'll sync before the ADC kernel uses d_luts

        // ── Phase 1 (parallel): LUT build + ID gather (using GPU's probe order) ──
        #pragma omp parallel for schedule(dynamic, 1) num_threads(N_STREAMS)
        for (int b = 0; b < batch_n; b++) {
            int q = batch_start + b;
            const float* qvec = queries.data() + (long long)q * DIM;

            // Use probe order from GPU coarse quantizer
            std::vector<int> probe_order(
                h_probe_orders.begin() + (size_t)b * NPROBE,
                h_probe_orders.begin() + (size_t)(b + 1) * NPROBE);



            // Compute candidate count and gather IDs
            int n_cands = 0;
            std::vector<int> local_ids;
            for (int p = 0; p < NPROBE; p++) {
                int c  = probe_order[p];
                int sz = index.list_sizes[c];
                if (sz == 0) continue;
                const int* ids_ptr = index.flat_ids.data() + index.list_offsets[c];
                local_ids.insert(local_ids.end(), ids_ptr, ids_ptr + sz);
                n_cands += sz;
            }

            batch_n_cands[b]     = n_cands;
            batch_local_ids[b]   = std::move(local_ids);
            batch_probe_order[b] = std::move(probe_order);
        }

        // ── Phase 2: prefix sum to get offsets ──────────────────────────
        h_offs[0] = 0;
        for (int b = 0; b < batch_n; b++)
            h_offs[b + 1] = h_offs[b] + batch_n_cands[b];
        for (int b = batch_n; b < BATCH_SIZE; b++)
            h_offs[b + 1] = h_offs[b];

        int total_cands = h_offs[batch_n];
        if (total_cands == 0) continue;

        // ── Phase 3 (parallel): copy codes using saved probe_order ──────
        // No re-computation — just memcpy from index into the right offset.
        #pragma omp parallel for schedule(dynamic, 1) num_threads(N_STREAMS)
        for (int b = 0; b < batch_n; b++) {
            int dst_offset = h_offs[b];
            int written    = 0;
            for (int p = 0; p < NPROBE; p++) {
                int c  = batch_probe_order[b][p];
                int sz = index.list_sizes[c];
                if (sz == 0) continue;
                std::memcpy(h_codes + (long long)(dst_offset + written) * M,
                            index.flat_codes.data() + (long long)index.list_offsets[c] * M,
                            (long long)sz * M * sizeof(uint8_t));
                written += sz;
            }
        }

        // ── Phase 4: GPU upload + batched kernel + download ─────────────

        CUDA_CHECK(cudaMemcpy(d_codes, h_codes,
                              (long long)total_cands * M * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offs, h_offs,
                              (BATCH_SIZE + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));

        int max_cands_q = 0;
        for (int b = 0; b < batch_n; b++)
            max_cands_q = std::max(max_cands_q, batch_n_cands[b]);

        const int threads = 256;
        const int blocks_x = (max_cands_q + threads - 1) / threads;
        dim3 grid(blocks_x, batch_n);
        adc_scan_batched<<<grid, threads>>>(d_luts, d_codes, d_offs, d_dists);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dists, d_dists,
                              (long long)total_cands * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // ── Phase 5 (parallel): top-k per query ─────────────────────────
        #pragma omp parallel for schedule(dynamic, 1) num_threads(N_STREAMS)
        for (int b = 0; b < batch_n; b++) {
            int q = batch_start + b;
            int n = batch_n_cands[b];
            if (n == 0) continue;

            int actual_k = std::min(k, n);
            std::vector<int> idx_sort(n);
            std::iota(idx_sort.begin(), idx_sort.end(), 0);

            float* my_dists = h_dists + h_offs[b];
            std::partial_sort(idx_sort.begin(), idx_sort.begin() + actual_k,
                              idx_sort.end(),
                              [my_dists](int a, int bb){
                                  return my_dists[a] < my_dists[bb];
                              });

            for (int j = 0; j < actual_k; j++)
                results[q][j] = batch_local_ids[b][idx_sort[j]];
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_luts));
    CUDA_CHECK(cudaFree(d_codes));
    CUDA_CHECK(cudaFree(d_offs));
    CUDA_CHECK(cudaFree(d_dists));
    CUDA_CHECK(cudaFreeHost(h_luts));
    CUDA_CHECK(cudaFreeHost(h_codes));
    CUDA_CHECK(cudaFreeHost(h_offs));
    CUDA_CHECK(cudaFreeHost(h_dists));
}
                                                                                                                                                                                                           
// ═══════════════════════════════════════════════════════════════════════════
// SECTION 10: Ground Truth Loader (Stage 1 binary format)                                                                                                                                                  
// ═══════════════════════════════════════════════════════════════════════════                                                                                                                              
bool load_ground_truth_bin(const std::string& path,
                            std::vector<std::vector<int>>& gt,                                                                                                                                               
                            int Q, int k)                                                                                                                                                                    
{
    std::ifstream f(path, std::ios::binary);                                                                                                                                                                
    if (!f) return false;                                                                                                                                                                                   
    gt.assign(Q, std::vector<int>(k));
    for (int q = 0; q < Q; q++)                                                                                                                                                                             
        f.read(reinterpret_cast<char*>(gt[q].data()), k * sizeof(int));                                                                                                                                     
    return true;
}                                                                                                                                                                                                           
                
                                                                                                                                                                                                            
// ═══════════════════════════════════════════════════════════════════════════
// SECTION 11: Main                                                                                                                                                                                         
// ═══════════════════════════════════════════════════════════════════════════
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
                                                                                                                                                                                                            
    std::cerr << "OMP threads available: " << omp_get_max_threads() << "\n";                                                                                                                                

    int actual_N, actual_dim, actual_Q, qd;                                                                                                                                                                 
    std::vector<float> db      = load_fvecs("data/sift/sift_base.fvecs",
                                            actual_N, actual_dim, N);                                                                                                                                       
    std::vector<float> queries = load_fvecs("data/sift/sift_query.fvecs",                                                                                                                                   
                                            actual_Q, qd, Q);                                                                                                                                               
                                                                                                                                                                                                            
    int N_train = std::min(N, 100000);
    std::vector<float> db_train(db.begin(),                                                                                                                                                                 
                                db.begin() + (long long)N_train * DIM);                                                                                                                                     
                                                                                                                                                                                                            
    std::cerr << "Building IVF-PQ optimized index (N=" << N << ")...\n";                                                                                                                                    
    Timer t_build;                                                                                                                                                                                          
    t_build.start();                                                                                                                                                                                        
    IVFPQIndexOpt index = build_index_opt(db, db_train, N, N_train);
    double build_ms = t_build.stop_ms();                                                                                                                                                                    
    std::cerr << "Index built in " << build_ms << " ms\n\n"; 
    // std::cerr << "Starting query phase...\n";  // ADD THIS
    // std::cerr.flush();                                                                                                                                                  
                                
    
    std::vector<std::vector<int>> results;                                                                                                                                                                  
    Timer t_query;                                                                                                                                                                                          
    t_query.start();
    query_ivfpq_batched(index, queries, Q, k, results);
    double query_ms = t_query.stop_ms();                                                                                                                                                                    

    // std::vector<std::vector<int>> ground_truth;
    // float recall = 0.f;
    // if (load_ground_truth_bin("benchmarks/results/ground_truth.bin",
    //                            ground_truth, Q, k)) {
    //     recall = compute_recall(ground_truth, results, k);
    // } else {
    //     std::cerr << "Warning: ground_truth.bin not found — run Stage 1 first.\n";
    // }
    auto ground_truth = load_ivecs("data/sift/sift_groundtruth.ivecs", Q, k);
    float recall = compute_recall(ground_truth, results, k);
                                                                                                                                                                                                            
    // if (!ground_truth.empty()) {
    //     const float* q0 = queries.data();
    //     std::vector<float> lut0((long long)M * KSUB);                                                                                                                                                       
    //     for (int m = 0; m < M; m++) {                                                                                                                                                                       
    //         const float* qsub   = q0 + m * DSUB;                                                                                                                                                            
    //         const float* book_m = index.codebooks.data() + (long long)m * KSUB * DSUB;                                                                                                                      
    //         for (int c = 0; c < KSUB; c++)                                                                                                                                                                  
    //             lut0[m * KSUB + c] = l2_sq_cpu(qsub, book_m + c * DSUB, DSUB);
    //     }                                                                                                                                                                                                   
    //     auto pq_dist = [&](int vec_id) {
    //         std::vector<uint8_t> code(M);                                                                                                                                                                   
    //         encode_vector(db.data() + (long long)vec_id * DIM, index.codebooks, code.data());
    //         float d = 0.f;                                                                                                                                                                                  
    //         for (int m = 0; m < M; m++) d += lut0[m * KSUB + code[m]];
    //         return d;                                                                                                                                                                                       
    //     };      
    //     int nn0  = ground_truth[0][0];                                                                                                                                                                      
    //     int nn1  = ground_truth[0][1];                                                                                                                                                                      
    //     int nn9  = ground_truth[0][9];
    //     int far0 = 50000;                                                                                                                                                                                   
    //     float true_nn0  = l2_sq_cpu(q0, db.data() + (long long)nn0  * DIM, DIM);                                                                                                                            
    //     float true_far0 = l2_sq_cpu(q0, db.data() + (long long)far0 * DIM, DIM);                                                                                                                            
    //     std::cerr << "\n── PQ Sanity Check ──────────────────────────\n"                                                                                                                                    
    //             << "True  dist to NN#0  (id=" << nn0  << "): " << true_nn0  << "\n"                                                                                                                       
    //             << "True  dist to far   (id=" << far0 << "): " << true_far0 << "\n"                                                                                                                       
    //             << "PQ    dist to NN#0  (id=" << nn0  << "): " << pq_dist(nn0)  << "\n"                                                                                                                   
    //             << "PQ    dist to NN#1  (id=" << nn1  << "): " << pq_dist(nn1)  << "\n"                                                                                                                   
    //             << "PQ    dist to NN#9  (id=" << nn9  << "): " << pq_dist(nn9)  << "\n"                                                                                                                   
    //             << "PQ    dist to far   (id=" << far0 << "): " << pq_dist(far0) << "\n"                                                                                                                   
    //             << "─────────────────────────────────────────────\n";                                                                                                                                     
    // }           
                                                                                                                                                                                                            
    {           
        std::ofstream f("benchmarks/results/ivf_pq_optimized.csv", std::ios::app);                                                                                                                          
        f << "ivf_pq_optimized," << N << "," << DIM << "," << k << ","
        << build_ms << "," << query_ms << "," << recall << "\n";                                                                                                                                          
    }
                                                                                                                                                                                                            
    double qps = Q / (query_ms / 1000.0);
    std::cout << "Stage      : ivf_pq_optimized\n"                                                                                                                                                          
            << "N          : " << N          << "\n"                                                                                                                                                      
            << "dim        : " << DIM        << "\n"
            << "k          : " << k          << "\n"                                                                                                                                                      
            << "queries    : " << Q          << "\n"
            << "nlist      : " << NLIST      << "\n"                                                                                                                                                      
            << "nprobe     : " << NPROBE     << "\n"                                                                                                                                                      
            << "M          : " << M          << "\n"
            << "n_streams  : " << N_STREAMS  << "\n"                                                                                                                                                      
            << "build_ms   : " << build_ms   << "\n"                                                                                                                                                      
            << "query_ms   : " << query_ms   << "\n"
            << "QPS        : " << qps        << "\n"                                                                                                                                                      
            << "recall@k   : " << recall     << "\n";
                                                                                                                                                                                                            
    return 0;   
}                                                   