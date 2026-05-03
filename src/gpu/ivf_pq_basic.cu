// ═══════════════════════════════════════════════════════════════════════════
// Stage 4: IVF-PQ Basic
//
// CPU handles:  k-means training, PQ codebook training, index building,
//               coarse quantizer at query time, LUT construction, top-k
// GPU handles:  ADC scan (the hot path — approximate distance computation)
//
// Flow:
//   build_index():
//     1. CPU k-means → 256 coarse centroids
//     2. CPU PQ training → 8 codebooks (one per subspace)
//     3. Encode all DB vectors → 8-byte PQ codes
//     4. Group codes into inverted lists by cluster assignment
//
//   query_ivfpq():
//     1. CPU: find top NPROBE clusters for query (coarse quantizer)
//     2. CPU: build LUT — LUT[m][c] = dist(query_subvec_m, codebook[m][c])
//     3. GPU: adc_scan_basic — each thread sums M table lookups for one vector
//     4. CPU: partial sort → top-k
// ═══════════════════════════════════════════════════════════════════════════

#include "../common/utils.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

// ── Error checking macro (same as previous stages) ───────────────────────────
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
// These are compile-time constants so the compiler can unroll loops and size
// shared-memory arrays. If you change DIM, you must also change M so DIM%M==0.
static constexpr int DIM    = 128;  // vector dimensionality
static constexpr int NLIST  = 256;  // number of coarse clusters
static constexpr int NPROBE = 32;   // clusters searched per query
static constexpr int M      = 32;    // PQ subspaces
static constexpr int NBITS  = 8;    // bits per PQ code → 2^8 = 256 codewords
static constexpr int KSUB   = 1 << NBITS;  // = 256 codewords per subspace
static constexpr int DSUB   = DIM / M;     // = 16 dims per subspace


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1: Utility
// ═══════════════════════════════════════════════════════════════════════════

// CPU squared L2 distance (same as Stage 1)
inline float l2_sq_cpu(const float* a, const float* b, int d) {
    float s = 0.f;
    for (int i = 0; i < d; i++) { float diff = a[i]-b[i]; s += diff*diff; }
    return s;
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2: CPU K-Means (Lloyd's Algorithm)
//
// Used twice:
//   (a) to train the coarse quantizer (NLIST centroids over full DIM-dim vectors)
//   (b) inside train_pq() per subspace (KSUB centroids over DSUB-dim subvectors)
//
// Args:
//   vectors   : N * dim, row-major
//   N, dim    : dataset size and dimensionality
//   ncentroids: how many clusters (k)
//   max_iter  : Lloyd's iterations
//   centroids : output, ncentroids * dim
//   assignments: output, N — which centroid each vector belongs to
// ═══════════════════════════════════════════════════════════════════════════
void kmeans_cpu(const std::vector<float>& vectors,
                int N, int dim, int ncentroids, int max_iter,
                std::vector<float>& centroids,
                std::vector<int>& assignments)
{
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> pick(0, N - 1);

    // ── Initialize centroids by sampling random vectors ───────────────────
    // (Simple random init; k-means++ would be more stable but adds complexity)
    centroids.assign(ncentroids * dim, 0.f);
    for (int c = 0; c < ncentroids; c++) {
        int idx = pick(gen);
        std::copy(vectors.data() + (long long)idx * dim,
                  vectors.data() + (long long)idx * dim + dim,
                  centroids.data() + c * dim);
    }

    assignments.resize(N, 0);

    for (int iter = 0; iter < max_iter; iter++) {

        // ── Assignment step: each vector → nearest centroid ───────────────
        for (int i = 0; i < N; i++) {
            float best = std::numeric_limits<float>::max();
            int   best_c = 0;
            for (int c = 0; c < ncentroids; c++) {
                float d = l2_sq_cpu(vectors.data() + (long long)i * dim,
                                    centroids.data() + c * dim, dim);
                if (d < best) { best = d; best_c = c; }
            }
            assignments[i] = best_c;
        }

        // ── Update step: recompute centroids as cluster mean ──────────────
        std::vector<float> new_cents(ncentroids * dim, 0.f);
        std::vector<int>   counts(ncentroids, 0);

        for (int i = 0; i < N; i++) {
            int c = assignments[i];
            counts[c]++;
            const float* v = vectors.data() + (long long)i * dim;
            float*       cc = new_cents.data() + c * dim;
            for (int d = 0; d < dim; d++) cc[d] += v[d];
        }

        for (int c = 0; c < ncentroids; c++) {
            if (counts[c] > 0) {
                float inv = 1.f / counts[c];
                float* cc = new_cents.data() + c * dim;
                for (int d = 0; d < dim; d++) cc[d] *= inv;
            } else {
                // Empty cluster — re-seed with a random vector so it doesn't
                // stay dead for the rest of training.
                int idx = pick(gen);
                std::copy(vectors.data() + (long long)idx * dim,
                          vectors.data() + (long long)idx * dim + dim,
                          new_cents.data() + c * dim);
            }
        }

        centroids = std::move(new_cents);
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3: CPU Product Quantization Training
//
// For each of M subspaces, extract the DSUB-dim subvectors from all N
// database vectors, then run k-means to get KSUB centroids. Those centroids
// ARE the codebook for that subspace.
//
// Output codebooks layout: M * KSUB * DSUB (row-major)
//   codebooks[m][c][d] = codebooks[m*KSUB*DSUB + c*DSUB + d]
// ═══════════════════════════════════════════════════════════════════════════
void train_pq(const std::vector<float>& db, int N,
              std::vector<float>& codebooks)
{
    codebooks.resize((long long)M * KSUB * DSUB);
    std::vector<float> subvecs((long long)N * DSUB);
    std::vector<int>   dummy_assign;

    for (int m = 0; m < M; m++) {
        // Extract subvectors for this subspace
        // Subspace m covers dimensions [m*DSUB, (m+1)*DSUB)
        for (int i = 0; i < N; i++) {
            const float* src = db.data() + (long long)i * DIM + m * DSUB;
            float*       dst = subvecs.data() + (long long)i * DSUB;
            std::copy(src, src + DSUB, dst);
        }

        // Run k-means on these DSUB-dim subvectors → KSUB centroids
        std::vector<float> sub_centroids;
        kmeans_cpu(subvecs, N, DSUB, KSUB, /*max_iter=*/8,
                   sub_centroids, dummy_assign);

        // Save into codebooks
        std::copy(sub_centroids.begin(), sub_centroids.end(),
                  codebooks.data() + (long long)m * KSUB * DSUB);

        std::cerr << "  PQ subspace " << m+1 << "/" << M << " trained\n";
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4: CPU PQ Encoding
//
// Compress one DIM-dim vector into M bytes.
// For each subspace m: find the codeword (0..KSUB-1) whose centroid is
// closest to the subvector, write that index as one byte.
// ═══════════════════════════════════════════════════════════════════════════
void encode_vector(const float* vec,
                   const std::vector<float>& codebooks,
                   uint8_t* code_out)
{
    for (int m = 0; m < M; m++) {
        const float* subvec   = vec + m * DSUB;
        const float* book_m   = codebooks.data() + (long long)m * KSUB * DSUB;

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
// SECTION 5: IVF-PQ Index Data Structure
//
// This holds everything produced during index building.
// At query time, only this struct + the query vectors are needed.
// ═══════════════════════════════════════════════════════════════════════════
struct IVFPQIndex {
    int N;      // number of database vectors
    int nlist;  // number of coarse clusters (= NLIST)

    // Coarse quantizer centroids: nlist * DIM
    std::vector<float> coarse_centroids;

    // PQ codebooks: M * KSUB * DSUB
    std::vector<float> codebooks;

    // Inverted lists:
    //   list_ids[c]   = vector IDs (original indices into DB) for cluster c
    //   list_codes[c] = flat PQ codes for those vectors (each vector = M bytes)
    //   list_sizes[c] = how many vectors are in cluster c
    std::vector<std::vector<int>>     list_ids;
    std::vector<std::vector<uint8_t>> list_codes;
    std::vector<int>                  list_sizes;
};


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 6: Index Building
//
// Orchestrates the three offline phases and returns a ready-to-query index.
// ═══════════════════════════════════════════════════════════════════════════
IVFPQIndex build_index(const std::vector<float>& db,
                       const std::vector<float>& db_train,
                       int N, int N_train)
{
    IVFPQIndex idx;
    idx.N     = N;
    idx.nlist = NLIST;

    // ── Phase 1: Coarse quantizer training ───────────────────────────────
    std::cerr << "[1/3] Training coarse quantizer (" << NLIST
              << " clusters, 8 iters)...\n";
    std::vector<int> coarse_assign;
    kmeans_cpu(db_train, N_train, DIM, NLIST, /*max_iter=*/8,
            idx.coarse_centroids, coarse_assign);
    std::cerr << "      Done.\n";

    // ── Phase 2: PQ codebook training ────────────────────────────────────
    std::cerr << "[2/3] Training PQ codebooks (M=" << M
              << ", KSUB=" << KSUB << ")...\n";
    train_pq(db_train, N_train, idx.codebooks);
    std::cerr << "      Done.\n";

    // ── Phase 3: Encode vectors and populate inverted lists ───────────────
    // For each DB vector:
    //   - Its coarse cluster is already known from Phase 1 (coarse_assign[i])
    //   - Encode it to M bytes using the PQ codebooks from Phase 2
    //   - Append (id, code) to the appropriate inverted list
    std::cerr << "[3/3] Building inverted lists and encoding vectors...\n";
    idx.list_ids.resize(NLIST);
    idx.list_codes.resize(NLIST);
    idx.list_sizes.resize(NLIST, 0);

    std::vector<int> full_assign(N);
    for (int i = 0; i < N; i++) {
        float best = std::numeric_limits<float>::max();
        int   best_c = 0;
        for (int c = 0; c < NLIST; c++) {
            float d = l2_sq_cpu(db.data() + (long long)i * DIM,
                                idx.coarse_centroids.data() + c * DIM, DIM);
            if (d < best) { best = d; best_c = c; }
        }
        full_assign[i] = best_c;
    }
    std::vector<uint8_t> code(M);
    for (int i = 0; i < N; i++) {
        int c = full_assign[i];
        idx.list_ids[c].push_back(i);
        encode_vector(db.data() + (long long)i * DIM, idx.codebooks, code.data());
        idx.list_codes[c].insert(idx.list_codes[c].end(), code.begin(), code.end());
        idx.list_sizes[c]++;
    }
    std::cerr << "      Done.\n";

    // Print some stats to sanity-check the index
    int min_sz = N, max_sz = 0;
    for (int c = 0; c < NLIST; c++) {
        min_sz = std::min(min_sz, idx.list_sizes[c]);
        max_sz = std::max(max_sz, idx.list_sizes[c]);
    }
    std::cerr << "      List sizes — min: " << min_sz
              << "  max: " << max_sz
              << "  avg: " << N/NLIST << "\n";

    return idx;
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7: GPU ADC Scan Kernel (Basic Version)
//
// This is the GPU hot path. Given a precomputed lookup table (LUT) and a
// flat array of PQ-encoded vectors, compute the approximate L2 distance
// from the query to each encoded vector.
//
// For each candidate vector i:
//   dist ≈ LUT[0][code[i*M+0]] + LUT[1][code[i*M+1]] + ... + LUT[M-1][code[i*M+M-1]]
//
// "Basic" means: LUT is read from global memory.
// "Optimized" (Stage 5) will load LUT into shared memory first.
//
// Args:
//   lut          : M * KSUB floats — distance table for this query
//   codes        : n_cands * M uint8 — PQ codes of candidate vectors
//   dists_out    : n_cands floats — output approximate distances
//   n_cands      : number of candidate vectors to score
// ═══════════════════════════════════════════════════════════════════════════
__global__ void adc_scan_basic(
    const float*   __restrict__ lut,
    const uint8_t* __restrict__ codes,
    float*         __restrict__ dists_out,
    int n_cands)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cands) return;

    // Sum M table lookups to get the approximate distance.
    // Each lookup: read one byte (the codeword), use it as an index into
    // the LUT row for that subspace.
    float dist = 0.f;
    #pragma unroll
    for (int m = 0; m < M; m++) {
        uint8_t c = codes[i * M + m];
        dist += lut[m * KSUB + c];   // global memory read (unoptimized for now)
    }
    dists_out[i] = dist;
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 8: Query Function
//
// For each query vector:
//   1. Coarse quantizer (CPU): find top NPROBE cluster centroids
//   2. LUT construction (CPU): precompute M*KSUB distance table
//   3. ADC scan (GPU): score all candidates in those NPROBE clusters
//   4. Top-k (CPU): sort scores, return k best vector IDs
// ═══════════════════════════════════════════════════════════════════════════
void query_ivfpq(
    const IVFPQIndex&         index,
    const std::vector<float>& queries,
    int Q, int k,
    std::vector<std::vector<int>>& results)
{
    results.assign(Q, std::vector<int>(k, -1));

    // ── Allocate GPU buffers once, reuse across all queries ───────────────
    // Worst case: all N vectors are candidates (all in NPROBE clusters)
    const int max_cands = index.N;

    float*   d_lut   = nullptr;
    uint8_t* d_codes = nullptr;
    float*   d_dists = nullptr;

    CUDA_CHECK(cudaMalloc(&d_lut,   (long long)M * KSUB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_codes, (long long)max_cands * M * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_dists, (long long)max_cands * sizeof(float)));

    // Reusable host buffer for scores
    std::vector<float>   h_dists;
    std::vector<uint8_t> all_codes;
    std::vector<int>     all_ids;

    for (int q = 0; q < Q; q++) {
        const float* qvec = queries.data() + (long long)q * DIM;

        // ── Step 1: Coarse quantizer — find top NPROBE clusters ───────────
        // Compute distance from query to all NLIST centroids.
        // This is O(NLIST * DIM) — fast because NLIST=256 is small.
        std::vector<float> cdists(NLIST);
        for (int c = 0; c < NLIST; c++)
            cdists[c] = l2_sq_cpu(qvec,
                                  index.coarse_centroids.data() + c * DIM,
                                  DIM);

        // Partially sort to get the NPROBE nearest centroids
        std::vector<int> probe_order(NLIST);
        std::iota(probe_order.begin(), probe_order.end(), 0);
        std::partial_sort(probe_order.begin(),
                          probe_order.begin() + NPROBE,
                          probe_order.end(),
                          [&cdists](int a, int b) {
                              return cdists[a] < cdists[b];
                          });
        // probe_order[0..NPROBE-1] are now the NPROBE nearest cluster IDs

        // ── Step 2: Build LUT for this query ─────────────────────────────
        // LUT[m][c] = ||query_subvec_m - codebook[m][c]||^2
        // This is what makes ADC fast: compute these M*KSUB distances once,
        // then reuse them for every candidate vector in every probe cluster.
        std::vector<float> lut((long long)M * KSUB);
        for (int m = 0; m < M; m++) {
            const float* qsub   = qvec + m * DSUB;
            const float* book_m = index.codebooks.data() + (long long)m * KSUB * DSUB;
            for (int c = 0; c < KSUB; c++) {
                lut[m * KSUB + c] = l2_sq_cpu(qsub, book_m + c * DSUB, DSUB);
            }
        }

        // ── Step 3: Gather candidates from NPROBE inverted lists ─────────
        all_codes.clear();
        all_ids.clear();

        for (int p = 0; p < NPROBE; p++) {
            int c  = probe_order[p];
            int sz = index.list_sizes[c];
            if (sz == 0) continue;

            // Append vector IDs from this cluster
            all_ids.insert(all_ids.end(),
                           index.list_ids[c].begin(),
                           index.list_ids[c].end());

            // Append PQ codes from this cluster (sz * M bytes)
            all_codes.insert(all_codes.end(),
                             index.list_codes[c].begin(),
                             index.list_codes[c].end());
        }

        int n_cands = static_cast<int>(all_ids.size());
        if (n_cands == 0) continue;

        // ── Step 4: GPU ADC scan ──────────────────────────────────────────
        // Upload LUT and codes, launch kernel, download scores
        CUDA_CHECK(cudaMemcpy(d_lut, lut.data(),
                              (long long)M * KSUB * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_codes, all_codes.data(),
                              (long long)n_cands * M * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));

        const int threads = 256;
        const int blocks  = (n_cands + threads - 1) / threads;
        adc_scan_basic<<<blocks, threads>>>(d_lut, d_codes, d_dists, n_cands);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        h_dists.resize(n_cands);
        CUDA_CHECK(cudaMemcpy(h_dists.data(), d_dists,
                              (long long)n_cands * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // ── Step 5: Top-k selection (CPU) ─────────────────────────────────
        int actual_k = std::min(k, n_cands);
        std::vector<int> idx_sort(n_cands);
        std::iota(idx_sort.begin(), idx_sort.end(), 0);
        std::partial_sort(idx_sort.begin(),
                          idx_sort.begin() + actual_k,
                          idx_sort.end(),
                          [&h_dists](int a, int b) {
                              return h_dists[a] < h_dists[b];
                          });

        for (int j = 0; j < actual_k; j++)
            results[q][j] = all_ids[idx_sort[j]];
    }

    CUDA_CHECK(cudaFree(d_lut));
    CUDA_CHECK(cudaFree(d_codes));
    CUDA_CHECK(cudaFree(d_dists));
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 9: Ground Truth Loader (same format as Stage 1)
// ═══════════════════════════════════════════════════════════════════════════
bool load_ground_truth(const std::string& path,
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
// SECTION 10: Main
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

    // ── Data generation (identical seed to all previous stages) ──────────
    int actual_N, actual_dim, actual_Q, qd;
    std::vector<float> db      = load_fvecs("data/sift/sift_base.fvecs",
                                            actual_N, actual_dim, N);
    std::vector<float> queries = load_fvecs("data/sift/sift_query.fvecs",
                                            actual_Q, qd, Q);

    // Training subset — 100K is enough to learn good centroids and codebooks
    int N_train = std::min(N, 100000);
    std::vector<float> db_train(db.begin(),
                                db.begin() + (long long)N_train * DIM);

    // ── Build index (offline phase — timed separately) ────────────────────
    std::cerr << "Building IVF-PQ index (N=" << N << ")...\n";
    Timer t_build;
    t_build.start();
    IVFPQIndex index = build_index(db, db_train, N, N_train);
    double build_ms = t_build.stop_ms();
    std::cerr << "Index built in " << build_ms << " ms\n\n";

    // ── Query phase (timed) ───────────────────────────────────────────────
    std::vector<std::vector<int>> results;

    Timer t_query;
    t_query.start();
    query_ivfpq(index, queries, Q, k, results);
    double query_ms = t_query.stop_ms();

    // ── Recall@k against Stage 1 ground truth ────────────────────────────
    std::vector<std::vector<int>> ground_truth;
    float recall = 0.f;
    if (load_ground_truth("benchmarks/results/ground_truth.bin",
                          ground_truth, Q, k)) {
        recall = compute_recall(ground_truth, results, k);
    } else {
        std::cerr << "Warning: ground_truth.bin not found — run Stage 1 first.\n";
    }

    // // ── PQ Sanity Check (ADD HERE) ────────────────────────────────────────
    // const float* q0 = queries.data();
    // std::vector<float> lut0((long long)M * KSUB);
    // for (int m = 0; m < M; m++) {
    //     const float* qsub   = q0 + m * DSUB;
    //     const float* book_m = index.codebooks.data() + (long long)m * KSUB * DSUB;
    //     for (int c = 0; c < KSUB; c++)
    //         lut0[m * KSUB + c] = l2_sq_cpu(qsub, book_m + c * DSUB, DSUB);
    // }
    // auto pq_dist = [&](int vec_id) {
    //     std::vector<uint8_t> code(M);
    //     encode_vector(db.data() + (long long)vec_id * DIM, index.codebooks, code.data());
    //     float d = 0.f;
    //     for (int m = 0; m < M; m++) d += lut0[m * KSUB + code[m]];
    //     return d;
    // };
    // int nn0  = ground_truth[0][0];
    // int nn1  = ground_truth[0][1];
    // int nn9  = ground_truth[0][9];
    // int far0 = 50000;
    // float true_nn0  = l2_sq_cpu(q0, db.data() + (long long)nn0  * DIM, DIM);
    // float true_far0 = l2_sq_cpu(q0, db.data() + (long long)far0 * DIM, DIM);
    // std::cerr << "\n── PQ Sanity Check ──────────────────────────\n"
    //           << "True  dist to NN#0  (id=" << nn0  << "): " << true_nn0  << "\n"
    //           << "True  dist to far   (id=" << far0 << "): " << true_far0 << "\n"
    //           << "PQ    dist to NN#0  (id=" << nn0  << "): " << pq_dist(nn0)  << "\n"
    //           << "PQ    dist to NN#1  (id=" << nn1  << "): " << pq_dist(nn1)  << "\n"
    //           << "PQ    dist to NN#9  (id=" << nn9  << "): " << pq_dist(nn9)  << "\n"
    //           << "PQ    dist to far   (id=" << far0 << "): " << pq_dist(far0) << "\n"
    //           << "─────────────────────────────────────────────\n";

    // ── Log and print ─────────────────────────────────────────────────────
    {
        std::ofstream f("benchmarks/results/ivf_pq_basic.csv", std::ios::app);
        f << "ivf_pq_basic," << N << "," << DIM << "," << k << ","
          << build_ms << "," << query_ms << "," << recall << "\n";
    }

    double qps = Q / (query_ms / 1000.0);
    std::cout << "Stage      : ivf_pq_basic\n"
              << "N          : " << N          << "\n"
              << "dim        : " << DIM        << "\n"
              << "k          : " << k          << "\n"
              << "queries    : " << Q          << "\n"
              << "nlist      : " << NLIST      << "\n"
              << "nprobe     : " << NPROBE     << "\n"
              << "M          : " << M          << "\n"
              << "build_ms   : " << build_ms   << "\n"
              << "query_ms   : " << query_ms   << "\n"
              << "QPS        : " << qps        << "\n"
              << "recall@k   : " << recall     << "\n";

    return 0;
}
