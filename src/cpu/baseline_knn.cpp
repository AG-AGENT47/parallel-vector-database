#include "common/utils.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Squared L2 distance between two dim-dimensional vectors.
inline float l2_sq(const float* a, const float* b, int dim) {
    float d = 0.f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

// Brute-force exact k-NN for all Q queries against N database vectors.
// db      : flat row-major float array, size N*dim
// queries : flat row-major float array, size Q*dim
// results : output, size Q*k  (index of nearest neighbors, sorted by distance)
void brute_force_knn(
    const std::vector<float>& db,
    const std::vector<float>& queries,
    int N, int Q, int dim, int k,
    std::vector<std::vector<int>>& results)
{
    std::vector<int> idx(N);
    for (int q = 0; q < Q; q++) {
        const float* qvec = queries.data() + (long long)q * dim;

        // Build distance array for this query.
        std::vector<float> dists(N);
        for (int i = 0; i < N; i++) {
            dists[i] = l2_sq(qvec, db.data() + (long long)i * dim, dim);
        }

        // Index array [0, 1, ..., N-1].
        std::iota(idx.begin(), idx.end(), 0);

        // Partial sort: first k indices by ascending distance.
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&dists](int a, int b) { return dists[a] < dists[b]; });

        results[q].assign(idx.begin(), idx.begin() + k);
    }
}

// Write ground truth to a flat binary file: Q*k int32 values, row-major.
// Future GPU stages load this file to compute recall@k without re-running brute force.
void save_ground_truth(const std::string& path,
                       const std::vector<std::vector<int>>& results,
                       int Q, int k)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Warning: could not write ground truth to " << path << "\n";
        return;
    }
    for (int q = 0; q < Q; q++) {
        f.write(reinterpret_cast<const char*>(results[q].data()),
                k * sizeof(int));
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <n_vectors> <dim> <k> <n_queries>\n";
        return 1;
    }

    const int N  = std::atoi(argv[1]);
    const int dim = std::atoi(argv[2]);
    const int k  = std::atoi(argv[3]);
    const int Q  = std::atoi(argv[4]);

    if (k > N) {
        std::cerr << "Error: k (" << k << ") cannot exceed n_vectors (" << N << ")\n";
        return 1;
    }

    // ── Data generation ───────────────────────────────────────────────────────
    int actual_N, actual_dim, actual_Q, qd;
    std::vector<float> db      = load_fvecs("data/sift/sift_base.fvecs",
                                            actual_N, actual_dim, N);
    std::vector<float> queries = load_fvecs("data/sift/sift_query.fvecs",
                                            actual_Q, qd, Q);
    std::cerr << "First vector first 5 values: "
          << db[0] << " " << db[1] << " " << db[2] << " "
          << db[3] << " " << db[4] << "\n";
    // ── Brute-force k-NN ──────────────────────────────────────────────────────
    std::vector<std::vector<int>> results(Q, std::vector<int>(k));

    Timer t;
    t.start();
    brute_force_knn(db, queries, N, Q, dim, k, results);
    double ms = t.stop_ms();

    // ── Recall (should be 1.0 — this is exact search / ground truth) ──────────
    // We compare results against itself to verify the recall infrastructure.
    float recall = compute_recall(results, results, k);

    // ── Save ground truth for future GPU stages ────────────────────────────────
    save_ground_truth("benchmarks/results/ground_truth.bin", results, Q, k);

    // Also log what dataset was used
    std::cout << "Dataset: SIFT1M (sift_base.fvecs)\n";

    // ── Log to CSV ────────────────────────────────────────────────────────────
    log_result("benchmarks/results/baseline.csv",
               "cpu_brute_force", N, dim, k, ms, recall);

    // ── Print summary ─────────────────────────────────────────────────────────
    double qps = Q / (ms / 1000.0);
    std::cout << "Stage    : cpu_brute_force\n"
              << "N        : " << N   << "\n"
              << "dim      : " << dim << "\n"
              << "k        : " << k   << "\n"
              << "queries  : " << Q   << "\n"
              << "time_ms  : " << ms  << "\n"
              << "QPS      : " << qps << "\n"
              << "recall@k : " << recall << "\n";

    return 0;
}
