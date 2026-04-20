#pragma once
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

// ── Timer ──────────────────────────────────────────────
// Usage:
//   Timer t;
//   t.start();
//   ... do work ...
//   double ms = t.stop_ms();
struct Timer {
    std::chrono::high_resolution_clock::time_point _start;

    void start() {
        _start = std::chrono::high_resolution_clock::now();
    }

    double stop_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - _start).count();
    }
};

// ── Recall@k evaluator ─────────────────────────────────
// Given ground truth neighbors and approximate results,
// what fraction of true neighbors did we find?
// Perfect = 1.0, terrible = 0.0
float compute_recall(
    const std::vector<std::vector<int>>& ground_truth,  // true top-k for each query
    const std::vector<std::vector<int>>& approx_results, // our answers
    int k)
{
    int correct = 0, total = 0;
    for (int q = 0; q < ground_truth.size(); q++) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                if (approx_results[q][j] == ground_truth[q][i]) {
                    correct++;
                    break;
                }
            }
            total++;
        }
    }
    return (float)correct / total;
}

// ── Result logger ──────────────────────────────────────
// Appends a row to a CSV file for tracking benchmark results
void log_result(const std::string& filename,
                const std::string& stage,
                int n_vectors, int dim, int k,
                double time_ms, float recall) {
    std::ofstream f(filename, std::ios::app);
    f << stage << "," << n_vectors << "," << dim << ","
      << k << "," << time_ms << "," << recall << "\n";
}

// ── SIFT1M / fvecs loader ──────────────────────────────────────────────────
// Reads a .fvecs file into a flat float vector.
// Sets n = number of vectors, dim = dimensionality.
// Pass n_limit > 0 to load only the first n_limit vectors (useful for subsets).
inline std::vector<float> load_fvecs(const std::string& path,
                                     int& n, int& dim,
                                     int n_limit = -1)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Error: cannot open " << path << "\n";
        std::exit(1);
    }

    // Read dimensionality from the first 4 bytes
    f.read(reinterpret_cast<char*>(&dim), 4);
    long long bytes_per_vec = 4LL + dim * 4LL;

    // Compute total number of vectors from file size
    f.seekg(0, std::ios::end);
    long long file_size = f.tellg();
    int total_n = static_cast<int>(file_size / bytes_per_vec);

    n = (n_limit > 0) ? std::min(n_limit, total_n) : total_n;

    std::vector<float> data(static_cast<long long>(n) * dim);
    f.seekg(0);
    for (int i = 0; i < n; i++) {
        int d_check;
        f.read(reinterpret_cast<char*>(&d_check), 4);  // skip per-vector dim
        f.read(reinterpret_cast<char*>(data.data() + (long long)i * dim),
               dim * sizeof(float));
    }
    return data;
}

// ── SIFT1M / ivecs ground truth loader ────────────────────────────────────
// Reads a .ivecs file (ground truth neighbor indices).
// Each row has `gt_k` neighbors; we only keep the first `k` of them.
inline std::vector<std::vector<int>> load_ivecs(const std::string& path,
                                                int n_queries, int k)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Error: cannot open " << path << "\n";
        std::exit(1);
    }

    std::vector<std::vector<int>> gt(n_queries, std::vector<int>(k));
    for (int q = 0; q < n_queries; q++) {
        int gt_k;
        f.read(reinterpret_cast<char*>(&gt_k), 4);  // how many neighbors stored
        std::vector<int> row(gt_k);
        f.read(reinterpret_cast<char*>(row.data()), gt_k * sizeof(int));
        // Keep only the first k (we may ask for k=10, file has k=100)
        for (int j = 0; j < std::min(k, gt_k); j++)
            gt[q][j] = row[j];
    }
    return gt;
}