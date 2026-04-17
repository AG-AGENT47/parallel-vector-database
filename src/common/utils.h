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