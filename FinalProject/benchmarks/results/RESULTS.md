# Benchmark Results

Final runs: N=1,000,000 vectors, dim=128, k=10, Q=100 queries, SIFT1M dataset.
Platform: NVIDIA RTX 4000 Ada (sm_89), Euler HPC cluster, CUDA 13.0.0.
Speedup computed as (Stage 1 query time) / (Stage N query time).

| Stage | Description | Build (ms) | Query (ms) | QPS | Recall@10 | Speedup vs S1 | Date | Job ID |
|---|---|---|---|---|---|---|---|---|
| 1 | cpu_brute_force | — | 46,415.7 | 2.15 | 1.000 | 1× | 2026-04-30 | 352694 |
| 2 | gpu_naive | — | 6,756.2 | 14.85 | 0.998 | 6.9× | 2026-04-30 | 352698 |
| 3 | gpu_optimized (smem tiled) | — | 1,122.9 | 90.69 | 0.998 | 42.1× | 2026-04-30 | 352699 |
| 4 | ivf_pq_basic | 341,857 | 534.5 | 187.1 | 0.720 | 86.9× | 2026-04-30 | 352710 |
| 5 | ivf_pq_optimized (OMP+smem+SoA+streams) | 143,685 | 363.1 | 275.4 | 0.720 | 127.8× | 2026-04-30 | 352713 |

## Notes

- Stage 4→5 build improvement: **2.38×** (341,857 ms → 143,685 ms) via OpenMP k-means.
- Stage 4→5 query improvement: **1.47×** (534.5 ms → 363.1 ms) via shared-memory LUT, flat SoA inverted lists, and CUDA streams.
- Recall unchanged from Stage 4→5: NLIST=256 and NPROBE=32 held constant to isolate implementation effects.
- Stage 2/3 recall=0.998 (not 1.0): floating-point accumulation order differs between CPU and GPU; rare tie-breaking differences. Not a bug.
- Stage 4/5 recall=0.72: expected for NLIST=256 on 1M vectors (FAISS recommends ~4000). Accuracy/speed trade-off of approximate search.
