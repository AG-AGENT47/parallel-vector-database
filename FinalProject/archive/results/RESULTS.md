# Benchmark Results

All runs: N=100K vectors, dim=128, k=10, Q=1000 queries, seed=42.

| Stage | Description | Time (ms) | QPS | Recall@k | Speedup vs Stage 1 | Date | Machine |
|---|---|---|---|---|---|---|---|
| 1 | cpu_brute_force | 74911.4 | 13.35 | 1.0 | 1x | 2026-04-17 | Euler RTX 4000 Ada |
