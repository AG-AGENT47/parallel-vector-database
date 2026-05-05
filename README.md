# GPU-Accelerated Approximate Nearest Neighbor Search Engine

Final project for **ME/CS/ECE 759 — Parallel Programming, Spring 2026**
University of Wisconsin–Madison

**Team:** Avyakt Garg (garg62@wisc.edu) · Harshit Agarwal (hagarwal23@wisc.edu)

---

## What This Is

A 5-stage GPU-accelerated ANN search engine implementing the IVF-PQ algorithm — the same core algorithm behind Meta's FAISS. Given a query vector, the engine finds the k most similar vectors from a database of 1 million 128-dimensional SIFT1M vectors.

Each stage adds one or more HPC techniques over the previous, making the contribution of each technique measurable in isolation.

| Stage | Description | QPS | Recall@10 | Speedup vs S1 |
|-------|-------------|-----|-----------|---------------|
| 1 | CPU brute-force baseline | 2.15 | 1.000 | 1× |
| 2 | Naive GPU k-NN | 14.85 | 0.998 | 6.9× |
| 3 | Shared-memory tiled GPU k-NN | 90.69 | 0.998 | 42.1× |
| 4 | IVF-PQ (serial k-means, global LUT) | 187.1 | 0.720 | 86.9× |
| 5 | IVF-PQ (OpenMP + smem LUT + SoA + CUDA streams) | 275.4 | 0.720 | 127.8× |

Benchmarked on an NVIDIA RTX 4000 Ada (sm_89), Euler HPC cluster, N=1M, Q=100.

---

## Repository Structure

```
parallel-vector-database/
├── CMakeLists.txt
├── src/
│   ├── common/utils.h               # Timer, compute_recall, log_result, fvecs loader
│   ├── cpu/baseline_knn.cpp         # Stage 1
│   └── gpu/
│       ├── naive_knn.cu             # Stage 2
│       ├── optimized_knn.cu         # Stage 3
│       ├── ivf_pq_basic.cu          # Stage 4
│       └── ivf_pq_optimized.cu      # Stage 5
├── slurm/                           # SLURM job scripts (Euler instruction partition)
├── benchmarks/results/              # CSV logs + ground_truth.bin
└── final_results/                   # Final benchmark .out files (N=1M, all 5 stages)
```

---

## Getting Started (Euler HPC Cluster)

### 1. Clone and build

```bash
ssh garg62@euler.engr.wisc.edu
git clone git@github.com:AG-AGENT47/parallel-vector-database.git
cd parallel-vector-database
module load nvidia/cuda/13.0.0
mkdir -p build && cd build
cmake ..
make baseline_knn naive_knn optimized_knn ivf_pq_basic ivf_pq_optimized
```

### 2. Download SIFT1M dataset

The dataset is not included in the repo (2.6 GB). Download it once on Euler:

```bash
mkdir -p ~/parallel-vector-database/data/sift
cd ~/parallel-vector-database/data/sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar xzf sift.tar.gz
```

### 3. Run via SLURM

**Stage 1 must complete before Stages 2–5** — it writes `benchmarks/results/ground_truth.bin` used for recall computation.

```bash
cd ~/parallel-vector-database

# Run Stage 1 first and wait for it to finish:
sbatch slurm/baseline_knn.slurm
squeue -u $USER

# Then submit the rest:
sbatch slurm/naive_knn.slurm
sbatch slurm/optimized_knn.slurm
sbatch slurm/ivf_pq_basic.slurm
sbatch slurm/ivf_pq_optimized.slurm
```

Results are printed to `output-<jobid>.out` and logged as CSV rows in `benchmarks/results/`.

---

## Acknowledgements

We would like to thank **Professor Dan Negrut** and the entire ME/CS/ECE 759 teaching team for an excellent course. The structured progression from serial baselines through GPU optimization gave us the framework to tackle a genuinely complex parallel system. The course materials, best practices guide, and access to the Euler HPC cluster made this project possible.

---

## License

Released as open source under the BSD 3-Clause License.
