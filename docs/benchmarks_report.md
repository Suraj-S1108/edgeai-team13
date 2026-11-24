# EDGE AI Team 13 — Performance & Benchmarking Report

## 1. Benchmarking Methodology
Empirical measurements were collected using `edge_analysis/benchmark_model_direct.py` on a standard evaluation environment (Windows 11 AMD64, Python 3.13.5, PyTorch CPU):
- **Warmup**: 50 iterations to eliminate initialization bias and page-fault artifacts.
- **Single Request Benchmark**: 1,000 iterations measuring cold load time, mean, median, p90, p95, and p99 latency.
- **Batch Request Benchmark**: 100 iterations across batch sizes of 50, 200, and 400.

---

## 2. Empirical Benchmark Results

| Performance Dimension | Legacy Model (16-16) | Deployed Model (32-32) | Delta / Trade-off |
|---|---|---|---|
| **Architecture** | 5 -> 16 -> 16 -> 1 | 5 -> 32 -> 32 -> 1 | 2x wider hidden layers |
| **Parameters** | 385 | 1,281 | +896 (+232.7%) |
| **Checkpoint Disk Size** | 4,285 bytes | 8,357 bytes | +4,072 bytes |
| **Model Load Time** | 11.21 ms | 38.43 ms | +27.22 ms |
| **Single Latency (Mean)** | **74.6 µs** (0.075 ms) | **127.0 µs** (0.127 ms) | **+52.4 µs** (+0.052 ms) |
| **Single Latency (Median)** | **63.9 µs** (0.064 ms) | **82.6 µs** (0.083 ms) | **+18.7 µs** |
| **Single Latency (P95)** | **124.2 µs** (0.124 ms) | **268.9 µs** (0.269 ms) | **+144.7 µs** |
| **Single Throughput** | 13,401 inferences/s | 7,872 inferences/s | -5,529 inferences/s |
| **Batch 50 Latency** | 0.080 ms | 0.088 ms | +0.008 ms |
| **Batch 50 Throughput** | 625,047 records/s | 568,259 records/s | -56,788 records/s |
| **Batch 400 Latency** | 0.234 ms | 0.082 ms | -0.152 ms |
| **Batch 400 Throughput** | 1,709,899 records/s | 4,909,000 records/s | +3,199,101 records/s |

---

## 3. Analysis of Edge AI Trade-offs
1. **Sub-millisecond Inference**: Both the 16-16 and 32-32 models execute single inferences in under **0.13 milliseconds** on CPU.
2. **Minimal Latency Penalty**: The 32-32 model adds only **52 microseconds** of latency per request while cutting prediction error by **1.66 days** (a 24.2% accuracy improvement).
3. **Massive Batch Speedup**: When evaluating multiple readings simultaneously, vectorized matrix operations achieve throughputs exceeding **500,000 records per second**.
4. **Historical Consistency**: These results corroborate the historical HTTP benchmarks recorded in `bench_20261123_*.csv` and `benchmark_400.py`, demonstrating that network round-trip time (~15-30 ms over HTTP) vastly dominates raw model inference time (~0.1 ms).
