"""
edge_analysis/benchmark_model_direct.py -- Direct PyTorch CPU Inference Benchmark
Compares the Legacy 16-16 model with the Deployed 32-32 model under strictly equivalent conditions.

Measures:
- Model Loading Time (ms)
- Single-sample inference latency: mean, median, p90, p95, p99, min, max (ms)
- Single-sample throughput (inferences/sec)
- Batch inference latency (batch size 50, 200, 400)
- Batch throughput (records/sec)
- MACs, Parameters, and Checkpoint file size
- Saves results to edge_analysis/model_benchmark_comparison.csv
"""

import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from app.model import FEATURES, BatteryNet
DEPLOYED_CKPT = ROOT / "model" / "model.pt"
DEPLOYED_CFG = ROOT / "model" / "config.json"
LEGACY_CKPT = ROOT / "model" / "backup_16_16" / "model.pt"
LEGACY_CFG = ROOT / "model" / "backup_16_16" / "config.json"
OUTPUT_CSV = ROOT / "edge_analysis" / "model_benchmark_comparison.csv"


def benchmark_model(name, ckpt_path, cfg_path, n_single=1000, n_warmup=50):
    torch.set_num_threads(1)
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Measure load time
    t_load_start = time.perf_counter()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    net = BatteryNet(tuple(cfg["hidden"]))
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    load_time_ms = (time.perf_counter() - t_load_start) * 1000

    params = sum(p.numel() for p in net.parameters())
    ckpt_size_b = os.path.getsize(ckpt_path)
    mu = np.array(cfg["mu"], dtype=np.float32)
    sd = np.array(cfg["sd"], dtype=np.float32)

    # Synthetic realistic sensor inputs
    single_raw = np.array([[3.85, 24.5, 12, -68, 14]], dtype=np.float32)
    single_tensor = torch.from_numpy((single_raw - mu) / sd)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = net(single_tensor)

    # 2. Single record benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(n_single):
            t0 = time.perf_counter()
            _ = net(single_tensor)
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p90_lat = np.percentile(latencies, 90)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    throughput_single = 1000.0 / mean_lat if mean_lat > 0 else 0

    # 3. Batch benchmarks
    batch_results = {}
    for b_size in [50, 200, 400]:
        batch_raw = np.tile(single_raw, (b_size, 1))
        batch_tensor = torch.from_numpy((batch_raw - mu) / sd)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = net(batch_tensor)

        b_times = []
        with torch.no_grad():
            for _ in range(100):
                t0 = time.perf_counter()
                _ = net(batch_tensor)
                b_times.append((time.perf_counter() - t0) * 1000)

        b_mean = np.mean(b_times)
        b_thru = (b_size / (b_mean / 1000.0))
        batch_results[f"batch_{b_size}_latency_ms"] = round(b_mean, 4)
        batch_results[f"batch_{b_size}_throughput_rps"] = round(b_thru, 1)

    return {
        "model_name": name,
        "architecture": f"5 -> {' -> '.join(map(str, cfg['hidden']))} -> 1",
        "parameters": params,
        "checkpoint_size_bytes": ckpt_size_b,
        "load_time_ms": round(load_time_ms, 3),
        "single_latency_mean_ms": round(mean_lat, 5),
        "single_latency_median_ms": round(median_lat, 5),
        "single_latency_p90_ms": round(p90_lat, 5),
        "single_latency_p95_ms": round(p95_lat, 5),
        "single_latency_p99_ms": round(p99_lat, 5),
        "single_latency_min_ms": round(min_lat, 5),
        "single_latency_max_ms": round(max_lat, 5),
        "single_throughput_rps": round(throughput_single, 1),
        **batch_results
    }


def main():
    print("=================================================================")
    print("  EDGE AI Team 13 — Model Latency and Throughput Benchmarker")
    print("=================================================================\n")
    print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    print(f"PyTorch threads: {torch.get_num_threads()}\n")

    results = []

    print("Benchmarking Production Candidate (32-32)...")
    res_32 = benchmark_model("Production Candidate (32-32)", DEPLOYED_CKPT, DEPLOYED_CFG)
    results.append(res_32)

    if os.path.exists(LEGACY_CKPT) and os.path.exists(LEGACY_CFG):
        print("Benchmarking Legacy Baseline (16-16)...")
        res_16 = benchmark_model("Legacy Baseline (16-16)", LEGACY_CKPT, LEGACY_CFG)
        results.append(res_16)

    print("\n========== BENCHMARK RESULTS COMPARISON ==========\n")
    headers = [
        "Metric",
        "Legacy (16-16)",
        "Deployed (32-32)",
        "Delta / Impact"
    ]

    table_rows = [
        ("Architecture", res_16["architecture"], res_32["architecture"], "Wider hidden layers"),
        ("Parameters", str(res_16["parameters"]), str(res_32["parameters"]), f"+{res_32['parameters'] - res_16['parameters']} (+232.7%)"),
        ("Checkpoint Size", f"{res_16['checkpoint_size_bytes']} B", f"{res_32['checkpoint_size_bytes']} B", f"+{res_32['checkpoint_size_bytes'] - res_16['checkpoint_size_bytes']} B"),
        ("Model Load Time", f"{res_16['load_time_ms']} ms", f"{res_32['load_time_ms']} ms", f"{res_32['load_time_ms'] - res_16['load_time_ms']:+.3f} ms"),
        ("Single Latency (Mean)", f"{res_16['single_latency_mean_ms']*1000:.1f} µs", f"{res_32['single_latency_mean_ms']*1000:.1f} µs", f"{(res_32['single_latency_mean_ms'] - res_16['single_latency_mean_ms'])*1000:+.1f} µs"),
        ("Single Latency (Median)", f"{res_16['single_latency_median_ms']*1000:.1f} µs", f"{res_32['single_latency_median_ms']*1000:.1f} µs", f"{(res_32['single_latency_median_ms'] - res_16['single_latency_median_ms'])*1000:+.1f} µs"),
        ("Single Latency (P95)", f"{res_16['single_latency_p95_ms']*1000:.1f} µs", f"{res_32['single_latency_p95_ms']*1000:.1f} µs", f"{(res_32['single_latency_p95_ms'] - res_16['single_latency_p95_ms'])*1000:+.1f} µs"),
        ("Single Throughput", f"{res_16['single_throughput_rps']:,.0f} req/s", f"{res_32['single_throughput_rps']:,.0f} req/s", f"{res_32['single_throughput_rps'] - res_16['single_throughput_rps']:+,.0f} req/s"),
        ("Batch 50 Latency", f"{res_16['batch_50_latency_ms']} ms", f"{res_32['batch_50_latency_ms']} ms", f"{res_32['batch_50_latency_ms'] - res_16['batch_50_latency_ms']:+.3f} ms"),
        ("Batch 50 Throughput", f"{res_16['batch_50_throughput_rps']:,.0f} rec/s", f"{res_32['batch_50_throughput_rps']:,.0f} rec/s", f"{res_32['batch_50_throughput_rps'] - res_16['batch_50_throughput_rps']:+,.0f} rec/s"),
        ("Batch 400 Latency", f"{res_16['batch_400_latency_ms']} ms", f"{res_32['batch_400_latency_ms']} ms", f"{res_32['batch_400_latency_ms'] - res_16['batch_400_latency_ms']:+.3f} ms"),
        ("Batch 400 Throughput", f"{res_16['batch_400_throughput_rps']:,.0f} rec/s", f"{res_32['batch_400_throughput_rps']:,.0f} rec/s", f"{res_32['batch_400_throughput_rps'] - res_16['batch_400_throughput_rps']:+,.0f} rec/s"),
    ]

    print(f"{'Metric':<26} | {'Legacy (16-16)':<16} | {'Deployed (32-32)':<18} | {'Delta / Impact'}")
    print("-" * 80)
    for m, l, d, diff in table_rows:
        print(f"{m:<26} | {l:<16} | {d:<18} | {diff}")

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved benchmark comparison results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
