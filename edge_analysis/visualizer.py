"""
visualizer.py
=============
Generates publication-quality graphs for Mission 4 Edge AI analysis:
  1. Parameters versus Hidden Width (params_vs_width.png)
  2. Float32 versus Int8 Memory Footprint (f32_vs_int8_memory.png)
  3. Memory Usage versus 300 KB SRAM Limit with Cutoff Boundaries (memory_vs_sram_limit.png)
  4. ESP32-S3 Inference Latency vs Width (inference_latency_vs_width.png)
  5. Peak Live Activation Memory across Widths (activation_memory.png)

Author: Nandish (Mission 4 - Edge Device & Memory Analysis)
"""

import os
import tempfile
from typing import Dict, Any, List

# Ensure Matplotlib config directory is writable on Windows
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from edge_analysis.parameter_calculator import compare_five_widths
from edge_analysis.memory_calculator import compare_memory_across_widths, compare_activation_memory_across_widths
from edge_analysis.feasibility import (
    calculate_exact_sram_cutoff_widths,
    SRAM_LIMIT_BINARY_BYTES,
    SYSTEM_OVERHEAD_WIFI_BYTES,
)
from edge_analysis.inference_estimator import compare_inference_timing_across_five_widths


def set_plot_style():
    """Apply clean engineering visual styling."""
    plt.style.use("default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#cccccc"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.6


def plot_parameters_vs_width(output_path: str = "edge_analysis/params_vs_width.png") -> str:
    """
    Generate graph of parameters and MACs versus hidden layer width w (32 to 2048).
    """
    set_plot_style()
    df = compare_five_widths()
    widths = df["Width (w)"].tolist()
    total_params = df["Total Trainable Params (w^2+8w+1)"].tolist()
    total_macs = df["Total MACs (w^2+6w)"].tolist()

    # Dense curve for smooth plotting
    w_curve = np.linspace(16, 2100, 300)
    params_curve = w_curve**2 + 8 * w_curve + 1
    macs_curve = w_curve**2 + 6 * w_curve

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Plot lines
    ax.plot(w_curve, params_curve, color="#1f77b4", linewidth=2.2, label=r"Total Parameters: $P = w^2 + 8w + 1$")
    ax.plot(w_curve, macs_curve, color="#ff7f0e", linestyle="--", linewidth=1.8, label=r"Total MACs: $MAC = w^2 + 6w$")

    # Scatter points for the 5 benchmark widths
    ax.scatter(widths, total_params, color="#1f77b4", s=70, zorder=5, edgecolors="black")
    ax.scatter(widths, total_macs, color="#ff7f0e", s=60, zorder=5, edgecolors="black")

    # Annotate points
    for w, p in zip(widths, total_params):
        offset = (15, 10) if w < 1000 else (-120, -15)
        ax.annotate(
            f"w={w}\n{p:,} params",
            xy=(w, p),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f4f8", ec="#b0c4de", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
        )

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Hidden Layer Width $w$ (log scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count (log scale)", fontsize=11, fontweight="bold")
    ax.set_title("Mission 4: Neural Network Parameters & MACs vs Width\nArchitecture: 5 -> w -> w -> 1", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="both")
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_f32_vs_int8_memory(output_path: str = "edge_analysis/f32_vs_int8_memory.png") -> str:
    """
    Generate graph comparing Float32, Realistic Int8, and Naive Int8 memory footprints.
    """
    set_plot_style()
    df = compare_memory_across_widths()
    widths = df["Width (w)"].tolist()
    f32_kib = df["F32 Total KiB"].tolist()
    real_int8_kib = df["Real Int8 Total KiB"].tolist()
    naive_int8_kib = df["Naive Int8 KiB"].tolist()

    x = np.arange(len(widths))
    bar_width = 0.26

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    rects1 = ax.bar(x - bar_width, f32_kib, bar_width, label="Float32 (4 bytes/param)", color="#d62728", alpha=0.9)
    rects2 = ax.bar(x, real_int8_kib, bar_width, label="Realistic INT8 (Int8 W + Int32 B + Meta)", color="#2ca02c", alpha=0.9)
    rects3 = ax.bar(x + bar_width, naive_int8_kib, bar_width, label="Naive INT8 (1 byte/param)", color="#1f77b4", alpha=0.7)

    # Annotate bar values
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:,.1f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, rotation=45)

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:,.1f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, rotation=45)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"w={w}\n({df.loc[i, 'Total Params']:,} p)" for i, w in enumerate(widths)], fontsize=9.5)
    ax.set_xlabel("Hidden Width $w$ and Total Parameter Count", fontsize=11, fontweight="bold")
    ax.set_ylabel("Model Memory (KiB, log scale)", fontsize=11, fontweight="bold")
    ax.set_title("Float32 vs INT8 Model Memory Footprint Across Benchmark Widths", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="both", axis="y")
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_memory_vs_sram_limit(output_path: str = "edge_analysis/memory_vs_sram_limit.png") -> str:
    """
    Generate graph of memory usage versus the 300 KB SRAM limit with shaded feasible regions.
    """
    set_plot_style()
    cutoffs = calculate_exact_sram_cutoff_widths(SRAM_LIMIT_BINARY_BYTES)
    f32_cutoff = cutoffs["f32_total"]["max_integer_width"]      # 273
    int8_cutoff = cutoffs["int8_realistic"]["max_integer_width"] # 550

    # Dense curve of widths
    w_vals = np.linspace(16, 800, 400)
    # Total inference memory (Model + Peak Activation) in KiB
    # Float32: 4*(w^2 + 8w + 1) + 4*(2w) = 4w^2 + 40w + 4 bytes
    f32_kib = (4 * w_vals**2 + 40 * w_vals + 4) / 1024.0
    # Int8 Realistic: (w^2 + 6w) + 4*(2w + 1) + 24 + 2w = w^2 + 16w + 28 bytes
    int8_kib = (w_vals**2 + 16 * w_vals + 28) / 1024.0

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)

    # 300 KB limit line
    sram_limit_kib = 300.0  # 307.2 KiB
    ax.axhline(sram_limit_kib, color="#d62728", linestyle="--", linewidth=2.0, label="300 KB SRAM Evaluation Limit (307.2 KiB)")

    # System-aware remaining ML limit (after 120 KB WiFi/RTOS overhead)
    sys_ml_limit_kib = (SRAM_LIMIT_BINARY_BYTES - SYSTEM_OVERHEAD_WIFI_BYTES) / 1024.0  # ~187.2 KiB
    ax.axhline(sys_ml_limit_kib, color="#8c564b", linestyle=":", linewidth=1.6, label="Available ML SRAM with WiFi/RTOS (187.2 KiB)")

    # Shaded feasible zones
    ax.fill_between(w_vals, 0, sram_limit_kib, color="#e6f5d0", alpha=0.4, label="SRAM Feasible Region (< 300 KB)")

    # Plot memory curves
    ax.plot(w_vals, f32_kib, color="#d62728", linewidth=2.2, label="Float32 Total Inference Memory (Model + Activations)")
    ax.plot(w_vals, int8_kib, color="#2ca02c", linewidth=2.2, label="INT8 Realistic Inference Memory (Model + Activations)")

    # Vertical cutoff markers
    ax.axvline(f32_cutoff, color="#d62728", linestyle="-.", linewidth=1.4, alpha=0.8)
    ax.text(f32_cutoff - 8, 450, f"Float32 Cutoff\nw = {f32_cutoff}", color="#d62728", fontsize=9, fontweight="bold", ha="right")

    ax.axvline(int8_cutoff, color="#2ca02c", linestyle="-.", linewidth=1.4, alpha=0.8)
    ax.text(int8_cutoff + 8, 450, f"INT8 Cutoff\nw = {int8_cutoff}", color="#2ca02c", fontsize=9, fontweight="bold", ha="left")

    # Scatter points for benchmark widths 32, 128, 256, 512
    bench_w = [32, 128, 256, 512]
    bench_f32 = [(4 * w**2 + 40 * w + 4) / 1024.0 for w in bench_w]
    bench_int8 = [(w**2 + 16 * w + 28) / 1024.0 for w in bench_w]

    ax.scatter(bench_w, bench_f32, color="#d62728", s=65, zorder=6, edgecolors="black")
    ax.scatter(bench_w, bench_int8, color="#2ca02c", s=65, zorder=6, edgecolors="black")

    for w, f, i in zip(bench_w, bench_f32, bench_int8):
        status_f = "FITS" if f <= sram_limit_kib else "EXCEEDS"
        status_i = "FITS" if i <= sram_limit_kib else "EXCEEDS"
        ax.annotate(f"w={w}\nF32: {f:.1f} KiB ({status_f})\nINT8: {i:.1f} KiB ({status_i})",
                    xy=(w, max(f, i)), xytext=(0, 15), textcoords="offset points",
                    ha="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999", alpha=0.85))

    ax.set_xlim(16, 750)
    ax.set_ylim(0, 600)
    ax.set_xlabel("Hidden Layer Width $w$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Memory Footprint (KiB)", fontsize=11, fontweight="bold")
    ax.set_title("ESP32-S3 SRAM Feasibility & Exact Cutoff Boundaries\n300 KB Limit vs Total Inference Memory", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_inference_latency(output_path: str = "edge_analysis/inference_latency_vs_width.png") -> str:
    """
    Generate graph comparing ESP32-S3 inference latency (Float32 vs INT8 SIMD) across widths.
    """
    set_plot_style()
    df = compare_inference_timing_across_five_widths()
    widths = df["Width (w)"].tolist()
    int8_ms = df["INT8 SIMD (ms)"].tolist()
    f32_ms = df["Float32 DSP (ms)"].tolist()

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    ax.plot(widths, f32_ms, marker="o", color="#d62728", linewidth=2.0, markersize=7, label="Float32 FPU (ESP-DSP @ 24 MMAC/s)")
    ax.plot(widths, int8_ms, marker="s", color="#2ca02c", linewidth=2.0, markersize=7, label="INT8 SIMD (ESP-NN Vector @ 123.5 MMAC/s)")

    for w, f, i in zip(widths, f32_ms, int8_ms):
        speedup = f / i if i > 0 else 0
        ax.annotate(f"{speedup:.1f}x faster", xy=(w, i), xytext=(0, -18), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold", color="#2ca02c")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("Hidden Width $w$ (log scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Inference Time per Prediction (ms, log scale)", fontsize=11, fontweight="bold")
    ax.set_title("ESP32-S3 Estimated Inference Time vs Model Width\nXtensa LX7 @ 240 MHz (ESP-NN SIMD vs Float32)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="both")
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_activation_memory(output_path: str = "edge_analysis/activation_memory.png") -> str:
    """
    Generate graph comparing peak live activation memory vs parameter memory.
    """
    set_plot_style()
    df_act = compare_activation_memory_across_widths()
    df_mem = compare_memory_across_widths()

    widths = df_act["Width (w)"].tolist()
    act_f32_bytes = df_act["Peak Live F32 (Bytes)"].tolist()
    param_f32_bytes = df_mem["F32 Total Bytes"].tolist()

    x = np.arange(len(widths))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    ax.bar(x - bar_width/2, act_f32_bytes, bar_width, label="Peak Live Activation Memory (Float32)", color="#17becf", alpha=0.9)
    ax.bar(x + bar_width/2, param_f32_bytes, bar_width, label="Static Parameter Memory (Float32)", color="#ff7f0e", alpha=0.9)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"w={w}" for w in widths], fontsize=10)
    ax.set_xlabel("Hidden Width $w$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Memory in Bytes (log scale)", fontsize=11, fontweight="bold")
    ax.set_title("Comparison of Peak Live Activation Memory vs Static Parameter Memory\n(Sequential Layer Execution with Double Buffer Reuse)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="both", axis="y")
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def generate_all_visualizations(output_dir: str = "edge_analysis") -> Dict[str, str]:
    """
    Generate all required visualization graphics and return their file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "params_vs_width": plot_parameters_vs_width(os.path.join(output_dir, "params_vs_width.png")),
        "f32_vs_int8_memory": plot_f32_vs_int8_memory(os.path.join(output_dir, "f32_vs_int8_memory.png")),
        "memory_vs_sram_limit": plot_memory_vs_sram_limit(os.path.join(output_dir, "memory_vs_sram_limit.png")),
        "inference_latency": plot_inference_latency(os.path.join(output_dir, "inference_latency_vs_width.png")),
        "activation_memory": plot_activation_memory(os.path.join(output_dir, "activation_memory.png")),
    }
    return paths


if __name__ == "__main__":
    print("Generating visualizations...")
    saved = generate_all_visualizations()
    for k, p in saved.items():
        print(f"  Saved: {k} -> {p}")
