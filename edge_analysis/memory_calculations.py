"""
edge_analysis/memory_calculations.py -- Edge AI Resource and Memory Calculations
EDGE AI Team 13 - Battery Life Prediction

Calculates and compares:
1. Parameter counts (weights, biases, total)
2. Theoretical parameter storage (FP32, FP16, INT8)
3. Input, output, and activation tensor memory
4. MACs (Multiply-Accumulate operations) and FLOPs
5. Checkpoint serialization size on disk
6. ESP32-S3 SRAM (300 KiB) and PSRAM (8 MiB) budget compliance
7. Bare-metal Microcontroller runtime arena estimates vs CPython runtime
8. Hidden layer width scaling comparison
"""

import csv
import json
import os
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model" / "config.json"
BACKUP_CONFIG_PATH = ROOT / "model" / "backup_16_16" / "config.json"
MODEL_PATH = ROOT / "model" / "model.pt"
BACKUP_MODEL_PATH = ROOT / "model" / "backup_16_16" / "model.pt"
OUTPUT_WIDTH_CSV = ROOT / "edge_analysis" / "width_comparison.csv"
OUTPUT_COMPARE_CSV = ROOT / "edge_analysis" / "model_memory_comparison.csv"

# ESP32-S3 Hardware Budgets
SRAM_BUDGET = 300 * 1024         # 300 KiB (307,200 bytes)
PSRAM_BUDGET = 8 * 1024 * 1024   # 8 MiB (8,388,608 bytes)


def calculate_architecture_details(hidden_widths, model_file=None):
    """
    Detailed resource breakdown for architecture: 5 -> hidden_widths -> 1
    """
    layer_sizes = [5, *hidden_widths, 1]

    layer_breakdown = []
    total_weights = 0
    total_biases = 0
    total_macs = 0
    max_activation_dim = max(hidden_widths)

    for idx, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]), 1):
        weights = in_dim * out_dim
        biases = out_dim
        params = weights + biases
        macs = weights

        total_weights += weights
        total_biases += biases
        total_macs += macs

        layer_breakdown.append({
            "layer": f"Linear_{idx}",
            "in": in_dim,
            "out": out_dim,
            "weights": weights,
            "biases": biases,
            "params": params,
            "macs": macs,
            "fp32_bytes": params * 4,
            "int8_bytes": params * 1
        })

    total_params = total_weights + total_biases
    fp32_bytes = total_params * 4
    fp16_bytes = total_params * 2
    int8_bytes = total_params * 1

    # Tensor buffers (FP32)
    input_tensor_bytes = 5 * 4
    output_tensor_bytes = 1 * 4
    peak_activation_bytes = max_activation_dim * 4

    # Microcontroller inference arena estimate (weights + ping-pong buffers + scratchpad)
    # Ping-pong buffer = 2 * peak_activation_bytes + scratch
    bare_metal_arena_fp32 = fp32_bytes + (2 * peak_activation_bytes) + 256
    bare_metal_arena_int8 = int8_bytes + (2 * max_activation_dim) + 256

    # Checkpoint size on disk
    checkpoint_file_size = None
    if model_file and os.path.exists(model_file):
        try:
            checkpoint_file_size = os.path.getsize(model_file)
        except OSError:
            checkpoint_file_size = None

    return {
        "architecture": " -> ".join(map(str, layer_sizes)),
        "hidden_layers": len(hidden_widths),
        "total_weights": total_weights,
        "total_biases": total_biases,
        "total_params": total_params,
        "fp32_bytes": fp32_bytes,
        "fp16_bytes": fp16_bytes,
        "int8_bytes": int8_bytes,
        "input_tensor_bytes": input_tensor_bytes,
        "output_tensor_bytes": output_tensor_bytes,
        "peak_activation_bytes": peak_activation_bytes,
        "total_macs": total_macs,
        "total_flops": total_macs * 2,
        "checkpoint_file_size": checkpoint_file_size,
        "bare_metal_arena_fp32": bare_metal_arena_fp32,
        "bare_metal_arena_int8": bare_metal_arena_int8,
        "sram_fp32_utilization_pct": round((fp32_bytes / SRAM_BUDGET) * 100, 3),
        "sram_int8_utilization_pct": round((int8_bytes / SRAM_BUDGET) * 100, 3),
        "fits_sram_fp32": fp32_bytes <= SRAM_BUDGET,
        "fits_sram_int8": int8_bytes <= SRAM_BUDGET,
        "fits_psram_fp32": fp32_bytes <= PSRAM_BUDGET,
        "layer_breakdown": layer_breakdown
    }


def main():
    print("=================================================================")
    print("  EDGE AI Team 13 — Memory and Resource Calculation Engine")
    print("=================================================================\n")

    # 1. Compare Deployed 32-32 with Legacy 16-16
    deployed_res = calculate_architecture_details([32, 32], MODEL_PATH)
    legacy_res = calculate_architecture_details([16, 16], BACKUP_MODEL_PATH)

    models_comparison = [
        {"Model": "Legacy Baseline (16-16)", **{k: v for k, v in legacy_res.items() if k != "layer_breakdown"}},
        {"Model": "Deployed Candidate (32-32)", **{k: v for k, v in deployed_res.items() if k != "layer_breakdown"}}
    ]

    print("--- MODEL COMPARISON: 16-16 vs 32-32 ---")
    for m in models_comparison:
        print(f"\nModel: {m['Model']}")
        print(f"  Architecture:           {m['architecture']}")
        print(f"  Weights:                {m['total_weights']}")
        print(f"  Biases:                 {m['total_biases']}")
        print(f"  Total Parameters:       {m['total_params']}")
        print(f"  FP32 Parameter Memory:  {m['fp32_bytes']} B ({m['fp32_bytes']/1024:.2f} KiB)")
        print(f"  FP16 Parameter Memory:  {m['fp16_bytes']} B ({m['fp16_bytes']/1024:.2f} KiB)")
        print(f"  INT8 Parameter Memory:  {m['int8_bytes']} B ({m['int8_bytes']/1024:.2f} KiB)")
        print(f"  Peak Activation Memory: {m['peak_activation_bytes']} B")
        print(f"  MACs / FLOPs:           {m['total_macs']} MACs / {m['total_flops']} FLOPs")
        if m['checkpoint_file_size'] is not None:
            ckpt_size_kib = m['checkpoint_file_size'] / 1024
            print(f"  Checkpoint File Size:   {m['checkpoint_file_size']} B ({ckpt_size_kib:.2f} KiB)")
        else:
            print("  Checkpoint File Size:   Not available")
        print(f"  Microcontroller Arena:  {m['bare_metal_arena_fp32']} B (FP32) | {m['bare_metal_arena_int8']} B (INT8)")
        print(f"  SRAM Budget (300 KiB):  {m['sram_fp32_utilization_pct']}% utilized (Fits: {m['fits_sram_fp32']})")

    # Save comparison CSV
    with open(OUTPUT_COMPARE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(models_comparison[0].keys()))
        writer.writeheader()
        writer.writerows(models_comparison)
    print(f"\nSaved model memory comparison to: {OUTPUT_COMPARE_CSV}")

    # 2. Width Scaling Analysis
    print("\n--- WIDTH SCALING ANALYSIS (Two Hidden Layers: 5 -> W -> W -> 1) ---")
    widths = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    width_rows = []

    for w in widths:
        res = calculate_architecture_details([w, w])
        row = {
            "width": w,
            "architecture": res["architecture"],
            "params": res["total_params"],
            "float32_bytes": res["fp32_bytes"],
            "int8_bytes": res["int8_bytes"],
            "activation_bytes": res["peak_activation_bytes"],
            "macs": res["total_macs"],
            "fits_sram_float32_weights": res["fits_sram_fp32"],
            "fits_sram_int8_weights": res["fits_sram_int8"],
            "fits_psram_float32_weights": res["fits_psram_fp32"],
            "sram_utilization_pct": res["sram_fp32_utilization_pct"]
        }
        width_rows.append(row)
        print(f"  Width {w:4d}: {res['total_params']:8d} params | FP32: {res['fp32_bytes']/1024:8.2f} KiB | SRAM: {res['sram_fp32_utilization_pct']:6.2f}% | Fits: {res['fits_sram_fp32']}")

    with open(OUTPUT_WIDTH_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(width_rows[0].keys()))
        writer.writeheader()
        writer.writerows(width_rows)
    print(f"Saved width comparison to: {OUTPUT_WIDTH_CSV}")

    # 3. Exact Width Limits for 300 KiB SRAM
    def param_formula(w):
        return w * w + 8 * w + 1

    max_w_fp32 = max(w for w in range(1, 10000) if param_formula(w) * 4 <= SRAM_BUDGET)
    max_w_int8 = max(w for w in range(1, 10000) if param_formula(w) <= SRAM_BUDGET)

    print("\n--- EXACT MAXIMUM EQUAL-WIDTH FOR 300 KiB SRAM ---")
    print(f"  Maximum Width (FP32 weights): {max_w_fp32} ({param_formula(max_w_fp32)*4} B / {SRAM_BUDGET} B)")
    print(f"  Maximum Width (INT8 weights): {max_w_int8} ({param_formula(max_w_int8)} B / {SRAM_BUDGET} B)")
    print("\nCalculations completed successfully.")


if __name__ == "__main__":
    main()
