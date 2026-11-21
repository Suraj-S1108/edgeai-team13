"""
run_analysis.py
===============
Master orchestration script for Mission 4 Edge AI Analysis.
Executes the full pipeline:
  1. Parameter calculations across widths
  2. Float32 and INT8 memory analysis
  3. Activation buffer analysis
  4. ESP32-S3 SRAM & PSRAM feasibility
  5. Inference timing estimation
  6. Server vs Edge deployment trade-offs
  7. Active repository model inspection & dynamic INT8 quantization
  8. Publication-quality visual graph generation
  9. Structured CSV dataset exports
 10. Technical Markdown report & summary generation

Usage:
------
    python -m edge_analysis.run_analysis
    python edge_analysis/run_analysis.py

Author: Nandish (Mission 4 - Edge Device & Memory Analysis)
"""

import sys
import os
import time

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from edge_analysis.parameter_calculator import compare_five_widths
from edge_analysis.memory_calculator import (
    compare_memory_across_widths,
    compare_activation_memory_across_widths,
)
from edge_analysis.feasibility import (
    compare_feasibility_across_five_widths,
    calculate_exact_sram_cutoff_widths,
    SRAM_LIMIT_BINARY_BYTES,
)
from edge_analysis.inference_estimator import compare_inference_timing_across_five_widths
from edge_analysis.deployment_comparison import (
    generate_deployment_comparison_table,
    calculate_energy_consumption_comparison,
)
from edge_analysis.model_analyzer import (
    load_repository_model_config,
    analyze_model_edge_metrics,
    perform_quantization_and_evaluation,
)
from edge_analysis.visualizer import generate_all_visualizations
from edge_analysis.reporter import run_all_reporting


def run_full_mission4_pipeline(output_dir: str = "edge_analysis"):
    """
    Execute entire Mission 4 analysis autonomously and print progress.
    """
    t0 = time.perf_counter()
    print("=" * 72)
    print("AI255TBD - EDGE AI - MODULE 1 - TEAM 13")
    print("MISSION 4: EDGE-DEVICE AND MEMORY-ANALYSIS PIPELINE")
    print("Owner: Nandish (Senior Edge AI & Neural Network Optimization Engineer)")
    print("Target Device: Espressif ESP32-S3 (Xtensa LX7 @ 240 MHz, 512KB SRAM, 8MB PSRAM)")
    print("=" * 72)

    # 1. Parameter Calculations
    print("\n[1/8] Computing Parameter & MAC Counts across Benchmark Widths...")
    df_params = compare_five_widths()
    print(df_params[["Width (w)", "Architecture", "Total Trainable Params (w^2+8w+1)", "Total MACs (w^2+6w)"]].to_string(index=False))

    # 2. Memory Analysis
    print("\n[2/8] Computing Float32 vs Realistic INT8 Memory Footprints...")
    df_mem = compare_memory_across_widths()
    print(df_mem[["Width (w)", "F32 Total KiB", "Real Int8 Total KiB", "Real Int8 Ratio", "Peak Act F32 (Bytes)"]].to_string(index=False))

    # 3. Activation Memory
    print("\n[3/8] Computing Activation Buffer Footprints with Sequential Double-Buffering...")
    df_act = compare_activation_memory_across_widths()
    print(df_act[["Width (w)", "Peak Live Elements", "Peak Live F32 (Bytes)", "Peak Live Int8 (Bytes)"]].to_string(index=False))

    # 4. SRAM & PSRAM Feasibility
    print("\n[4/8] Evaluating ESP32-S3 SRAM (300 KB limit) & PSRAM Feasibility...")
    df_feas = compare_feasibility_across_five_widths()
    print(df_feas[["Width (w)", "F32 Total (KiB)", "Fits 300KB F32", "Real Int8 (KiB)", "Fits 300KB Int8", "Fits 8MB PSRAM F32", "Fits 8MB PSRAM Int8"]].to_string(index=False))

    cutoffs = calculate_exact_sram_cutoff_widths(SRAM_LIMIT_BINARY_BYTES)
    print(f"  -> Exact Float32 Cutoff: w <= {cutoffs['f32_total']['max_integer_width']} (exact: {cutoffs['f32_total']['exact']})")
    print(f"  -> Exact INT8 Realistic Cutoff: w <= {cutoffs['int8_realistic']['max_integer_width']} (exact: {cutoffs['int8_realistic']['exact']})")
    print(f"  -> System-Aware Cutoff (with 120KB WiFi/RTOS): F32 w <= {cutoffs['system_aware_120kb_overhead']['f32_max_integer_width']}, INT8 w <= {cutoffs['system_aware_120kb_overhead']['int8_max_integer_width']}")

    # 5. Inference Timing
    print("\n[5/8] Estimating Hardware Inference Computation & Timing on ESP32-S3...")
    df_inf = compare_inference_timing_across_five_widths()
    print(df_inf[["Width (w)", "Total MACs", "INT8 SIMD (ms)", "Float32 DSP (ms)", "SIMD vs F32 Speedup", "INT8 PSRAM Bound"]].to_string(index=False))

    # 6. Deployment Trade-off & Battery Energy Math
    print("\n[6/8] Evaluating Cloud/Server vs ESP32-S3 Edge Deployment...")
    energy = calculate_energy_consumption_comparison()
    print(f"  -> Server Energy per Reading: {energy['energy_server_joules_per_reading']} Joules (Wi-Fi 220 mA @ 3.0s)")
    print(f"  -> Edge Energy per Reading:   {energy['energy_edge_joules_per_reading']} Joules (MCU 40 mA @ 2.0ms)")
    print(f"  -> Energy Reduction Factor:   {energy['energy_reduction_factor']:,}x LESS ENERGY on Edge!")
    print(f"  -> Battery Lifespan (3000mAh, Hourly Readings): Server = {energy['theoretical_days_on_battery_server']} days vs Edge = {energy['theoretical_days_on_battery_edge']} days")

    # 7. Model Analysis & Actual Dynamic Quantization
    print("\n[7/8] Analyzing Active Repository Model & Performing Actual INT8 Quantization...")
    cfg = load_repository_model_config()
    print(f"  -> Active Model Team: {cfg['team']}, Hidden: {cfg['hidden']}, Status: {cfg['model_status']}")
    metrics = analyze_model_edge_metrics(cfg)
    print(f"  -> Model Params: {metrics['total_params']}, Float32 Size: {metrics['total_bytes_f32']} B ({metrics['total_kib_f32']} KiB)")
    print(f"  -> Fits 300KB SRAM (F32): {metrics['fits_300kb_sram_f32']}, Fits 300KB SRAM (INT8): {metrics['fits_300kb_sram_int8']}")

    q_res = perform_quantization_and_evaluation(output_dir=output_dir)
    print(f"  -> Validation MAE (Float32 Baseline): {q_res['mae_float32']:.4f} days")
    print(f"  -> Validation MAE (Quantized INT8):    {q_res['mae_int8']:.4f} days")
    print(f"  -> MAE Delta:                         {q_res['mae_delta']:+.4f} days ({q_res['percent_mae_change']:+.2f}%)")
    print(f"  -> Verified Artifact Saved:           {q_res['artifact_json_path']}")

    # 8. Visualizations & Deliverables
    print("\n[8/8] Rendering Publication Graphs, CSVs, and Technical Reports...")
    graphs = generate_all_visualizations(output_dir)
    for name, path in graphs.items():
        print(f"  [Graph] {name} -> {path}")

    reports = run_all_reporting(output_dir)
    for name, path in reports.items():
        print(f"  [Deliverable] {name} -> {path}")

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 72)
    print(f"MISSION 4 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print(f"All deliverables are ready for review in: {os.path.abspath(output_dir)}")
    print("=" * 72)


if __name__ == "__main__":
    run_full_mission4_pipeline()

