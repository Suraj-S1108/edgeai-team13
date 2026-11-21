"""
Unit tests for inference_estimator.py
"""

import pytest
from edge_analysis.inference_estimator import (
    estimate_layer_inference_time,
    estimate_inference_for_width,
    compare_inference_timing_across_five_widths,
    generate_c_measurement_harness,
    THROUGHPUT_BENCHMARKS,
)


def test_throughput_benchmarks_presence():
    assert "int8_simd_esp_nn" in THROUGHPUT_BENCHMARKS
    assert "float32_esp_dsp" in THROUGHPUT_BENCHMARKS
    assert THROUGHPUT_BENCHMARKS["int8_simd_esp_nn"]["throughput_mmacs"] == 123.5
    assert THROUGHPUT_BENCHMARKS["float32_esp_dsp"]["throughput_mmacs"] == 24.0


def test_estimate_layer_inference_time():
    # 123.5 MMACs/sec: 123,500,000 MACs take 1 second
    # For 123,500 MACs -> 0.001 s (1 ms)
    res = estimate_layer_inference_time(123_500, "int8_simd_esp_nn", use_psram=False)
    assert abs(res["time_ms"] - 1.0) < 0.01

    # With PSRAM penalty (3.2x)
    res_psram = estimate_layer_inference_time(123_500, "int8_simd_esp_nn", use_psram=True)
    assert abs(res_psram["time_ms"] - 3.2) < 0.05


def test_estimate_inference_for_width_32():
    # w = 32: MACs = 1216
    # 1216 / 123.5e6 = 9.846 us = 0.00985 ms
    res32 = estimate_inference_for_width(32)
    assert res32["total_macs"] == 1216
    assert abs(res32["int8_simd_us"] - 9.85) < 0.1
    assert abs(res32["int8_simd_ms"] - 0.0098) < 0.001
    assert res32["simd_vs_f32_speedup"] > 5.0


def test_compare_inference_timing_across_five_widths_table():
    df = compare_inference_timing_across_five_widths()
    assert len(df) == 5
    assert "INT8 SIMD (ms)" in df.columns
    assert "Float32 DSP (ms)" in df.columns
    assert "INT8 PSRAM Bound" in df.columns
    # Check monotonic increase in execution time
    times = df["INT8 SIMD (ms)"].tolist()
    assert all(times[i] < times[i+1] for i in range(len(times)-1))


def test_c_measurement_harness_generation():
    code = generate_c_measurement_harness()
    assert "esp_timer_get_time()" in code
    assert "esp_nn.h" in code
    assert "run_inference_benchmark" in code

