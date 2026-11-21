"""
Unit tests for memory_calculator.py
"""

import pytest
from edge_analysis.parameter_calculator import calculate_two_hidden_mlp_formulas
from edge_analysis.memory_calculator import (
    calculate_float32_memory,
    calculate_int8_memory,
    calculate_activation_memory,
    compare_memory_across_widths,
    compare_activation_memory_across_widths,
)


def test_float32_memory_calculation():
    # w = 32: W = 1216, B = 65, P = 1281
    # Weight bytes = 1216 * 4 = 4864
    # Bias bytes = 65 * 4 = 260
    # Total bytes = 5124
    f32 = calculate_float32_memory(1216, 65)
    assert f32["weight_bytes"] == 4864
    assert f32["bias_bytes"] == 260
    assert f32["total_bytes"] == 5124
    assert f32["total_kb"] == 5.124
    assert round(f32["total_kib"], 4) == round(5124 / 1024.0, 4)


def test_int8_memory_bias_representation():
    # Verify that realistic embedded INT8 accounts for 4-byte INT32 biases
    int8_res = calculate_int8_memory(total_weights=1216, total_biases=65, num_layers=3, include_metadata=True)
    naive = int8_res["naive_int8"]
    real = int8_res["realistic_int8"]

    assert naive["total_bytes"] == 1281 * 1
    assert real["weight_bytes"] == 1216 * 1
    assert real["bias_bytes"] == 65 * 4  # INT32 biases = 260 bytes!
    assert real["metadata_bytes"] == 3 * 8  # 24 bytes metadata
    assert real["total_bytes"] == 1216 + 260 + 24  # = 1500 bytes (aligned)
    assert real["total_bytes"] > naive["total_bytes"]  # Realistic > naive due to int32 biases


def test_activation_memory_peak_live():
    # w = 32: Layer dims: 5 -> 32 -> 32 -> 1
    # Step live:
    # 5 + 32 = 37
    # 32 + 32 = 64
    # 32 + 1 = 33
    # Peak live elements = 64 elements
    act32 = calculate_activation_memory(input_dim=5, hidden_widths=(32, 32), output_dim=1, batch_size=1)
    assert act32["peak_live_elements"] == 64
    assert act32["float32"]["peak_live_bytes"] == 64 * 4  # 256 bytes
    assert act32["int8"]["peak_live_bytes"] == 64 * 1     # 64 bytes

    # w = 256: peak live = 2 * 256 = 512 elements
    act256 = calculate_activation_memory(input_dim=5, hidden_widths=(256, 256), output_dim=1, batch_size=1)
    assert act256["peak_live_elements"] == 512
    assert act256["float32"]["peak_live_bytes"] == 512 * 4  # 2048 bytes (2.0 KiB)
    assert act256["int8"]["peak_live_bytes"] == 512 * 1     # 512 bytes


def test_compare_memory_dataframe_integrity():
    df = compare_memory_across_widths()
    assert len(df) == 5
    assert "F32 Total Bytes" in df.columns
    assert "Real Int8 Total Bytes" in df.columns
    assert "Real Int8 Ratio" in df.columns
    # Check that compression ratio is between 3.0 and 4.0
    for ratio in df["Real Int8 Ratio"]:
        assert 3.0 <= ratio <= 4.0


def test_invalid_memory_inputs():
    with pytest.raises(ValueError):
        calculate_float32_memory(-10, 5)
    with pytest.raises(ValueError):
        calculate_int8_memory(100, -5)
    with pytest.raises(ValueError):
        calculate_activation_memory(0, (32, 32), 1)

