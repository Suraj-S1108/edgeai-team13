"""
Unit tests for feasibility.py
"""

import pytest
from edge_analysis.feasibility import (
    evaluate_width_feasibility,
    calculate_exact_sram_cutoff_widths,
    compare_feasibility_across_five_widths,
    SRAM_LIMIT_BINARY_BYTES,
    SRAM_LIMIT_DECIMAL_BYTES,
)


def test_feasibility_decisions_five_widths():
    # w = 32
    res32 = evaluate_width_feasibility(32)
    assert res32["fits_300kb_sram_f32"] is True
    assert res32["fits_300kb_sram_int8"] is True
    assert res32["fits_8mb_psram_f32"] is True

    # w = 128
    res128 = evaluate_width_feasibility(128)
    assert res128["fits_300kb_sram_f32"] is True
    assert res128["fits_300kb_sram_int8"] is True
    assert res128["fits_8mb_psram_f32"] is True

    # w = 256
    # F32 total = 270,340 bytes < 307,200 bytes (fits in isolated 300KB)
    res256 = evaluate_width_feasibility(256)
    assert res256["fits_300kb_sram_f32"] is True
    assert res256["fits_300kb_sram_int8"] is True
    assert res256["fits_8mb_psram_f32"] is True

    # w = 512
    # F32 total = 1,064,964 bytes > 307,200 bytes (FAILS F32)
    # INT8 real = 269,340 bytes < 307,200 bytes (FITS INT8)
    res512 = evaluate_width_feasibility(512)
    assert res512["fits_300kb_sram_f32"] is False
    assert res512["fits_300kb_sram_int8"] is True
    assert res512["fits_8mb_psram_f32"] is True

    # w = 2048
    # F32 total = 16,842,756 bytes > 8 MB PSRAM (FAILS 8MB PSRAM in F32!)
    # INT8 real = 4,223,004 bytes < 8 MB PSRAM (FITS 8MB PSRAM in INT8!)
    res2048 = evaluate_width_feasibility(2048)
    assert res2048["fits_300kb_sram_f32"] is False
    assert res2048["fits_300kb_sram_int8"] is False
    assert res2048["fits_8mb_psram_f32"] is False  # CRITICAL FINDING!
    assert res2048["fits_8mb_psram_int8"] is True


def test_exact_sram_cutoff_widths():
    cutoffs_bin = calculate_exact_sram_cutoff_widths(SRAM_LIMIT_BINARY_BYTES)
    # Binary limit 307,200 bytes:
    # 4(w^2 + 8w + 1) <= 307,200 -> w = 273
    assert cutoffs_bin["f32_total"]["max_integer_width"] == 273

    # INT8 realistic: w = 547
    assert cutoffs_bin["int8_realistic"]["max_integer_width"] == 547

    # Decimal limit 300,000 bytes:
    cutoffs_dec = calculate_exact_sram_cutoff_widths(SRAM_LIMIT_DECIMAL_BYTES)
    assert cutoffs_dec["f32_total"]["max_integer_width"] == 269
    assert cutoffs_dec["int8_realistic"]["max_integer_width"] == 540


def test_compare_feasibility_across_five_widths_table():
    df = compare_feasibility_across_five_widths()
    assert len(df) == 5
    assert "Fits 300KB F32" in df.columns
    assert "Fits 300KB Int8" in df.columns
    assert "Fits 8MB PSRAM F32" in df.columns
    assert df.loc[df["Width (w)"] == 256, "Fits 300KB F32"].values[0] == "YES"
    assert df.loc[df["Width (w)"] == 512, "Fits 300KB F32"].values[0] == "NO"
    assert df.loc[df["Width (w)"] == 512, "Fits 300KB Int8"].values[0] == "YES"
    assert df.loc[df["Width (w)"] == 2048, "Fits 8MB PSRAM F32"].values[0] == "NO"
    assert df.loc[df["Width (w)"] == 2048, "Fits 8MB PSRAM Int8"].values[0] == "YES"

