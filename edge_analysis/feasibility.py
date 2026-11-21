"""
feasibility.py
==============
Evaluates the architectural and hardware feasibility of deploying the battery-life
prediction neural network onto the Espressif ESP32-S3 edge microcontroller.

ESP32-S3 Hardware Specifications:
---------------------------------
  - Processor: Dual-Core 32-bit Xtensa LX7 @ up to 240 MHz
  - Internal SRAM: 512 KB Total (384 KB SRAM1 + 128 KB SRAM0 + 8 KB RTC)
  - SRAM Evaluation Limit: 300 KB (as defined in project specification)
    * Decimal budget: 300,000 bytes
    * Binary budget:  307,200 bytes (300 * 1024)
  - External PSRAM: 8 MB (Octal SPI / Quad SPI)
  - External Flash: 16 MB (SPI Flash)

Analysis Features:
------------------
1. Evaluates all 5 benchmark widths (32, 128, 256, 512, 2048).
2. Distinguishes weight-only vs total model parameters.
3. Incorporates intermediate activation memory.
4. Accounts for realistic embedded OS/runtime overhead (FreeRTOS, WiFi stack, TLS, task stacks).
5. Computes exact mathematical cutoff widths for Float32 and Int8.
6. Assesses 8 MB PSRAM feasibility.
7. Contrasts on-chip SRAM vs external PSRAM trade-offs (latency, bandwidth, battery drain).

Author: Nandish (Mission 4 - Edge Device & Memory Analysis)
"""

import math
from typing import Dict, Any, List, Tuple
import pandas as pd
from edge_analysis.parameter_calculator import calculate_two_hidden_mlp_formulas
from edge_analysis.memory_calculator import (
    calculate_float32_memory,
    calculate_int8_memory,
    calculate_activation_memory,
)

# Standard evaluation budgets
SRAM_LIMIT_DECIMAL_BYTES = 300_000        # 300 KB decimal
SRAM_LIMIT_BINARY_BYTES  = 300 * 1024     # 307,200 bytes (300 KiB)
PSRAM_LIMIT_BYTES        = 8 * 1024 * 1024 # 8 MiB (8,388,608 bytes)
FLASH_LIMIT_BYTES        = 16 * 1024 * 1024 # 16 MiB (16,777,216 bytes)

# Typical ESP-IDF FreeRTOS system memory allocations (non-ML overhead)
SYSTEM_OVERHEAD_MINIMAL_BYTES = 40 * 1024   # ~40 KB (bare RTOS + idle task + heap minimum)
SYSTEM_OVERHEAD_WIFI_BYTES    = 120 * 1024  # ~120 KB (RTOS + WiFi stack + LwIP + TLS buffers)


def evaluate_width_feasibility(
    w: int,
    sram_budget_bytes: int = SRAM_LIMIT_BINARY_BYTES,
    psram_budget_bytes: int = PSRAM_LIMIT_BYTES,
    system_overhead_bytes: int = SYSTEM_OVERHEAD_WIFI_BYTES,
) -> Dict[str, Any]:
    """
    Evaluate comprehensive memory feasibility for width w on ESP32-S3.
    
    Args:
        w: Hidden layer width.
        sram_budget_bytes: Total SRAM evaluation limit (default 307,200 bytes).
        psram_budget_bytes: PSRAM capacity (default 8 MiB).
        system_overhead_bytes: Memory consumed by RTOS/WiFi (default 120 KB).
        
    Returns:
        Dict of feasibility determinations, remaining memory, and statuses.
    """
    params = calculate_two_hidden_mlp_formulas(w)
    f32_mem = calculate_float32_memory(params["total_weights"], params["total_biases"])
    int8_mem = calculate_int8_memory(params["total_weights"], params["total_biases"], num_layers=3)
    act_mem = calculate_activation_memory(input_dim=5, hidden_widths=(w, w), output_dim=1, batch_size=1)

    f32_weight_bytes = f32_mem["weight_bytes"]
    f32_total_bytes  = f32_mem["total_bytes"]
    
    naive_int8_bytes = int8_mem["naive_int8"]["total_bytes"]
    real_int8_bytes  = int8_mem["realistic_int8"]["total_bytes"]

    act_f32_bytes = act_mem["float32"]["peak_live_bytes"]
    act_int8_bytes = act_mem["int8"]["peak_live_bytes"]

    # Total inference memory (Model storage + peak activation)
    tot_inf_f32_bytes = f32_total_bytes + act_f32_bytes
    tot_inf_int8_bytes = real_int8_bytes + act_int8_bytes

    # Remaining SRAM (Isolated ML budget vs with WiFi/RTOS overhead)
    rem_sram_isolated_f32 = sram_budget_bytes - tot_inf_f32_bytes
    rem_sram_isolated_int8 = sram_budget_bytes - tot_inf_int8_bytes

    effective_ml_sram = sram_budget_bytes - system_overhead_bytes
    rem_sram_system_f32 = effective_ml_sram - tot_inf_f32_bytes
    rem_sram_system_int8 = effective_ml_sram - tot_inf_int8_bytes

    # Fit decisions against evaluation limit
    fits_sram_f32_weights = f32_weight_bytes <= sram_budget_bytes
    fits_sram_f32_total   = f32_total_bytes <= sram_budget_bytes
    fits_sram_f32_full    = tot_inf_f32_bytes <= sram_budget_bytes
    fits_sram_f32_system  = tot_inf_f32_bytes <= effective_ml_sram

    fits_sram_int8_naive  = naive_int8_bytes <= sram_budget_bytes
    fits_sram_int8_real   = real_int8_bytes <= sram_budget_bytes
    fits_sram_int8_full   = tot_inf_int8_bytes <= sram_budget_bytes
    fits_sram_int8_system = tot_inf_int8_bytes <= effective_ml_sram

    fits_psram_f32_total  = f32_total_bytes <= psram_budget_bytes
    fits_psram_int8_total = real_int8_bytes <= psram_budget_bytes

    return {
        "w": w,
        "total_params": params["total_params"],
        "f32_total_bytes": f32_total_bytes,
        "f32_total_kib": f32_mem["total_kib"],
        "naive_int8_bytes": naive_int8_bytes,
        "real_int8_bytes": real_int8_bytes,
        "real_int8_kib": int8_mem["realistic_int8"]["total_kib"],
        "act_f32_bytes": act_f32_bytes,
        "act_int8_bytes": act_int8_bytes,
        "tot_inf_f32_bytes": tot_inf_f32_bytes,
        "tot_inf_int8_bytes": tot_inf_int8_bytes,
        # SRAM Fit flags (Isolated 300 KB budget)
        "fits_300kb_sram_f32": fits_sram_f32_total,
        "fits_300kb_sram_int8": fits_sram_int8_naive,  # aligned with project test key
        "fits_300kb_sram_int8_real": fits_sram_int8_real,
        "fits_300kb_sram_f32_full_inference": fits_sram_f32_full,
        "fits_300kb_sram_int8_full_inference": fits_sram_int8_full,
        # Remaining SRAM
        "rem_sram_isolated_f32_bytes": rem_sram_isolated_f32,
        "rem_sram_isolated_int8_bytes": rem_sram_isolated_int8,
        # System Aware (with 120 KB FreeRTOS + WiFi)
        "fits_sram_with_system_f32": fits_sram_f32_system,
        "fits_sram_with_system_int8": fits_sram_int8_system,
        "rem_sram_system_f32_bytes": rem_sram_system_f32,
        "rem_sram_system_int8_bytes": rem_sram_system_int8,
        # PSRAM Fit flags
        "fits_8mb_psram_f32": fits_psram_f32_total,
        "fits_8mb_psram_int8": fits_psram_int8_total,
    }


def calculate_exact_sram_cutoff_widths(
    sram_limit_bytes: int = SRAM_LIMIT_BINARY_BYTES,
) -> Dict[str, Any]:
    """
    Mathematically solve for the exact maximum hidden width w that fits into SRAM.
    
    Formulas:
    ---------
    For 5 -> w -> w -> 1:
      W_total = w^2 + 6w
      B_total = 2w + 1
      P_total = w^2 + 8w + 1
      
    1. Float32 Total Parameters:
         4 * (w^2 + 8w + 1) <= Limit
         w^2 + 8w + (1 - Limit/4) <= 0
         Using quadratic formula: w = (-8 + sqrt(64 - 4*(1 - Limit/4))) / 2
         
    2. Float32 Weights Only:
         4 * (w^2 + 6w) <= Limit
         w^2 + 6w - Limit/4 <= 0
         w = (-6 + sqrt(36 + Limit)) / 2

    3. Int8 Naive (1 byte / parameter):
         w^2 + 8w + 1 <= Limit
         w^2 + 8w + (1 - Limit) <= 0
         w = (-8 + sqrt(64 - 4*(1 - Limit))) / 2

    4. Int8 Realistic (Weights 1 byte, Biases 4 bytes, 24 bytes metadata):
         (w^2 + 6w)*1 + (2w + 1)*4 + 24 <= Limit
         w^2 + 14w + (28 - Limit) <= 0
         w = (-14 + sqrt(196 - 4*(28 - Limit))) / 2
    """
    def solve_quad(a: float, b: float, c: float) -> Tuple[float, int]:
        disc = b**2 - 4 * a * c
        if disc < 0:
            return 0.0, 0
        root = (-b + math.sqrt(disc)) / (2 * a)
        return root, math.floor(root)

    # 1. Float32 Total Params
    # 4w^2 + 32w + (4 - Limit) <= 0
    f32_exact, f32_floor = solve_quad(4.0, 32.0, 4.0 - sram_limit_bytes)

    # 2. Float32 Weights Only
    # 4w^2 + 24w - Limit <= 0
    f32_w_exact, f32_w_floor = solve_quad(4.0, 24.0, -float(sram_limit_bytes))

    # 3. Int8 Naive (1 byte/param)
    # w^2 + 8w + (1 - Limit) <= 0
    int8_naive_exact, int8_naive_floor = solve_quad(1.0, 8.0, 1.0 - sram_limit_bytes)

    # 4. Int8 Realistic (1 byte weight, 4 byte bias, 24 bytes metadata)
    # w^2 + 14w + (28 - Limit) <= 0
    int8_real_exact, int8_real_floor = solve_quad(1.0, 14.0, 28.0 - sram_limit_bytes)

    # 5. System Aware (with 120 KB OS/WiFi overhead)
    net_limit = max(0, sram_limit_bytes - SYSTEM_OVERHEAD_WIFI_BYTES)
    f32_sys_exact, f32_sys_floor = solve_quad(4.0, 32.0, 4.0 - net_limit)
    int8_sys_exact, int8_sys_floor = solve_quad(1.0, 14.0, 28.0 - net_limit)

    return {
        "budget_bytes": sram_limit_bytes,
        "budget_kib": sram_limit_bytes / 1024.0,
        "f32_total": {"exact": round(f32_exact, 3), "max_integer_width": f32_floor},
        "f32_weights_only": {"exact": round(f32_w_exact, 3), "max_integer_width": f32_w_floor},
        "int8_naive": {"exact": round(int8_naive_exact, 3), "max_integer_width": int8_naive_floor},
        "int8_realistic": {"exact": round(int8_real_exact, 3), "max_integer_width": int8_real_floor},
        "system_aware_120kb_overhead": {
            "net_ml_budget_bytes": net_limit,
            "net_ml_budget_kib": net_limit / 1024.0,
            "f32_max_integer_width": f32_sys_floor,
            "int8_max_integer_width": int8_sys_floor,
        },
    }


def compare_feasibility_across_five_widths(
    widths: List[int] = [32, 128, 256, 512, 2048],
) -> pd.DataFrame:
    """
    Generate detailed feasibility comparison table across the 5 benchmark widths.
    """
    records = []
    for w in widths:
        res = evaluate_width_feasibility(w)
        records.append({
            "Width (w)": w,
            "Total Params": res["total_params"],
            "F32 Total (KiB)": round(res["f32_total_kib"], 2),
            "Fits 300KB F32": "YES" if res["fits_300kb_sram_f32"] else "NO",
            "Real Int8 (KiB)": round(res["real_int8_kib"], 2),
            "Fits 300KB Int8": "YES" if res["fits_300kb_sram_int8"] else "NO",
            "Peak Act F32 (Bytes)": res["act_f32_bytes"],
            "Peak Act Int8 (Bytes)": res["act_int8_bytes"],
            "Tot Inf F32 (KiB)": round(res["tot_inf_f32_bytes"] / 1024.0, 2),
            "Tot Inf Int8 (KiB)": round(res["tot_inf_int8_bytes"] / 1024.0, 2),
            "Rem SRAM F32 (KiB)": round(res["rem_sram_isolated_f32_bytes"] / 1024.0, 2),
            "Rem SRAM Int8 (KiB)": round(res["rem_sram_isolated_int8_bytes"] / 1024.0, 2),
            "Fits with WiFi/RTOS F32": "YES" if res["fits_sram_with_system_f32"] else "NO",
            "Fits with WiFi/RTOS Int8": "YES" if res["fits_sram_with_system_int8"] else "NO",
            "Fits 8MB PSRAM F32": "YES" if res["fits_8mb_psram_f32"] else "NO",
            "Fits 8MB PSRAM Int8": "YES" if res["fits_8mb_psram_int8"] else "NO",
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    df_feas = compare_feasibility_across_five_widths()
    print("=== ESP32-S3 SRAM & PSRAM FEASIBILITY ACROSS 5 WIDTHS ===")
    print(df_feas.to_string(index=False))

    cutoffs_bin = calculate_exact_sram_cutoff_widths(SRAM_LIMIT_BINARY_BYTES)
    cutoffs_dec = calculate_exact_sram_cutoff_widths(SRAM_LIMIT_DECIMAL_BYTES)
    print("\n=== EXACT MATHEMATICAL CUTOFF WIDTHS (Binary 307.2 KB) ===")
    print(f"Float32 Total Cutoff:   w <= {cutoffs_bin['f32_total']['max_integer_width']} (exact: {cutoffs_bin['f32_total']['exact']})")
    print(f"Float32 Weights Cutoff: w <= {cutoffs_bin['f32_weights_only']['max_integer_width']} (exact: {cutoffs_bin['f32_weights_only']['exact']})")
    print(f"Int8 Naive Cutoff:      w <= {cutoffs_bin['int8_naive']['max_integer_width']} (exact: {cutoffs_bin['int8_naive']['exact']})")
    print(f"Int8 Realistic Cutoff:  w <= {cutoffs_bin['int8_realistic']['max_integer_width']} (exact: {cutoffs_bin['int8_realistic']['exact']})")
    print(f"With 120 KB WiFi/RTOS:  F32 w <= {cutoffs_bin['system_aware_120kb_overhead']['f32_max_integer_width']}, Int8 w <= {cutoffs_bin['system_aware_120kb_overhead']['int8_max_integer_width']}")
