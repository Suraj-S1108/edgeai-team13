"""
memory_calculator.py
====================
Provides memory consumption calculations for weights, biases, quantization metadata,
and intermediate runtime activations across Float32 and Int8 representations.

Distinguishes:
1. Float32 standard precision (4 bytes per parameter).
2. Naive Int8 representation (1 byte per parameter, commonly assumed in simplistic starter models).
3. Realistic Embedded Int8 representation (Standard in TensorFlow Lite Micro, ESP-DL, CMSIS-NN):
   - Int8 weights (1 byte/weight)
   - Int32 biases (4 bytes/bias to prevent overflow in accumulators)
   - Quantization scales (float32, 4 bytes/layer) and zero-points (int32/int8, 4 bytes/layer)
   - Alignment and tensor header overhead
4. Activation memory and buffer reuse under sequential execution.

Author: Nandish (Mission 4 - Edge Device & Memory Analysis)
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from edge_analysis.parameter_calculator import calculate_two_hidden_mlp_formulas, calculate_mlp_architecture


# Conversion constants
BYTES_PER_FLOAT32 = 4
BYTES_PER_INT8 = 1
BYTES_PER_INT32 = 4


def calculate_float32_memory(total_weights: int, total_biases: int) -> Dict[str, Any]:
    """
    Calculate memory required for Float32 parameters.
    
    Args:
        total_weights: Total count of weight parameters.
        total_biases: Total count of bias parameters.
        
    Returns:
        Dict with byte, KB (decimal 1000), and KiB (binary 1024) values.
    """
    if total_weights < 0 or total_biases < 0:
        raise ValueError("Weights and biases counts cannot be negative")

    weight_bytes = total_weights * BYTES_PER_FLOAT32
    bias_bytes = total_biases * BYTES_PER_FLOAT32
    total_bytes = weight_bytes + bias_bytes

    return {
        "weight_bytes": weight_bytes,
        "bias_bytes": bias_bytes,
        "total_bytes": total_bytes,
        "weight_kb": weight_bytes / 1000.0,
        "bias_kb": bias_bytes / 1000.0,
        "total_kb": total_bytes / 1000.0,
        "weight_kib": weight_bytes / 1024.0,
        "bias_kib": bias_bytes / 1024.0,
        "total_kib": total_bytes / 1024.0,
    }


def calculate_int8_memory(
    total_weights: int,
    total_biases: int,
    num_layers: int = 3,
    include_metadata: bool = True,
    alignment_bytes: int = 4,
) -> Dict[str, Any]:
    """
    Calculate memory required for Int8 quantized parameters.
    
    Evaluates both:
    1. Naive Int8: assumes 1 byte for all parameters (weights + biases).
    2. Realistic Embedded Int8:
       - Weights in INT8 (1 byte each)
       - Biases in INT32 (4 bytes each, required in INT8 GEMM/Conv to avoid overflow)
       - Per-layer quantization scale (float32, 4 bytes) and zero-point (int32, 4 bytes)
       - Memory alignment padding (aligned to 4-byte boundaries)

    Args:
        total_weights: Total weight parameters.
        total_biases: Total bias parameters.
        num_layers: Number of quantized linear layers (default 3 for 5 -> w -> w -> 1).
        include_metadata: Whether to include scale/zero-point metadata overhead.
        alignment_bytes: Word alignment boundary (default 4 bytes).
        
    Returns:
        Dict containing both naive and realistic embedded estimates.
    """
    if total_weights < 0 or total_biases < 0 or num_layers <= 0:
        raise ValueError("Parameter counts and num_layers must be non-negative/positive")

    total_params = total_weights + total_biases

    # 1. Naive Int8 (1 byte per parameter across the board)
    naive_bytes = total_params * BYTES_PER_INT8

    # 2. Realistic Embedded Int8
    real_weight_bytes = total_weights * BYTES_PER_INT8
    real_bias_bytes = total_biases * BYTES_PER_INT32  # standard 32-bit accumulator bias

    # Per-layer quantization metadata:
    # 1 float32 scale (4 bytes) + 1 int32 zero point (4 bytes) per layer = 8 bytes/layer
    metadata_bytes = (num_layers * 8) if include_metadata else 0

    unaligned_total = real_weight_bytes + real_bias_bytes + metadata_bytes
    # Alignment padding
    if alignment_bytes > 0:
        padding = (alignment_bytes - (unaligned_total % alignment_bytes)) % alignment_bytes
    else:
        padding = 0
    realistic_bytes = unaligned_total + padding

    return {
        "naive_int8": {
            "total_bytes": naive_bytes,
            "total_kb": naive_bytes / 1000.0,
            "total_kib": naive_bytes / 1024.0,
        },
        "realistic_int8": {
            "weight_bytes": real_weight_bytes,
            "bias_bytes": real_bias_bytes,
            "metadata_bytes": metadata_bytes,
            "padding_bytes": padding,
            "total_bytes": realistic_bytes,
            "total_kb": realistic_bytes / 1000.0,
            "total_kib": realistic_bytes / 1024.0,
        },
    }


def calculate_activation_memory(
    input_dim: int = 5,
    hidden_widths: tuple = (32, 32),
    output_dim: int = 1,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Calculate intermediate activation buffer memory requirements for sequential MLP inference.
    
    In sequential layer-by-layer evaluation on an embedded edge device:
    - Layer 1 computes: Input (5) -> Hidden 1 (w)
      Requires: Input buffer + Hidden 1 buffer simultaneously live.
    - Layer 2 computes: Hidden 1 (w) -> Hidden 2 (w)
      Requires: Hidden 1 buffer + Hidden 2 buffer live simultaneously.
    - Output layer computes: Hidden 2 (w) -> Output (1)
      Requires: Hidden 2 buffer + Output buffer live simultaneously.
      
    With an optimized double-buffering (ping-pong / arena allocator) strategy:
      Peak buffer capacity = 2 * max(layer_dimensions)
      For 5 -> w -> w -> 1 where w >= 5:
        Buffer A: max(5, w) = w elements
        Buffer B: max(w, 1) = w elements
        Peak simultaneous elements = 2 * w elements
        
    Args:
        input_dim: Input features (5).
        hidden_widths: Tuple of hidden layer sizes (e.g., (32, 32)).
        output_dim: Output features (1).
        batch_size: Inference batch size (default 1 for edge device).
        
    Returns:
        Dict with element counts and memory in Float32 and Int8.
    """
    if input_dim <= 0 or output_dim <= 0 or batch_size <= 0:
        raise ValueError("Dimensions and batch_size must be positive integers")

    dims = [input_dim] + list(hidden_widths) + [output_dim]
    
    # Per-layer activation element counts
    layer_elements = [d * batch_size for d in dims]
    
    # Sequential peak live memory calculation:
    # At any step i -> i+1, we need layer_elements[i] + layer_elements[i+1]
    step_live_elements = [layer_elements[i] + layer_elements[i+1] for i in range(len(layer_elements) - 1)]
    peak_live_elements = max(step_live_elements)
    
    # Ping-pong double buffer strategy:
    # Buffer A holds even layers, Buffer B holds odd layers
    max_even = max(layer_elements[0::2])
    max_odd = max(layer_elements[1::2])
    ping_pong_elements = max_even + max_odd
    
    # Total sum if NO buffer reuse (naive allocation)
    total_unshared_elements = sum(layer_elements)

    return {
        "batch_size": batch_size,
        "layer_dimensions": dims,
        "layer_elements": layer_elements,
        "step_live_elements": step_live_elements,
        "peak_live_elements": peak_live_elements,
        "ping_pong_elements": ping_pong_elements,
        "total_unshared_elements": total_unshared_elements,
        # Float32 (4 bytes per element)
        "float32": {
            "peak_live_bytes": peak_live_elements * BYTES_PER_FLOAT32,
            "peak_live_kib": (peak_live_elements * BYTES_PER_FLOAT32) / 1024.0,
            "ping_pong_bytes": ping_pong_elements * BYTES_PER_FLOAT32,
            "unshared_bytes": total_unshared_elements * BYTES_PER_FLOAT32,
        },
        # Int8 (1 byte per element)
        "int8": {
            "peak_live_bytes": peak_live_elements * BYTES_PER_INT8,
            "peak_live_kib": (peak_live_elements * BYTES_PER_INT8) / 1024.0,
            "ping_pong_bytes": ping_pong_elements * BYTES_PER_INT8,
            "unshared_bytes": total_unshared_elements * BYTES_PER_INT8,
        },
    }


def compare_memory_across_widths(widths: List[int] = [32, 128, 256, 512, 2048]) -> pd.DataFrame:
    """
    Comprehensive memory comparison across the 5 benchmark widths.
    
    Compares:
    - Float32 weights, biases, total bytes, KiB
    - Naive Int8 total bytes, KiB
    - Realistic Embedded Int8 (int8 weights + int32 biases + metadata) bytes, KiB
    - Memory savings and compression ratio
    - Peak live activation memory (Float32 vs Int8)
    """
    records = []
    for w in widths:
        params = calculate_two_hidden_mlp_formulas(w)
        f32 = calculate_float32_memory(params["total_weights"], params["total_biases"])
        int8 = calculate_int8_memory(params["total_weights"], params["total_biases"], num_layers=3)
        act = calculate_activation_memory(input_dim=5, hidden_widths=(w, w), output_dim=1, batch_size=1)

        f32_bytes = f32["total_bytes"]
        naive_int8_bytes = int8["naive_int8"]["total_bytes"]
        real_int8_bytes = int8["realistic_int8"]["total_bytes"]

        savings_naive = f32_bytes - naive_int8_bytes
        ratio_naive = f32_bytes / naive_int8_bytes if naive_int8_bytes > 0 else 0

        savings_real = f32_bytes - real_int8_bytes
        ratio_real = f32_bytes / real_int8_bytes if real_int8_bytes > 0 else 0

        records.append({
            "Width (w)": w,
            "Architecture": f"5 -> {w} -> {w} -> 1",
            "Total Params": params["total_params"],
            # Float32
            "F32 Weight Bytes": f32["weight_bytes"],
            "F32 Bias Bytes": f32["bias_bytes"],
            "F32 Total Bytes": f32_bytes,
            "F32 Total KiB": round(f32["total_kib"], 3),
            "F32 Total KB": round(f32["total_kb"], 3),
            # Naive Int8
            "Naive Int8 Bytes": naive_int8_bytes,
            "Naive Int8 KiB": round(int8["naive_int8"]["total_kib"], 3),
            "Naive Ratio": round(ratio_naive, 2),
            # Realistic Int8
            "Real Int8 W (Bytes)": int8["realistic_int8"]["weight_bytes"],
            "Real Int8 B (int32 Bytes)": int8["realistic_int8"]["bias_bytes"],
            "Real Int8 Meta Bytes": int8["realistic_int8"]["metadata_bytes"] + int8["realistic_int8"]["padding_bytes"],
            "Real Int8 Total Bytes": real_int8_bytes,
            "Real Int8 Total KiB": round(int8["realistic_int8"]["total_kib"], 3),
            "Real Int8 Ratio": round(ratio_real, 2),
            "Real Int8 Savings (Bytes)": savings_real,
            # Activation Memory (Peak Live)
            "Peak Act Elements": act["peak_live_elements"],
            "Peak Act F32 (Bytes)": act["float32"]["peak_live_bytes"],
            "Peak Act Int8 (Bytes)": act["int8"]["peak_live_bytes"],
        })
    return pd.DataFrame(records)


def compare_activation_memory_across_widths(widths: List[int] = [32, 128, 256, 512, 2048]) -> pd.DataFrame:
    """
    Dedicated activation memory breakdown for the 5 benchmark widths.
    """
    records = []
    for w in widths:
        act = calculate_activation_memory(input_dim=5, hidden_widths=(w, w), output_dim=1, batch_size=1)
        records.append({
            "Width (w)": w,
            "Input Elements (5)": act["layer_elements"][0],
            "Hidden 1 Elements (w)": act["layer_elements"][1],
            "Hidden 2 Elements (w)": act["layer_elements"][2],
            "Output Elements (1)": act["layer_elements"][3],
            "Total Unshared Elements": act["total_unshared_elements"],
            "Peak Live Elements": act["peak_live_elements"],
            "Peak Live F32 (Bytes)": act["float32"]["peak_live_bytes"],
            "Peak Live F32 (KiB)": round(act["float32"]["peak_live_kib"], 4),
            "Peak Live Int8 (Bytes)": act["int8"]["peak_live_bytes"],
            "Peak Live Int8 (KiB)": round(act["int8"]["peak_live_kib"], 4),
            "Double Buffer F32 (Bytes)": act["float32"]["ping_pong_bytes"],
            "Double Buffer Int8 (Bytes)": act["int8"]["ping_pong_bytes"],
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    df_mem = compare_memory_across_widths()
    print("=== MEMORY COMPARISON ACROSS 5 BENCHMARK WIDTHS ===")
    print(df_mem.to_string(index=False))

    df_act = compare_activation_memory_across_widths()
    print("\n=== ACTIVATION MEMORY COMPARISON ===")
    print(df_act.to_string(index=False))

