
"""
Phase 14: Edge AI Memory Calculations
Team 13 - Battery Life Prediction

Calculates:
1. Actual model parameters and memory
2. Float32 and INT8 weight storage
3. Largest hidden-layer activation memory
4. MACs per prediction
5. Width comparison for 32, 128, 256, 512 and 2048
6. SRAM and PSRAM weight-storage checks
7. Exact maximum hidden widths for 300 KiB SRAM
"""

import csv
import json
from pathlib import Path


# Project paths
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model" / "config.json"
OUTPUT_PATH = ROOT / "edge_analysis" / "width_comparison.csv"


# ESP32-S3 memory budgets from assignment
SRAM_BUDGET = 300 * 1024       # 300 KiB
PSRAM_BUDGET = 8 * 1024 * 1024  # 8 MiB


def calculate_model(hidden_widths):
    """
    Calculate model parameters, memory and MACs.

    Architecture:
    5 -> hidden layers -> 1
    """

    layer_sizes = [5, *hidden_widths, 1]

    total_params = 0
    total_macs = 0

    for inputs, outputs in zip(layer_sizes[:-1], layer_sizes[1:]):

        weights = inputs * outputs
        biases = outputs

        total_params += weights + biases
        total_macs += weights

    # Weight and bias storage
    float32_bytes = total_params * 4
    int8_bytes = total_params * 1

    # Largest hidden-layer activation, assuming float32
    activation_bytes = max(hidden_widths) * 4

    return {
        "architecture": " -> ".join(map(str, layer_sizes)),
        "params": total_params,
        "float32_bytes": float32_bytes,
        "int8_bytes": int8_bytes,
        "activation_bytes": activation_bytes,
        "macs": total_macs,

        # These checks concern weights and biases only.
        "fits_sram_float32_weights": float32_bytes <= SRAM_BUDGET,
        "fits_sram_int8_weights": int8_bytes <= SRAM_BUDGET,

        "fits_psram_float32_weights": float32_bytes <= PSRAM_BUDGET,
        "fits_psram_int8_weights": int8_bytes <= PSRAM_BUDGET,
    }


def calculate_exact_width_limits():
    """Find the largest equal hidden width fitting the SRAM weight budget."""

    def parameter_count(width):
        # Architecture: 5 -> width -> width -> 1
        return width * width + 8 * width + 1

    max_float32_width = max(
        width
        for width in range(1, 10000)
        if parameter_count(width) * 4 <= SRAM_BUDGET
    )

    max_int8_width = max(
        width
        for width in range(1, 10000)
        if parameter_count(width) <= SRAM_BUDGET
    )

    return max_float32_width, max_int8_width


def save_width_comparison(results):
    """Save width comparison results to CSV."""

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)


def main():

    # Load actual trained model configuration
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    actual_widths = config["hidden"]

    # -------------------------------------------------
    # ACTUAL TEAM 13 MODEL
    # -------------------------------------------------

    print("\n========== ACTUAL TEAM 13 MODEL ==========")

    actual = calculate_model(actual_widths)

    for key, value in actual.items():
        print(f"{key}: {value}")

    print(f"Validation MAE: {config['val_mae']} days")

    # -------------------------------------------------
    # WIDTH COMPARISON
    # -------------------------------------------------

    print("\n========== WIDTH COMPARISON ==========")

    widths = [32, 128, 256, 512, 2048]

    results = []

    for width in widths:

        result = calculate_model([width, width])

        results.append({
            "width": width,
            **result
        })

        print(f"\nWidth: {width}")
        print(f"Architecture: {result['architecture']}")
        print(f"Parameters: {result['params']}")
        print(f"Float32 bytes: {result['float32_bytes']}")
        print(f"INT8 bytes: {result['int8_bytes']}")
        print(f"Activation bytes: {result['activation_bytes']}")
        print(f"MACs: {result['macs']}")
        print(
            "Fits SRAM Float32 weights:",
            result["fits_sram_float32_weights"]
        )
        print(
            "Fits SRAM INT8 weights:",
            result["fits_sram_int8_weights"]
        )
        print(
            "Fits PSRAM Float32 weights:",
            result["fits_psram_float32_weights"]
        )
        print(
            "Fits PSRAM INT8 weights:",
            result["fits_psram_int8_weights"]
        )

    save_width_comparison(results)

    print(f"\nSaved comparison CSV: {OUTPUT_PATH}")

    # -------------------------------------------------
    # EXACT SRAM WIDTH LIMITS
    # -------------------------------------------------

    print("\n========== EXACT SRAM WIDTH LIMITS ==========")

    float32_limit, int8_limit = calculate_exact_width_limits()

    print(f"SRAM budget: {SRAM_BUDGET} bytes")
    print(f"Float32 maximum width: {float32_limit}")
    print(f"INT8 maximum width: {int8_limit}")

    # Verify the next width exceeds the budget
    next_float32 = calculate_model([float32_limit + 1] * 2)
    next_int8 = calculate_model([int8_limit + 1] * 2)

    print(
        "Float32 next width exceeds:",
        next_float32["float32_bytes"] > SRAM_BUDGET
    )

    print(
        "INT8 next width exceeds:",
        next_int8["int8_bytes"] > SRAM_BUDGET
    )

    print("\n========== CALCULATIONS COMPLETED ==========")


if __name__ == "__main__":
    main()
