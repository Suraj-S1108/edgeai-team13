# EDGE AI Team 13 — Memory & Microcontroller Hardware Resource Analysis

## 1. Overview
This report provides a formal engineering analysis of the resource requirements, memory footprint, and microcontroller deployment feasibility for Team 13's battery prediction models on an **Espressif ESP32-S3** microcontroller.

---

## 2. Hardware Resource Budgets (ESP32-S3)
- **Internal SRAM Budget**: 300 KiB (307,200 bytes)
- **External PSRAM Budget**: 8 MiB (8,388,608 bytes)
- **Target Deployment Constraint**: Model weights and working activation buffers must fit completely within internal SRAM without incurring PSRAM SPI bus latency penalties.

---

## 3. Detailed Architectural Resource Comparison

| Metric | Legacy Model (16-16) | Deployed Model (32-32) | Unit / Context |
|---|---|---|---|
| **Architecture** | 5 -> 16 -> 16 -> 1 | 5 -> 32 -> 32 -> 1 | Sequential Dense |
| **Layer 1 (Linear)** | 5 in, 16 out (80 W + 16 B) | 5 in, 32 out (160 W + 32 B) | Weights + Biases |
| **Layer 2 (Linear)** | 16 in, 16 out (256 W + 16 B) | 32 in, 32 out (1024 W + 32 B) | Weights + Biases |
| **Layer 3 (Linear)** | 16 in, 1 out (16 W + 1 B) | 32 in, 1 out (32 W + 1 B) | Weights + Biases |
| **Total Weights** | 352 | 1,216 | Trainable weights |
| **Total Biases** | 33 | 65 | Trainable biases |
| **Total Parameters** | **385** | **1,281** | Total parameters |
| **FP32 Weight Storage** | 1,540 bytes (1.50 KiB) | 5,124 bytes (5.00 KiB) | 4 bytes / parameter |
| **FP16 Weight Storage** | 770 bytes (0.75 KiB) | 2,562 bytes (2.50 KiB) | 2 bytes / parameter |
| **INT8 Weight Storage** | 385 bytes (0.38 KiB) | 1,281 bytes (1.25 KiB) | 1 byte / parameter |
| **Input Buffer** | 20 bytes | 20 bytes | 5 float32 elements |
| **Output Buffer** | 4 bytes | 4 bytes | 1 float32 element |
| **Peak Hidden Activation** | 64 bytes (16 float32) | 128 bytes (32 float32) | Largest intermediate tensor |
| **MAC Operations** | 352 | 1,216 | Multiplications & additions |
| **Floating Point Operations**| 704 FLOPs | 2,432 FLOPs | $2 \times \text{MACs}$ |
| **Disk Checkpoint (.pt)** | 4,285 bytes | 8,357 bytes | PyTorch ZIP container |
| **Microcontroller Arena (FP32)**| 1,924 bytes | 5,636 bytes | Weights + double activation buffer |
| **Microcontroller Arena (INT8)**| 673 bytes | 1,601 bytes | Quantized weights + arena |
| **ESP32-S3 SRAM Utilization** | **0.50%** | **1.67%** | % of 300 KiB internal SRAM |
| **Fits Internal SRAM (FP32)** | **YES** | **YES** | No external PSRAM needed |

---

## 4. Key Engineering Insights
1. **Negligible Memory Impact**: Transitioning from 16-16 to 32-32 increases parameter storage from 1.50 KiB to 5.00 KiB (+3.5 KiB). On a 300 KiB SRAM microcontroller, the model consumes only **1.67%** of memory, leaving 295 KiB free for the Wi-Fi stack, TLS crypto, FreeRTOS tasks, and application logic.
2. **Double-Buffering & Peak Memory**: Because inference proceeds sequentially layer-by-layer, intermediate activations can be allocated using a ping-pong buffer of size $2 \times 128 = 256$ bytes.
3. **Execution Complexity**: At 1,216 MACs (2,432 FLOPs), on an ESP32-S3 running at 240 MHz with dual-core Xtensa LX7 and vector instructions, inference takes less than **0.05 milliseconds** per prediction.
