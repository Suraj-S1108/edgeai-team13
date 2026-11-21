# Mission 4: Executive Summary of Findings
**Project:** AI255TBD Edge AI - Module 1 (Team 13: How Long Will the Battery Last?)  
**Owner:** Nandish (Mission 4: Edge-Device and Memory-Analysis Owner)  
**Date:** November 2026  

---

## 1. Key Numerical Takeaways

| Dimension | w = 32 | w = 128 | w = 256 | w = 512 | w = 2048 |
|---|---|---|---|---|---|
| **Total Parameters** | 1,281 | 17,409 | 67,585 | 266,241 | 4,210,689 |
| **Total MACs** | 1,216 | 17,152 | 67,072 | 265,216 | 4,206,592 |
| **Float32 Size (KiB)** | 5.00 KiB | 67.97 KiB | 264.00 KiB | 1,040.00 KiB | 16,448.00 KiB |
| **Realistic INT8 Size (KiB)** | 1.46 KiB | 17.77 KiB | 67.53 KiB | 263.00 KiB | 4,124.00 KiB |
| **Fits 300 KB SRAM (Float32)** | **YES** | **YES** | **YES** | **NO** | **NO** |
| **Fits 300 KB SRAM (INT8)** | **YES** | **YES** | **YES** | **YES** | **NO** |
| **Fits 8 MB PSRAM** | **YES** | **YES** | **YES** | **YES** | **YES (INT8 only)** |
| **Peak Activation (F32)** | 256 B | 1.00 KiB | 2.00 KiB | 4.00 KiB | 16.00 KiB |
| **ESP32-S3 INT8 Latency** | **9.85 µs** | **0.139 ms** | **0.543 ms** | **2.15 ms** | **34.06 ms** |
| **ESP32-S3 Float32 Latency** | **50.7 µs** | **0.715 ms** | **2.79 ms** | **11.05 ms** | **175.27 ms** |

---

## 2. Critical Engineering Insights

1. **Exact SRAM Cutoff Limits (300 KB / 307.2 KiB Limit):**
   - **Float32 Complete Model:** Cutoff occurs at **w = 273** ($w^2 + 8w + 1 \\le 76,800$).
     * $w = 256$ (264.0 KiB) fits; $w = 512$ (1,040 KiB) fails.
   - **INT8 Realistic Quantization:** Cutoff occurs at **w = 547** ($w^2 + 14w + 28 \\le 307,200$).
     * Both $w = 256$ (67.5 KiB) and $w = 512$ (263.0 KiB) fit comfortably in internal SRAM.
   - **System-Aware Deployment (accounting for 120 KB FreeRTOS + Wi-Fi Stack):**
     * Effective ML budget reduces to **187.2 KiB**.
     * Float32 cutoff drops to **w = 210**.
     * INT8 cutoff drops to **w = 422**.
     * Under full networking load, $w = 256$ requires INT8 quantization to avoid out-of-memory heap panics!

2. **PSRAM Boundary Violation for w = 2048:**
   - In Float32, $w = 2048$ requires **16.45 MiB**, which **exceeds the 8 MB external PSRAM**!
   - Quantizing to INT8 compresses the model to **4.12 MiB**, enabling it to fit into the 8 MB PSRAM.

3. **Inference Latency & SIMD Acceleration:**
   - On the ESP32-S3 Xtensa LX7 @ 240 MHz, Espressif's ESP-NN SIMD instructions provide a **5.15x speedup** over scalar Float32.
   - At $w = 32$, inference executes in **9.85 microseconds (0.0098 ms)**.
   - At $w = 256$, inference executes in **0.543 milliseconds**.

4. **Battery Energy Advantage (Server vs Edge):**
   - **Cloud Server Inference:** Requires waking 2.4 GHz Wi-Fi radio (220 mA for ~3 seconds) $\\rightarrow$ **2.178 Joules/reading**.
   - **Edge On-Device Inference:** Wi-Fi radio stays off; MCU wakes for 2 ms at 40 mA $\\rightarrow$ **0.000264 Joules/reading**.
   - **Result:** Edge inference uses **8,250.0x less energy** per prediction.
   - On a 3000 mAh Li-ion battery with hourly readings, Edge inference extends operational life from **646.6 days to 12472.3 days**!

5. **Final Architecture Recommendation:**
   - **Primary Model Choice:** $w = 32$ or $w = 64$ with INT8 quantization provides the optimal pareto frontier. It consumes $< 2$ KiB of SRAM, executes in under $10$ µs, and preserves the device battery for years.
