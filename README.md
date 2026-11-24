# EDGE AI Team 13 — Doorbell Battery Life Prediction API

> **Team 13 Members**: Suraj, Sujal Chavan, Nandish, Monika
> **Course / Program**: Edge AI
> **Production Model**: Cleaned 5 &rarr; 32 &rarr; 32 &rarr; 1 Neural Network (`Cleaned_NN_2L_32_32`)
> **Parameters**: 1,281 | **Validation MAE**: ~5.22 days | **Target Hardware**: ESP32-S3 (300 KiB SRAM Budget)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Production Architecture](#2-production-architecture)
3. [Dataset & Preprocessing](#3-dataset--preprocessing)
4. [Training Methodology & Model Selection](#4-training-methodology--model-selection)
5. [Validation Performance](#5-validation-performance)
6. [Edge AI Resource & Memory Calculations](#6-edge-ai-resource--memory-calculations)
7. [Benchmark Results](#7-benchmark-results)
8. [API Endpoints Specification](#8-api-endpoints-specification)
9. [Installation & Setup](#9-installation--setup)
10. [Starting the Backend API](#10-starting-the-backend-api)
11. [Automated Testing](#11-automated-testing)
12. [Manual API Testing Instructions](#12-manual-api-testing-instructions)
13. [Project Directory Structure](#13-project-directory-structure)
14. [Limitations & Future Work](#14-limitations--future-work)

---

## 1. Project Overview

This repository provides an Edge AI prediction service for battery-powered smart video doorbells. Telemetry data from five on-device sensors—battery terminal voltage, ambient temperature, daily chime frequency, Wi-Fi signal strength, and device operating age—is evaluated by a PyTorch neural network to estimate the remaining battery lifetime in days (`days_left`).

The production service is packaged with **FastAPI** and uses **Pydantic** schema validation to guard against corrupted or out-of-bounds telemetry. The runtime pipeline applies feature standardization, forwards inputs through the neural network, enforces non-negative prediction clipping, and logs granular server timing diagnostics.

Hardware analysis targets the **Espressif ESP32-S3** microcontroller (budgeted at 300 KiB internal SRAM). The production model requires only **5,124 bytes (5.00 KiB)** of FP32 parameter storage—utilizing approximately **1.668%** of the 300 KiB SRAM budget—making it fully capable of in-memory edge execution without external PSRAM.

---

## 2. Production Architecture

The production model is a two-hidden-layer fully connected feedforward network:

$$\text{Input (5)} \longrightarrow \text{Linear}(5 \rightarrow 32) + \text{ReLU} \longrightarrow \text{Linear}(32 \rightarrow 32) + \text{ReLU} \longrightarrow \text{Linear}(32 \rightarrow 1) \longrightarrow \text{Output (1)}$$

### Layer Specifications

| Layer | Input Dimension | Output Dimension | Weights | Biases | Total Parameters | MACs | FP32 Storage | INT8 Storage |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `linear1` + ReLU | 5 | 32 | $5 \times 32 = 160$ | 32 | 192 | 160 | 768 B | 192 B |
| `linear2` + ReLU | 32 | 32 | $32 \times 32 = 1,024$ | 32 | 1,056 | 1,024 | 4,224 B | 1,056 B |
| `linear3` (Output) | 32 | 1 | $32 \times 1 = 32$ | 1 | 33 | 32 | 132 B | 33 B |
| **Total** | — | — | **1,216** | **65** | **1,281** | **1,216** | **5,124 B** | **1,281 B** |

- **Floating-Point Operations (FLOPs)**: $2 \times \text{MACs} = 2 \times 1,216 = \mathbf{2,432\text{ FLOPs}}$
- **Active Production Checkpoint**: `model/model.pt` (PyTorch state dictionary, 8,357 bytes on disk, SHA-256: `850306bd59df75f1e808a05e12dba87df5650397900ca405577e55bed4452186`)
- **Active Production Configuration**: `model/config.json`
- **Network Definition**: `app/model.py` (`BatteryNet` class)

> [!NOTE]
> Historical 16–16 checkpoints located under `edge_analysis/mission5_results/models/` serve strictly as experimental ablation evidence for model-selection documentation and are not used by the production API.

---

## 3. Dataset & Preprocessing

### Dataset Summary
- **Original Field Dataset**: `data/team_13_train.csv` (2,000 rows, 0 nulls)
- **Sanitized Cleaned Dataset**: `data/team_13_train_cleaned.csv` (1,976 rows, 0 nulls)
- **Filtered Anomalies**: 24 corrupted rows removed (1.2% of the dataset)

### Data Hygiene Protocol & Glitch Removal Rationale
1. **Battery Voltage Equal to Zero (`battery_v <= 0`) — 11 rows removed**
   A single-cell Li-ion battery operates between ~3.0 V (undervoltage cutoff) and 4.2 V (full charge). At 0.0 V, the device has experienced an ADC brownout or disconnected power bus; the microcontroller cannot operate.
2. **Temperature Equal to or Below Zero (`temp_c <= 0`) — 8 rows removed**
   In all 8 corrupted records, the temperature was recorded as exactly **`-127.0 °C`**. In standard digital thermal sensors (such as Dallas/Maxim 1-Wire sensors), `-127 °C` / `0x80` is the digital power-on reset (POR) or bus read fault code, not outdoor temperature.
3. **Wi-Fi RSSI Equal to Zero (`wifi_rssi_dbm == 0`) — 5 rows removed**
   Wi-Fi Received Signal Strength Indicator (RSSI) is measured in negative decibel-milliwatts ($\text{dBm}$), typically between -90 dBm (barely readable) and -20 dBm (near router). Exactly `0 dBm` represents 1 milliwatt of received power at the antenna, which indicates an uninitialized Wi-Fi MAC layer register or failed frame header parse.
4. **Legitimate Operational Zeroes Preserved**:
   - `rings_per_day == 0` (11 records retained): A doorbell experiencing 0 chimes on a given day (vacant home, quiet day) is physically normal.
   - `age_months == 0` (48 records retained in cleaned set): A newly installed doorbell operating in its first month of deployment is physically expected.

### Normalization Statistics
Standardization is performed before feeding features into the network using statistics computed exclusively on the training split:

$$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$

Values stored in `model/config.json`:
- **Mean ($\boldsymbol{\mu}$)**:
  - `battery_v`: `3.771762810363242`
  - `temp_c`: `27.71404174573055`
  - `rings_per_day`: `7.8646426312460465`
  - `wifi_rssi_dbm`: `-66.59266287160025`
  - `age_months`: `18.32384566730044`
- **Standard Deviation ($\boldsymbol{\sigma}$)**:
  - `battery_v`: `0.2446153163909912`
  - `temp_c`: `6.795620698928833`
  - `rings_per_day`: `5.635627689361572`
  - `wifi_rssi_dbm`: `12.102129054069518`
  - `age_months`: `10.623634071350098`

### Cleaned Training-Data Ranges vs. Operational API Validation Ranges

> [!IMPORTANT]
> The API validation ranges in `app/main.py` represent **selected operational validation assumptions** designed to reject corrupted inputs and prevent server errors. They are **not** certified manufacturer hardware ratings, nor are they restricted to the empirical min/max of the 2,000 training records.

Real-world doorbells encounter operational conditions outside the empirical sample window of the training dataset (e.g. freezing outdoor weather down to 0 °C, deep cell discharge down to 2.50 V, high visitor bursts up to 1,000 rings, or mature installations up to 240 months). The table below contrasts the observed cleaned dataset boundaries with the implemented API validation bounds:

| Feature | Cleaned Training Range | Selected API Validation Range (`app/main.py`) | Operational Rationale |
|:---|:---:|:---:|:---|
| **`battery_v`** | $[3.35, 4.20]$ V | `ge=2.50, le=4.20` | Allows deep discharge down to Li-ion 2.50 V cutoff; rejects $\le 0$ V power brownouts and overvoltages $> 4.20$ V. |
| **`temp_c`** | $[8.0, 46.0]$ °C | `ge=0, le=85` | Covers outdoor temperatures from freezing point (0 °C) up to 85 °C electronics rating; rejects the negative `-127 °C` sensor disconnect glitch. |
| **`rings_per_day`** | $[0, 44]$ rings | `ge=0, le=1000` | Accepts 0 chimes as valid; accommodates high visitor bursts up to 1,000; rejects negative counts. |
| **`wifi_rssi_dbm`** | $[-88.0, -45.0]$ dBm | `ge=-100, le=-1` | Enforces negative dBm (receiver floor -100 dBm up to -1 dBm); rejects the `0 dBm` uninitialized radio glitch. |
| **`age_months`** | $[0, 36]$ months | `ge=0, le=240` | Accepts month 0 (new) up to 240 months (20-year hardware lifetime); rejects negative device ages. |

---

## 4. Training Methodology & Model Selection

- **Dataset Split**: 80% training (1,581 records) / 20% validation (395 records), fixed random seed 42.
- **Optimization**: Adam optimizer ($\text{learning rate} = 0.01$, $\text{weight decay} = 10^{-4}$, $\text{batch size} = 128$).
- **Objective Function**: Mean Squared Error (MSE).
- **Early Stopping & Checkpointing**: Trained across 1,500 epochs; best validation checkpoint saved at **Epoch 1,123**.
- **Model Selection Rationale**:
  - **Over 16–16 baseline**: Reduces validation MAE by **24.2%** (from 6.88 days down to 5.22 days) for an increase of only 896 parameters.
  - **Over 64–64 architecture**: The 64–64 model requires 4,609 parameters (3.6x larger) but achieves only a marginal 0.09-day MAE gain (5.13 vs 5.22 days) while exhibiting a larger overfitting gap (1.53 vs 0.89 days).
  - **Over linear baseline**: Reduces validation MAE from 26.05 days to 5.22 days.
  - **Pareto Optimality**: The 5 &rarr; 32 &rarr; 32 &rarr; 1 model provides the optimal trade-off between predictive accuracy and edge memory constraints.

---

## 5. Validation Performance

The table below summarizes performance metrics across models evaluated on the same 80/20 cleaned validation split (source: `edge_analysis/mission5_results/mission5_train_validation_complete.csv`):

| Model Architecture | Parameters | Best Epoch | Train MAE | Val MAE | Val RMSE | Val $R^2$ | Overfitting Gap (MAE) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cleaned NN_2L_32_32 (Production)** | **1,281** | **1,123** | **4.33 days** | **5.22 days** | **7.45 days** | **0.9888** | **0.89 days** |
| Cleaned NN_2L_64_64 | 4,609 | 1,332 | 3.60 days | 5.13 days | 7.65 days | 0.9882 | 1.53 days |
| Cleaned NN_2L_16_16 | 385 | 1,498 | 6.10 days | 6.88 days | 9.72 days | 0.9809 | 0.79 days |
| Cleaned NN_2L_8_8 | 129 | 1,500 | 6.34 days | 7.34 days | 10.88 days | 0.9761 | 1.00 days |
| Cleaned Linear Regression | 6 | — | 24.18 days | 26.05 days | 34.97 days | 0.7531 | 1.87 days |
| Cleaned Mean Baseline | 0 | — | 52.52 days | 55.62 days | 70.78 days | -0.0114 | 3.11 days |

> [!NOTE]
> Metrics from earlier uncleaned experiments (e.g. NN_2L_16_16 on raw uncleaned data yielding 7.14 days MAE, and Linear Regression yielding 32.48 days MAE) belong to exploratory baseline missions and are documented separately in `edge_analysis/mission5_results/mission5_complete_model_comparison.csv`.

---

## 6. Edge AI Resource & Memory Calculations

Hardware calculations evaluate theoretical suitability for the **Espressif ESP32-S3** microcontroller:
- **Internal SRAM Budget**: 300 KiB (307,200 bytes)
- **External PSRAM Budget**: 8 MiB (8,388,608 bytes)

> [!NOTE]
> These figures represent **theoretical memory calculations** and static tensor memory analyses derived from architecture formulas in `edge_analysis/memory_calculations.py`. They are not physical hardware measurements from an ESP32-S3 board.

### Deployed 32–32 Resource Breakdown

| Resource Metric | Value (Bytes) | Human-Readable | Calculation / Formula |
|:---|:---:|:---:|:---|
| **FP32 Parameter Memory** | 5,124 bytes | 5.00 KiB | $1,281\text{ parameters} \times 4\text{ bytes}$ |
| **FP16 Parameter Memory** | 2,562 bytes | 2.50 KiB | $1,281\text{ parameters} \times 2\text{ bytes}$ |
| **INT8 Parameter Memory** | 1,281 bytes | 1.25 KiB | $1,281\text{ parameters} \times 1\text{ byte}$ |
| **Input Tensor Buffer** | 20 bytes | 20 B | 5 float32 elements ($5 \times 4\text{ B}$) |
| **Output Tensor Buffer** | 4 bytes | 4 B | 1 float32 element ($1 \times 4\text{ B}$) |
| **Peak Activation Buffer** | 128 bytes | 128 B | Layer 1 / 2 output ($32 \times 4\text{ B}$) |
| **Multiply-Accumulate Operations (MACs)** | 1,216 MACs | 1,216 | $(5 \times 32) + (32 \times 32) + (32 \times 1)$ |
| **FLOPs per Inference** | 2,432 FLOPs | 2,432 | $2 \times \text{MACs}$ |
| **FP32 Microcontroller Arena Estimate** | 5,636 bytes | 5.50 KiB | $\text{Weights} + (2 \times \text{Peak Activation}) + 256\text{ B scratchpad}$ |
| **INT8 Microcontroller Arena Estimate** | 1,601 bytes | 1.56 KiB | $\text{INT8 Weights} + (2 \times 32\text{ B}) + 256\text{ B scratchpad}$ |
| **SRAM Budget Utilization (300 KiB)** | **1.668%** | **Fits in SRAM** | $(5,124\text{ B} / 307,200\text{ B}) \times 100$ |
| **Fits in ESP32-S3 SRAM (FP32)** | **True** | — | $5,124\text{ B} \le 307,200\text{ B}$ |
| **Fits in ESP32-S3 SRAM (INT8)** | **True** | — | $1,281\text{ B} \le 307,200\text{ B}$ |

### Comparison: Deployed 32–32 vs. Legacy 16–16 Baseline

Source: `edge_analysis/model_memory_comparison.csv`

| Metric | Legacy 16–16 Baseline | Deployed 32–32 Candidate |
|:---|:---:|:---:|
| **Architecture** | 5 &rarr; 16 &rarr; 16 &rarr; 1 | 5 &rarr; 32 &rarr; 32 &rarr; 1 |
| **Weights / Biases** | 352 / 33 | 1,216 / 65 |
| **Total Parameters** | 385 | 1,281 |
| **FP32 Storage** | 1,540 bytes (1.50 KiB) | 5,124 bytes (5.00 KiB) |
| **Peak Activation** | 64 bytes | 128 bytes |
| **MACs / FLOPs** | 352 / 704 | 1,216 / 2,432 |
| **FP32 Microcontroller Arena** | 1,924 bytes | 5,636 bytes |
| **SRAM Utilization (300 KiB)** | 0.501% | 1.668% |
| **Fits 300 KiB SRAM Budget** | True | True |

### Width Scaling Limits for 300 KiB SRAM
From `edge_analysis/width_comparison.csv`, scaling an equal-width network ($5 \rightarrow W \rightarrow W \rightarrow 1$):
- **Width 4 to 256**: All fit within the 300 KiB SRAM budget (Width 256 uses 264.00 KiB, 88.00%).
- **Width 512**: Exceeds the budget (1,040.00 KiB, 346.67% of SRAM).
- **Exact Theoretical Maximum Equal-Width**:
  - FP32 weights: **Width 273** (306,856 B / 307,200 B)
  - INT8 weights: **Width 550** (306,901 B / 307,200 B)

---

## 7. Benchmark Results

> [!IMPORTANT]
> **Understanding Benchmark Scope**:
> - **Direct Model Inference Benchmarks (PyTorch CPU)** measure raw tensor forward-pass computation in memory on a single CPU thread, isolating computational efficiency.
> - **End-to-End HTTP Benchmarks** measure complete client-server round-trip latency, including socket operations, TCP stack, FastAPI routing, Pydantic validation, feature transformation, inference, and JSON serialization.
> - Neither benchmark represents physical measurements from an ESP32-S3 microcontroller; both were conducted on a host x86 CPU.

### A. Direct Model Inference Benchmark (PyTorch CPU)
Measured via `edge_analysis/benchmark_model_direct.py` (1,000 single-inference iterations, single thread, synthetic valid telemetry):

| Benchmark Metric | Measured Value | Notes |
|:---|:---:|:---|
| **Model Load Time** | 38.43 ms | Loading weights and configuration into memory |
| **Single Inference Mean Latency** | **0.127 ms** ($127.0\text{ }\mu\text{s}$) | Average single-tensor forward pass |
| **Single Inference Median Latency** | **0.083 ms** ($82.6\text{ }\mu\text{s}$) | Median single-tensor forward pass |
| **Single Inference P90 Latency** | **0.164 ms** ($163.6\text{ }\mu\text{s}$) | 90th percentile latency |
| **Single Inference P95 Latency** | **0.269 ms** ($268.9\text{ }\mu\text{s}$) | 95th percentile latency |
| **Single Inference P99 Latency** | **0.786 ms** ($786.5\text{ }\mu\text{s}$) | 99th percentile latency |
| **Single-Sample Throughput** | **7,872 inferences/sec** | In-memory raw evaluation |
| **Batch 50 Throughput** | **568,259 records/sec** | Matrix-matrix multiply acceleration |
| **Batch 200 Throughput** | **2,466,669 records/sec** | Vectorized CPU tensor processing |
| **Batch 400 Throughput** | **4,909,000 records/sec** | Vectorized CPU tensor processing |

### B. End-to-End HTTP Batch Benchmark (400 Readings)
Evaluated using `benchmark_400.py` against the running FastAPI HTTP server:

| Evaluation Metric | Individual Requests | Batch Request | Outcome / Comparison |
|:---|:---:|:---:|:---|
| **Requests Sent** | 400 sequential POSTs | 1 POST with 400 rows | — |
| **Total Elapsed Time** | 819,066.462 ms (~819.1 s) | 2,060.93 ms (~2.06 s) | **397.43x Speedup** |
| **Average Latency / Item** | ~2,047.67 ms | ~5.15 ms | Eliminates 399 network round-trips |
| **Max Prediction Difference** | — | — | **0.000000 days (Exact match)** |

### C. End-to-End HTTP Batch Consistency Test (50 Readings)
Evaluated using `test_batch.py` against the live HTTP API:
- **Individual Requests**: 50 individual `/predict` calls
- **Batch Request**: 1 `/predict_batch` call with 50 rows
- **Maximum Difference**: **0.0 days** (all 50 predictions match identically)

---

## 8. API Endpoints Specification

The API exposes five endpoints:

```
GET  /               - API discovery and navigation metadata
GET  /health          - Service health, team metadata, parameter count, and validation MAE
GET  /model_info      - Layer dimensions, parameter counts, memory requirements, and SRAM fit
POST /predict         - Single sensor reading battery-life prediction
POST /predict_batch   - Batch sensor reading battery-life prediction
```

### Feature Names & Ordering
Incoming JSON objects must supply the five features matching the model's expected sequence:
1. `battery_v` (float)
2. `temp_c` (float)
3. `rings_per_day` (float)
4. `wifi_rssi_dbm` (float)
5. `age_months` (float)

### Response Diagnostics
Prediction responses include a `timings_ms` diagnostic object reporting execution phases:
- `parse`: JSON request payload deserialization time.
- `normalise`: Z-score standardization array transformation time.
- `inference`: PyTorch model forward-pass evaluation time.
- `build`: Prediction formatting, non-negative clipping (`np.maximum(y, 0.0)`), and response object construction.
- `server_total`: Total server-side execution time in milliseconds.

---

### Endpoint Documentation

#### 1. GET `/`
Returns top-level API navigation metadata:
```json
{
  "message": "EDGE AI Team 13 - Battery Life Prediction API",
  "docs": "/docs",
  "endpoints": ["/health", "/model_info", "/predict", "/predict_batch"]
}
```

#### 2. GET `/health`
Returns exact documented health fields:
```json
{
  "status": "ok",
  "team": 13,
  "team_members": [
    "Suraj",
    "Sujal Chavan",
    "Nandish",
    "Monika"
  ],
  "params": 1281,
  "val_mae": 5.22
}
```

#### 3. GET `/model_info`
Returns exact documented model architecture and edge memory metrics:
```json
{
  "layers": [
    {
      "name": "linear1",
      "in": 5,
      "out": 32,
      "params": 192,
      "bytes_f32": 768,
      "macs": 160
    },
    {
      "name": "linear2",
      "in": 32,
      "out": 32,
      "params": 1056,
      "bytes_f32": 4224,
      "macs": 1024
    },
    {
      "name": "linear3",
      "in": 32,
      "out": 1,
      "params": 33,
      "bytes_f32": 132,
      "macs": 32
    }
  ],
  "total_params": 1281,
  "total_bytes_f32": 5124,
  "total_bytes_int8": 1281,
  "total_macs": 1216,
  "fits_300kb_sram_f32": true,
  "fits_300kb_sram_int8": true
}
```

#### 4. POST `/predict`
Estimates battery days left for a single telemetry reading.

- **Request Body**:
```json
{
  "battery_v": 3.85,
  "temp_c": 24.5,
  "rings_per_day": 12.0,
  "wifi_rssi_dbm": -68.0,
  "age_months": 14.0
}
```

- **Response (`200 OK`)** *(prediction values are illustrative)*:
```json
{
  "days_left": 68.42,
  "timings_ms": {
    "parse": 0.042,
    "normalise": 0.038,
    "inference": 0.125,
    "build": 0.021,
    "server_total": 0.226
  }
}
```

#### 5. POST `/predict_batch`
Estimates battery days left for multiple telemetry readings in a single request (batch size: 1 to 10,000 rows).

- **Request Body**:
```json
{
  "rows": [
    {
      "battery_v": 4.10,
      "temp_c": 20.0,
      "rings_per_day": 5.0,
      "wifi_rssi_dbm": -55.0,
      "age_months": 2.0
    },
    {
      "battery_v": 3.65,
      "temp_c": 30.0,
      "rings_per_day": 20.0,
      "wifi_rssi_dbm": -78.0,
      "age_months": 22.0
    }
  ]
}
```

- **Response (`200 OK`)** *(prediction values are illustrative)*:
```json
{
  "days_left": [
    182.15,
    24.80
  ],
  "timings_ms": {
    "parse": 0.085,
    "normalise": 0.062,
    "inference": 0.180,
    "build": 0.035,
    "server_total": 0.362
  }
}
```

#### 6. Input Validation Errors (`422 Unprocessable Entity`)
If any input violates validation constraints (e.g., `battery_v` = 6.5 V, exceeding the 4.20 V ceiling), the API rejects the request without performing inference:
```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "battery_v"],
      "msg": "Input should be less than or equal to 4.2",
      "input": 6.5,
      "ctx": {"le": 4.2}
    }
  ]
}
```

---

## 9. Installation & Setup

### Prerequisites
- Python 3.10 to 3.13
- Git

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Suraj-S1108/edgeai-team13.git
   cd edgeai-team13
   ```

2. **Check out the active submitted branch**:
   ```bash
   # Switch to the final integration branch
   git checkout feature/final-32-32-integration
   ```

3. **Create and activate a virtual environment**:
   - **Windows PowerShell**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS / Bash**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

4. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 10. Starting the Backend API

Start the local development server using Uvicorn:

- **Windows PowerShell**:
  ```powershell
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  ```
- **Linux / macOS / Bash**:
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  ```

### Interactive Documentation & Health Checks
- **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI Schema (JSON)**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)
- **Service Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **Model Information**: [http://localhost:8000/model_info](http://localhost:8000/model_info)

---

## 11. Automated Testing

The automated test suite runs 17 self-contained regression tests verifying model loading, configuration schemas, normalization vectors, direct PyTorch inference, API endpoints, error handling, and file integrity.

### Run Automated Tests
```bash
python tests/test_application.py
```

### Verified Test Suite Output (17/17 Passed)
```
=================================================================
  EDGE AI Team 13 — Complete Automated Application Test Suite
=================================================================

[PASS] Test 01: Model loading (1,281 parameters)
[PASS] Test 02: Configuration loading
[PASS] Test 03: Feature normalisation statistics
[PASS] Test 04: Single prediction direct inference
[PASS] Test 05: Batch prediction direct inference
[PASS] Test 06: Out-of-range input rejection (HTTP 422)
[PASS] Test 07: Missing input field rejection (HTTP 422)
[PASS] Test 08: Incorrect input type rejection (HTTP 422)
[PASS] Test 09: Empty batch rejection (HTTP 422)
[PASS] Test 10: /health endpoint (exact documented schema)
[PASS] Test 11: /model_info endpoint (exact documented schema)
[PASS] Test 12: /predict endpoint contract & timings
[PASS] Test 13: /predict_batch endpoint contract
[PASS] Test 14: Root / endpoint returns API navigation information
[PASS] Test 15: Production 32-32 checkpoint tensor dimensions (5->32->32->1)
[PASS] Test 16: Memory calculation artifacts
[PASS] Test 17: Benchmark comparison artifacts

-----------------------------------------------------------------
  Summary: 17/17 tests passed (100.0%)
-----------------------------------------------------------------
```

---

## 12. Manual API Testing Instructions

### A. Windows PowerShell Testing

1. **Verify Health Endpoint**:
   ```powershell
   Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | ConvertTo-Json
   ```

2. **Verify Model Info Endpoint**:
   ```powershell
   Invoke-RestMethod -Uri "http://localhost:8000/model_info" -Method Get | ConvertTo-Json -Depth 4
   ```

3. **Test Valid Single Prediction**:
   ```powershell
   $body = @{
       battery_v = 3.85
       temp_c = 24.5
       rings_per_day = 12
       wifi_rssi_dbm = -68
       age_months = 14
   } | ConvertTo-Json

   Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json
   ```

4. **Test Out-of-Range Rejection (Expects HTTP 422)**:
   ```powershell
   # Battery voltage 6.5 V exceeds upper bound of 4.20 V
   try {
       $body = @{
           battery_v = 6.5
           temp_c = 24.5
           rings_per_day = 12
           wifi_rssi_dbm = -68
           age_months = 14
       } | ConvertTo-Json
       Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
   } catch {
       $_.Exception.Response.StatusCode
   }
   ```

5. **Run Batch Speedup Benchmark**:
   ```powershell
   python benchmark_400.py http://localhost:8000
   ```

6. **Run 50-Reading Batch Consistency Test**:
   ```powershell
   python test_batch.py http://localhost:8000
   ```

### B. Standard `curl` Testing (Bash / Linux / macOS)

1. **Check Service Health**:
   ```bash
   curl -s -X GET http://localhost:8000/health | jq .
   ```

2. **Check Model Architecture & Memory**:
   ```bash
   curl -s -X GET http://localhost:8000/model_info | jq .
   ```

3. **Predict Single Reading**:
   ```bash
   curl -s -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"battery_v": 3.85, "temp_c": 24.5, "rings_per_day": 12, "wifi_rssi_dbm": -68, "age_months": 14}' | jq .
   ```

4. **Test Out-of-Range Input Rejection (HTTP 422)**:
   ```bash
   curl -i -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"battery_v": 6.5, "temp_c": 24.5, "rings_per_day": 12, "wifi_rssi_dbm": -68, "age_months": 14}'
   ```

---

## 13. Project Directory Structure

```
edgeai-team13/
├── app/                                  # Active FastAPI application package
│   ├── __init__.py                       # Package initializer
│   ├── main.py                           # FastAPI endpoints, Pydantic schemas, server timing
│   └── model.py                          # Production BatteryNet architecture (5->32->32->1)
├── client/                               # Benchmark utilities
│   └── bench.py                          # Client benchmarking utility
├── data/                                 # Datasets
│   ├── team_13_train.csv                 # 2,000 raw training records (contains 24 glitches)
│   └── team_13_train_cleaned.csv         # 1,976 sanitized training records
├── docs/                                 # Technical documentation
│   ├── api_documentation.md              # Detailed API specification and contract
│   ├── benchmarks_report.md              # Extended benchmark findings
│   ├── manual_testing_guide.md           # Step-by-step verification commands
│   ├── memory_analysis_report.md         # ESP32-S3 edge hardware memory breakdown
│   ├── mission5_report.md                # Mission 5 dataset cleaning & model selection
│   └── model_comparison_report.md        # Comprehensive multi-model evaluation
├── edge_analysis/                        # Edge AI scripts, artifacts & historical evidence
│   ├── benchmark_model_direct.py         # Direct PyTorch CPU latency & throughput benchmarker
│   ├── memory_calculations.py            # Edge AI resource and memory calculation engine
│   ├── mission3_report.md                # Mission 3 latency report
│   ├── mission4_report.md                # Mission 4 hardware calculation report
│   ├── mission5_32_vs_16_prediction.ipynb # Residual analysis between 32-32 and 16-16 models
│   ├── mission5_common_validation_comparison.csv
│   ├── mission5_depth_comparison.csv
│   ├── mission5_model_comparison.csv
│   ├── model_benchmark_comparison.csv    # Benchmark metrics comparing 16-16 and 32-32
│   ├── model_memory_comparison.csv       # Memory metrics comparing 16-16 and 32-32
│   ├── width_comparison.csv              # Width scaling analysis (widths 4 to 2048)
│   └── mission5_results/                 # Mission 5 training artifacts
│       ├── models/                       # Historical experiment checkpoints (1L & 2L candidates)
│       │   ├── Cleaned_NN_1L_*.pt
│       │   ├── Cleaned_NN_2L_*.pt        # Experimental candidates (including 16_16 and 64_64)
│       │   └── Original_NN_*.pt          # Baselines trained on uncleaned data
│       ├── cleaning_audit.csv            # Glitch row indices and values
│       ├── mission5_complete_model_comparison.csv
│       ├── mission5_train_validation_complete.csv
│       └── *.png                         # Training loss curves and comparison charts
├── model/                                # Active production model artifacts
│   ├── config.json                       # Active 32-32 configuration, scaler parameters, MAE
│   └── model.pt                          # Active 32-32 PyTorch weights (1,281 parameters)
├── tests/                                # Automated test suite
│   └── test_application.py               # 17-point regression and verification test suite
├── .devcontainer/                        # Dev container environment definition
│   └── devcontainer.json
├── .gitignore                            # Git exclusion rules
├── benchmark_400.py                      # 400-record batch speedup benchmark script
├── mission1_dataset_analysis.ipynb       # Mission 1 dataset exploration notebook
├── mission2_neural_network_experiments.ipynb # Mission 2 initial architecture experiments
├── mission5_analysis.ipynb               # Mission 5 data cleaning analysis notebook
├── mission5_model_training.ipynb         # Mission 5 retraining and architecture evaluation
├── requirements.txt                      # Project Python dependencies
├── test_batch.py                         # 50-reading batch vs individual consistency test
├── train.py                              # Reproducible model training pipeline
└── README.md                             # Master project documentation
```

---

## 14. Limitations & Future Work

### Limitations
1. **Single Train/Validation Split**: Validation was performed on a single 80/20 split (seed 42, 395 validation records). While validation samples were strictly held out, nested $k$-fold cross-validation would provide tighter confidence bounds across varied edge distributions.
2. **Extrapolation Beyond Cleaned Data Boundaries**: Input validation ranges allow realistic sensor values (e.g. temperatures between 0 °C and 8 °C, or 46 °C and 85 °C) that extend beyond the empirical training set window ($[8.0, 46.0]$ °C). Model accuracy on readings outside the training support has no empirically established error bound.
3. **Host x86 CPU Execution**: Benchmarks and latency figures were measured on a host x86 CPU running CPython and PyTorch. Memory metrics are theoretical calculations and do not account for physical microcontroller runtime overhead (e.g. ESP-IDF RTOS stack, FreeRTOS task context, Wi-Fi driver heap allocation).

### Future Work
1. **Post-Training Quantization (PTQ)**: Quantize FP32 weights to INT8 using PyTorch or ONNX, reducing storage from 5,124 bytes to 1,281 bytes and enabling SIMD acceleration on ESP32-S3 vector instructions.
2. **Bare-Metal Microcontroller Deployment**: Compile the network into a C-array using TensorFlow Lite for Microcontrollers (TFLM) or Espressif ESP-DL to measure true cycle counts and flash footprint on physical ESP32-S3 silicon.
3. **On-Device Anomaly Filtering**: Embed lightweight sensor sanity checks (rejecting $\le 0\text{ V}$, $-127\text{ }^\circ\text{C}$, and $0\text{ dBm}$) into the firmware sensor read loop prior to invoking the neural network.
