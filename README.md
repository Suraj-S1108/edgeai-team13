# EDGE AI — Team 13
## How Long Will the Battery Last?

**AI255TBD Edge AI · Module 1 Assignment: Neural Networks from Scratch to Deployment**

A neural-network-based battery-life prediction system for a smart doorbell.

The objective is to predict the remaining battery life in days using sensor readings. The trained model is served through a FastAPI application that provides individual predictions, batch predictions, model information and request-stage timing measurements.

---

# 1. Team Information

**Team Number:** 13

**Team Members:**
1. Suraj
2. Sujal Chavan
3. Nandish
4. Monika

---

# 2. Project Overview

The doorbell company needs an accurate estimate of how many days of battery life remain instead of a simple low-battery warning.

This project uses a small feed-forward neural network trained on Team 13's battery dataset.

The application provides:

- Battery-life predictions in days.
- Input validation to reject invalid sensor readings.
- Individual and batch prediction endpoints.
- Per-layer model information.
- Model parameter, memory and MAC calculations.
- Request-stage timing measurements.
- Analysis of the model's suitability for edge deployment.

The API is designed to run in GitHub Codespaces and can be accessed through a public port 8000 URL.

---

# 3. Repository Structure

| Path                                  | Description                                       |
| ------------------------------------- | ------------------------------------------------- |
| `app/`                                | FastAPI application and neural-network model       |
| `app/main.py`                         | API endpoints, validation, inference and timing   |
| `app/model.py`                        | Neural-network architecture                       |
| `client/`                             | API benchmarking client                           |
| `client/bench.py`                     | Measures API response times and stage timings     |
| `data/`                               | Training datasets                                 |
| `model/`                              | Trained model weights and configuration           |
| `model/model.pt`                      | Saved neural-network weights                      |
| `model/config.json`                   | Model architecture and normalization configuration |
| `edge_analysis/`                      | Mission 3 and Mission 4 analysis                  |
| `edge_analysis/mission3_report.md`    | Performance benchmarking report                   |
| `edge_analysis/mission4_report.md`    | Edge memory and deployment analysis               |
| `edge_analysis/memory_calculations.py`| Model memory calculations                         |
| `edge_analysis/width_comparison.csv`  | Model-width comparison results                    |
| `train.py`                            | Model training script                             |
| `test_batch.py`                       | Batch prediction consistency test                 |
| `benchmark_400.py`                    | Individual versus batch benchmark                 |
| `requirements.txt`                    | Python dependencies                               |
| `README.md`                           | Project documentation                             |

---

# 4. Dataset Information

**Dataset:** `data/team_13_train.csv`

The dataset contains 2,000 rows and six columns.

### Input features

| Feature         | Description                     | Unit   |
| --------------- | ------------------------------- | ------ |
| `battery_v`     | Battery voltage                 | Volts  |
| `temp_c`        | Temperature                     | °C     |
| `rings_per_day` | Number of doorbell rings per day | Count  |
| `wifi_rssi_dbm` | Wi-Fi signal strength           | dBm    |
| `age_months`    | Device age                      | Months |

**Target variable:** `days_left`

The target represents the estimated remaining battery life in days.

Only Team 13's dataset is used for model training and evaluation.

---

# 5. Neural Network Architecture

The final trained neural network uses the following architecture:

**5 → 16 → 16 → 1**

| Layer    | Input | Output |
| -------- | ----: | -----: |
| Input    |     5 |      5 |
| Linear 1 |     5 |     16 |
| Linear 2 |    16 |     16 |
| Linear 3 |    16 |      1 |

The network uses ReLU activation functions between its hidden layers.

### Model summary

| Metric                                  |     Result |
| --------------------------------------- | ---------: |
| Input features                          |          5 |
| Hidden layers                           |          2 |
| Hidden neurons                          |     16, 16 |
| Output neurons                          |          1 |
| Total parameters                        |        385 |
| Validation MAE                          |  6.42 days |
| Total MACs                              |        352 |
| Float32 weight and bias storage         | 1,540 bytes |
| Theoretical INT8 weight and bias storage|  385 bytes |

The validation MAE of 6.42 days is measured on the project's validation data. It is not the final MAE on the instructor's 400 hidden readings.

---

# 6. API Endpoints

The application is built using FastAPI.

Start the API from the repository root:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

### 6.1 Health endpoint

**GET `/health`**

Returns the API status, team information and model details.

Example response:

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
  "params": 385,
  "val_mae": 6.42
}
```

### 6.2 Model information endpoint

**GET `/model_info`**

Returns the per-layer parameter counts, float32 storage sizes, MACs and theoretical INT8 storage information.

Example response:

```json
{
  "layers": [
    {
      "name": "linear1",
      "in": 5,
      "out": 16,
      "params": 96,
      "bytes_f32": 384,
      "macs": 80
    },
    {
      "name": "linear2",
      "in": 16,
      "out": 16,
      "params": 272,
      "bytes_f32": 1088,
      "macs": 256
    },
    {
      "name": "linear3",
      "in": 16,
      "out": 1,
      "params": 17,
      "bytes_f32": 68,
      "macs": 16
    }
  ],
  "total_params": 385,
  "total_bytes_f32": 1540,
  "total_bytes_int8": 385,
  "total_macs": 352,
  "fits_300kb_sram_f32": true,
  "fits_300kb_sram_int8": true
}
```

The INT8 storage values are theoretical calculations. An actual quantized model has not been implemented or accuracy-tested.

### 6.3 Individual prediction endpoint

**POST `/predict`**

Accepts one sensor reading and returns the estimated remaining battery life.

Example request:

```json
{
  "battery_v": 3.86,
  "temp_c": 28,
  "rings_per_day": 5,
  "wifi_rssi_dbm": -66,
  "age_months": 17
}
```

Example response:

```json
{
  "days_left": 142.77,
  "timings_ms": {
    "parse": 0.004,
    "normalise": 1.673,
    "inference": 677.356,
    "build": 0.138,
    "server_total": 679.201
  }
}
```

Timing values shown above are from an individual test request and can vary between executions.

### 6.4 Batch prediction endpoint

**POST `/predict_batch`**

Accepts multiple sensor readings in a single request.

Example request:

```json
{
  "rows": [
    {
      "battery_v": 3.86,
      "temp_c": 28,
      "rings_per_day": 5,
      "wifi_rssi_dbm": -66,
      "age_months": 17
    }
  ]
}
```

Returns a list of battery-life predictions and timing information.

### 6.5 Input validation

The API validates sensor readings using defined input ranges.

| Feature         |    Minimum |    Maximum |
| --------------- | ---------: | ---------: |
| Battery voltage |        0 V |        5 V |
| Temperature     |     -40 °C |      85 °C |
| Rings per day   |          0 |       1000 |
| Wi-Fi RSSI      |   -100 dBm |      0 dBm |
| Device age      |   0 months | 240 months |

Invalid readings are rejected with HTTP 422 responses rather than receiving predictions.

---

# 7. Mission 1 — Dataset Analysis and Baselines

Mission 1 focuses on understanding the dataset, examining feature relationships and establishing baseline prediction performance.

The project includes dataset analysis and neural-network experimentation notebooks:

- `mission1_dataset_analysis.ipynb`
- `mission2_neural_network_experiments.ipynb`

These notebooks contain the exploratory analysis and model experimentation work.

---

# 8. Mission 2 — Neural Network Training

Mission 2 focuses on neural-network architecture, input normalization, training and model evaluation.

The final selected model is:

**5 → 16 → 16 → 1**

The trained weights and configuration are saved in:

- `model/model.pt`
- `model/config.json`

The model contains 385 trainable parameters and achieves a validation MAE of 6.42 days.

---

# 9. Mission 3 — Performance Analysis and Benchmarking

Mission 3 investigates the time required to process battery-life prediction requests.

The experiments examine:

- Laptop-to-localhost request timing.
- Keep-alive versus no-keepalive requests.
- Laptop-to-GitHub-Codespace request timing.
- Codespace-localhost request timing.
- Individual versus batch prediction performance.
- Parsing, normalization, inference and response-building stages.

### Verified benchmark results

| Experiment                                               |        Result |
| -------------------------------------------------------- | ------------: |
| Laptop localhost, keep-alive p50 round trip              |      1.978 ms |
| Laptop localhost, no-keepalive p50 round trip            |  2,049.505 ms |
| Laptop to Codespace p50 round trip                       |     34.024 ms |
| Codespace localhost p50 round trip                       |      1.989 ms |
| 400 individual versus one batch request                  | 47.76× speedup |
| Maximum prediction difference in 400-reading comparison  |     0.01 days |

The 400-reading comparison showed that batch predictions matched individual predictions within the benchmark's 0.01-day tolerance.

A separate 50-reading test also produced a maximum prediction difference of 0.0 days.

The network-and-framework timing is calculated as round-trip time minus measured server time. It represents everything outside the measured server stages, not pure network latency alone.

**Detailed report:** [Mission 3 — Performance Analysis](edge_analysis/mission3_report.md)

---

# 10. Mission 4 — Edge AI Memory and Deployment Analysis

Mission 4 evaluates whether the selected neural network could fit within the memory constraints of an ESP32-S3-based doorbell.

The analysis covers:

- Float32 model storage.
- Theoretical INT8 model storage.
- Activation memory estimates.
- Model-width comparisons.
- SRAM and PSRAM capacity.
- MAC calculations.
- Theoretical inference-time estimates.
- Cloud versus edge deployment considerations.

### Selected model memory summary

| Metric                                         |      Result |
| ---------------------------------------------- | ----------: |
| Total parameters                               |         385 |
| Float32 weight and bias storage                | 1,540 bytes |
| Theoretical INT8 storage                       |   385 bytes |
| Total MACs                                     |         352 |
| Largest single hidden-layer activation estimate|    64 bytes |

The activation estimate represents the largest individual hidden-layer output buffer under the calculation assumptions. It is not a measurement of complete runtime RAM usage.

The memory-fit calculations compare model weights and biases against the specified memory budgets. They do not include all application runtime, stack, scratch-buffer or framework overhead.

Theoretical INT8 storage does not mean that the deployed model has already been quantized.

No actual ESP32-S3 hardware inference benchmark has been performed.

**Detailed report:** [Mission 4 — Edge Memory and Deployment Analysis](edge_analysis/mission4_report.md)

---

# 11. Testing and Verification

The following checks have been performed locally.

| Test                                            | Result                            |
| ----------------------------------------------- | --------------------------------- |
| `/health` endpoint                              | Verified                          |
| Team number and parameter count                 | Verified                          |
| Team member information                         | Verified locally                  |
| `/model_info` layer calculations                | Verified                          |
| Invalid battery voltage                         | HTTP 422                          |
| Invalid temperature                             | HTTP 422                          |
| Invalid device age                              | HTTP 422                          |
| Invalid rings per day                           | HTTP 422                          |
| Invalid Wi-Fi RSSI                              | HTTP 422                          |
| `/predict` timing fields                        | Verified                          |
| 50-reading batch consistency                    | Passed                            |
| 400-reading individual versus batch comparison  | Passed within 0.01-day tolerance  |

### 50-reading batch consistency test

Run:

```bash
python test_batch.py http://localhost:8000
```

The test generates 50 reproducible readings, sends them through `/predict_batch`, sends the same readings individually through `/predict`, and compares the predictions.

Verified result:

```text
Batch predictions: 50
Individual predictions: 50
Maximum prediction difference: 0.0 days

SUCCESS: All 50 batch predictions match individual predictions.
```

### 400-reading benchmark

Run:

```bash
python benchmark_400.py
```

This benchmark compares individual predictions against batch predictions for 400 readings.

Verified result:

- Individual requests: 400
- Batch predictions: 400
- Speedup: 47.76×
- Maximum prediction difference: 0.01 days
- Result: SUCCESS within the script's tolerance.

**Important:** The instructor's separate 400 hidden readings have not yet been provided. Their final evaluation MAE remains pending.

---

# 12. Installation and Setup

### Requirements

- Python 3
- PyTorch
- FastAPI
- Uvicorn
- NumPy
- GitHub Codespaces or a local Python environment

### Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

### Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Open the API documentation

```text
http://localhost:8000/docs
```

### Test the health endpoint

```text
http://localhost:8000/health
```

### Test model information

```text
http://localhost:8000/model_info
```

### Run the batch test

```bash
python test_batch.py http://localhost:8000
```

### Run the client benchmark

```bash
python client/bench.py --url YOUR_API_ADDRESS --n 50
```

---

# 13. GitHub Codespaces Deployment

The application can be hosted using GitHub Codespaces.

1. Open the project repository on GitHub.
2. Create or open a Codespace on the required branch.
3. Wait for dependencies to install.
4. Start Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

5. Open the **Ports** tab.
6. Locate port `8000`.
7. Set its visibility to **Public**.
8. Copy the public forwarding URL.

The public endpoint will have a format similar to:

```text
https://YOUR-CODESPACE-8000.app.github.dev
```

Test:

```text
YOUR_API_ADDRESS/health
YOUR_API_ADDRESS/model_info
YOUR_API_ADDRESS/docs
```

After a Codespace restart, start Uvicorn again and check port 8000 visibility.

Keep the Codespace running during the instructor's evaluation window.

---

# 14. Final Submission Requirements

The assignment evaluation requires:

1. A working public API endpoint.
2. A GitHub repository link.
3. The trained `model/` directory committed to the repository.
4. Correct `/health` information.
5. Accurate `/model_info` calculations.
6. Correct rejection of five invalid readings.
7. All required timing fields in prediction responses.
8. Batch predictions matching individual predictions.
9. Acceptable accuracy on the instructor's 400 hidden readings.

The hidden-reading accuracy test will be performed when the instructor provides the evaluation readings.

The public endpoint must remain accessible during the instructor's check window.

---

# 15. Project Summary

Team 13 developed a neural-network-based battery-life prediction API using a trained feed-forward neural network.

The final model uses a 5 → 16 → 16 → 1 architecture with 385 parameters and a validation MAE of 6.42 days.

The application provides individual and batch predictions, input validation, model information and stage-level timing measurements.

The performance and edge-deployment analyses examine the trade-offs between server-based inference and running a compact model on an embedded device.

The project includes the trained model, API implementation, benchmarking scripts, experimental results and detailed Mission 3 and Mission 4 reports.

---

**Team 13 — EDGE AI**

*Predict honestly. Measure carefully. Explain everything.*
