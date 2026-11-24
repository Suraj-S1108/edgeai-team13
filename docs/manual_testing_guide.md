# EDGE AI Team 13 — Manual Verification & Testing Guide

This guide provides exact, step-by-step instructions for manually running and evaluating the EDGE AI Team 13 Battery Life Prediction API.

---

## 1. Environment Setup

Verify dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Start the Backend Server

Start Uvicorn on port 8000:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Expected terminal output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

---

## 3. Test API via Swagger UI (Interactive Browser)

Open your browser to:
- **`http://localhost:8000/docs`** (Interactive Swagger UI)
- **`http://localhost:8000/redoc`** (ReDoc Specification)

You can click **"Try it out"** on any endpoint (`/health`, `/model_info`, `/predict`, `/predict_batch`) to execute live requests directly from your browser.

---

## 4. Test API Endpoints via cURL or PowerShell

### A. Root Endpoint (`GET /`)
```bash
curl -X GET http://localhost:8000/
```
**Expected Response**:
```json
{
  "message": "EDGE AI Team 13 - Battery Life Prediction API",
  "docs": "/docs",
  "endpoints": [
    "/health",
    "/model_info",
    "/predict",
    "/predict_batch"
  ]
}
```

### B. Health Endpoint (`GET /health`)
```bash
curl -X GET http://localhost:8000/health
```
**Expected Response**:
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

### C. Model Info Endpoint (`GET /model_info`)
```bash
curl -X GET http://localhost:8000/model_info
```
**Expected Response**:
- Contains `layers` with 3 linear layers
- `total_params`: 1281
- `total_bytes_f32`: 5124
- `total_bytes_int8`: 1281
- `total_macs`: 1216
- `fits_300kb_sram_f32`: true
- `fits_300kb_sram_int8`: true

### D. Single Prediction (`POST /predict` — Valid)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"battery_v\": 3.85, \"temp_c\": 24.5, \"rings_per_day\": 12, \"wifi_rssi_dbm\": -68, \"age_months\": 14}"
```
**Expected Response**:
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

### E. Single Prediction (`POST /predict` — Invalid Rejection)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"battery_v\": 6.5, \"temp_c\": 25.0, \"rings_per_day\": 10, \"wifi_rssi_dbm\": -65, \"age_months\": 12}"
```
**Expected Response**:
- HTTP Status: `422 Unprocessable Entity`

### F. Batch Prediction (`POST /predict_batch`)
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d "{\"rows\": [{\"battery_v\": 4.1, \"temp_c\": 20.0, \"rings_per_day\": 5, \"wifi_rssi_dbm\": -55, \"age_months\": 2}, {\"battery_v\": 3.5, \"temp_c\": 30.0, \"rings_per_day\": 25, \"wifi_rssi_dbm\": -80, \"age_months\": 24}]}"
```
**Expected Response**:
- Returns an array with 2 predictions in `days_left` and batch timings.

---

## 5. Automated Evaluation Scripts

### Automated Regression Suite:
```bash
python tests/test_application.py
```

### Batch Speedup Benchmark (requires server running):
```bash
python benchmark_400.py http://localhost:8000
```

### Direct Model CPU Benchmark:
```bash
python edge_analysis/benchmark_model_direct.py
```

### Memory & Hardware Resource Analysis:
```bash
python edge_analysis/memory_calculations.py
```
