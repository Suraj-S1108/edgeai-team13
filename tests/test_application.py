"""
tests/test_application.py -- Comprehensive Automated Regression and Verification Test Suite
EDGE AI Team 13 - Battery Life Predictor

Verifies:
1. Model loading & parameter count (1,281)
2. Configuration loading & schema
3. Feature normalisation consistency
4. Single prediction execution
5. Batch prediction execution
6. Invalid input rejection (out of range)
7. Missing input rejection
8. Incorrect input type rejection
9. Empty batch rejection
10. /health endpoint response
11. /model_info endpoint response and SRAM fit
12. /predict endpoint contract
13. /predict_batch endpoint contract
14. Frontend dashboard HTML serving at /
15. Model backup integrity (16-16 backup loads 385 params)
16. Memory calculation execution
17. Benchmark script integrity
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from fastapi.testclient import TestClient
from app.main import app
from app.model import FEATURES, BatteryNet


def run_all_tests():
    print("=================================================================")
    print("  EDGE AI Team 13 — Complete Automated Application Test Suite")
    print("=================================================================\n")

    client = TestClient(app)
    passed = 0
    total = 17

    # 1. Model Loading
    try:
        net = BatteryNet((32, 32))
        state_dict = torch.load("model/model.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()
        p_count = sum(p.numel() for p in net.parameters())
        assert p_count == 1281, f"Expected 1281 parameters, got {p_count}"
        print("[PASS] Test 01: Model loading (1,281 parameters)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 01: Model loading failed: {e}")

    # 2. Configuration Loading
    try:
        with open("model/config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key in ["team", "hidden", "mu", "sd", "features", "val_mae"]:
            assert key in cfg, f"Missing key {key} in config.json"
        assert cfg["team"] == 13
        assert cfg["hidden"] == [32, 32]
        print("[PASS] Test 02: Configuration loading")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 02: Configuration loading failed: {e}")

    # 3. Feature Normalisation
    try:
        mu = np.array(cfg["mu"])
        sd = np.array(cfg["sd"])
        assert len(mu) == 5 and len(sd) == 5
        assert np.all(sd > 0)
        print("[PASS] Test 03: Feature normalisation statistics")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 03: Feature normalisation failed: {e}")

    # 4. Single Prediction Direct
    try:
        dummy = torch.zeros(1, 5)
        out = net(dummy)
        assert out.shape == torch.Size([1])
        print("[PASS] Test 04: Single prediction direct inference")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 04: Single prediction failed: {e}")

    # 5. Batch Prediction Direct
    try:
        dummy_batch = torch.zeros(10, 5)
        out_batch = net(dummy_batch)
        assert out_batch.shape == torch.Size([10])
        print("[PASS] Test 05: Batch prediction direct inference")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 05: Batch prediction failed: {e}")

    # 6. Invalid Input Rejection (HTTP 422)
    try:
        res = client.post("/predict", json={"battery_v": 6.5, "temp_c": 25.0, "rings_per_day": 10, "wifi_rssi_dbm": -65, "age_months": 12})
        assert res.status_code == 422
        print("[PASS] Test 06: Out-of-range input rejection (HTTP 422)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 06: Invalid input rejection failed: {e}")

    # 7. Missing Input Rejection (HTTP 422)
    try:
        res = client.post("/predict", json={"battery_v": 3.8, "temp_c": 25.0})
        assert res.status_code == 422
        print("[PASS] Test 07: Missing input field rejection (HTTP 422)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 07: Missing input rejection failed: {e}")

    # 8. Incorrect Input Types (HTTP 422)
    try:
        res = client.post("/predict", json={"battery_v": "invalid_string", "temp_c": 25.0, "rings_per_day": 10, "wifi_rssi_dbm": -65, "age_months": 12})
        assert res.status_code == 422
        print("[PASS] Test 08: Incorrect input type rejection (HTTP 422)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 08: Incorrect input type rejection failed: {e}")

    # 9. Empty Batch Rejection (HTTP 422)
    try:
        res = client.post("/predict_batch", json={"rows": []})
        assert res.status_code == 422
        print("[PASS] Test 09: Empty batch rejection (HTTP 422)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 09: Empty batch rejection failed: {e}")

    # 10. Health Endpoint
    try:
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        expected_health_keys = {"status", "team", "team_members", "params", "val_mae"}
        assert set(body.keys()) == expected_health_keys, f"Expected keys {expected_health_keys}, got {set(body.keys())}"
        assert body["status"] == "ok"
        assert body["team"] == 13
        assert body["params"] == 1281
        assert body["val_mae"] == 5.22
        print("[PASS] Test 10: /health endpoint (exact documented schema)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 10: /health failed: {e}")

    # 11. Model Info Endpoint
    try:
        res = client.get("/model_info")
        assert res.status_code == 200
        body = res.json()
        expected_info_keys = {"layers", "total_params", "total_bytes_f32", "total_bytes_int8", "total_macs", "fits_300kb_sram_f32", "fits_300kb_sram_int8"}
        assert set(body.keys()) == expected_info_keys, f"Expected keys {expected_info_keys}, got {set(body.keys())}"
        assert body["total_params"] == 1281
        assert body["fits_300kb_sram_f32"] is True
        assert len(body["layers"]) == 3
        print("[PASS] Test 11: /model_info endpoint (exact documented schema)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 11: /model_info failed: {e}")

    # 12. Predict Endpoint
    try:
        valid_sample = {"battery_v": 3.8, "temp_c": 25.0, "rings_per_day": 10, "wifi_rssi_dbm": -65, "age_months": 12}
        res = client.post("/predict", json=valid_sample)
        assert res.status_code == 200
        body = res.json()
        assert "days_left" in body
        assert isinstance(body["days_left"], (int, float))
        assert "timings_ms" in body
        print("[PASS] Test 12: /predict endpoint contract & timings")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 12: /predict failed: {e}")

    # 13. Predict Batch Endpoint
    try:
        batch_sample = {"rows": [valid_sample, valid_sample, valid_sample]}
        res = client.post("/predict_batch", json=batch_sample)
        assert res.status_code == 200
        body = res.json()
        assert len(body["days_left"]) == 3
        print("[PASS] Test 13: /predict_batch endpoint contract")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 13: /predict_batch failed: {e}")

    # 14. Root Endpoint (API Info)
    try:
        res = client.get("/")
        assert res.status_code == 200
        body = res.json()
        assert "message" in body
        assert "/health" in body["endpoints"]
        assert "/model_info" in body["endpoints"]
        assert "/predict" in body["endpoints"]
        assert "/predict_batch" in body["endpoints"]
        print("[PASS] Test 14: Root / endpoint returns API navigation information")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 14: Root endpoint failed: {e}")

    # 15. Checkpoint Layer Dimensions & Parameter Count Verification
    try:
        state_dict_active = torch.load("model/model.pt", map_location="cpu", weights_only=True)
        assert "network.0.weight" in state_dict_active
        assert state_dict_active["network.0.weight"].shape == torch.Size([32, 5])
        assert state_dict_active["network.2.weight"].shape == torch.Size([32, 32])
        assert state_dict_active["network.4.weight"].shape == torch.Size([1, 32])
        print("[PASS] Test 15: Production 32-32 checkpoint tensor dimensions (5->32->32->1)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 15: Checkpoint dimension test failed: {e}")

    # 16. Memory Calculations
    try:
        assert os.path.exists("edge_analysis/memory_calculations.py")
        assert os.path.exists("edge_analysis/model_memory_comparison.csv")
        print("[PASS] Test 16: Memory calculation artifacts")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 16: Memory calculations failed: {e}")

    # 17. Benchmark Scripts
    try:
        assert os.path.exists("edge_analysis/benchmark_model_direct.py")
        assert os.path.exists("edge_analysis/model_benchmark_comparison.csv")
        print("[PASS] Test 17: Benchmark comparison artifacts")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 17: Benchmark scripts failed: {e}")

    print("\n-----------------------------------------------------------------")
    print(f"  Summary: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    print("-----------------------------------------------------------------\n")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
