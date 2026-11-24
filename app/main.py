"""
app/main.py -- EDGE AI Team 13 Battery-Life Prediction API

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /
        Returns API information and endpoint links.

    GET  /health
        Returns API status, team number, model parameters and validation MAE.

    GET  /model_info
        Returns layer details, parameters, memory estimates, MACs and SRAM fit.

    POST /predict
        Predicts remaining battery life for one sensor reading.

    POST /predict_batch
        Predicts remaining battery life for multiple sensor readings.

Input validation:
    Out-of-range sensor readings are rejected with HTTP 422.
    The API does not return predictions for invalid readings.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.model import FEATURES, BatteryNet

CONFIG_PATH = Path("model/config.json")
MODEL_PATH = Path("model/model.pt")

cfg = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU = np.array(cfg["mu"], np.float32)
SD = np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    battery_v: float = Field(ge=2.50, le=4.20)
    temp_c: float = Field(ge=0, le=85)
    rings_per_day: float = Field(ge=0, le=1000)
    wifi_rssi_dbm: float = Field(ge=-100, le=-1)
    age_months: float = Field(ge=0, le=240)

class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)


@app.get("/")
def root():
    return {
        "message": "EDGE AI Team 13 - Battery Life Prediction API",
        "docs": "/docs",
        "endpoints": ["/health", "/model_info", "/predict", "/predict_batch"]
    }


@app.get("/model_info")
def model_info():
    layers = []

    total_params = 0
    total_bytes_f32 = 0
    total_bytes_int8 = 0
    total_macs = 0

    linear_number = 0

    # Inspect the actual network and count Linear layers.
    for layer in net.network:
        if isinstance(layer, torch.nn.Linear):
            linear_number += 1

            in_features = layer.in_features
            out_features = layer.out_features

            # Parameters = weights + biases
            params = sum(p.numel() for p in layer.parameters())

            # Storage calculations
            bytes_f32 = params * 4
            bytes_int8 = params * 1

            # MACs = input features * output features
            macs = in_features * out_features

            layers.append({
                "name": f"linear{linear_number}",
                "in": in_features,
                "out": out_features,
                "params": params,
                "bytes_f32": bytes_f32,
                "macs": macs
            })

            total_params += params
            total_bytes_f32 += bytes_f32
            total_bytes_int8 += bytes_int8
            total_macs += macs

    # ESP32-S3 SRAM budget: approximately 300 KB.
    SRAM_BUDGET = 300 * 1024

    return {
        "layers": layers,
        "total_params": total_params,
        "total_bytes_f32": total_bytes_f32,
        "total_bytes_int8": total_bytes_int8,
        "total_macs": total_macs,
        "fits_300kb_sram_f32": total_bytes_f32 <= SRAM_BUDGET,
        "fits_300kb_sram_int8": total_bytes_int8 <= SRAM_BUDGET
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "team": cfg["team"],
        "team_members": [
            "Suraj",
            "Sujal Chavan",
            "Nandish",
            "Monika"
        ],
        "params": N_PARAMS,
        "val_mae": cfg["val_mae"]
    }


async def run(data, schema):
    t_all = time.perf_counter()
    t = {}

    t0 = time.perf_counter()
    rows = data.rows if isinstance(data, Batch) else [data]
    t["parse"] = ms(t0)

    t0 = time.perf_counter()
    x = (
        np.array(
            [[getattr(r, f) for f in FEATURES] for r in rows],
            np.float32
        ) - MU
    ) / SD
    t["normalise"] = ms(t0)

    t0 = time.perf_counter()
    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()
    t["inference"] = ms(t0)

    t0 = time.perf_counter()
    y_clipped = np.maximum(y, 0.0)
    if y_clipped.ndim == 0:
        days = [round(float(y_clipped), 2)]
    else:
        days = [round(float(v), 2) for v in y_clipped]

    body = {
        "days_left": days if isinstance(data, Batch) else days[0]
    }
    t["build"] = ms(t0)

    t["server_total"] = ms(t_all)
    body["timings_ms"] = t

    return body


@app.post("/predict")
async def predict(data: Reading):
    return await run(data, Reading)


@app.post("/predict_batch")
async def predict_batch(data: Batch):
    return await run(data, Batch)
