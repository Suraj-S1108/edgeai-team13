
"""
app/main.py -- Doorbell battery prediction API.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health
GET  /model_info
POST /predict
POST /predict_batch
"""

import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, BatteryNet


# --------------------------------------------------
# Load trained model and configuration
# --------------------------------------------------

cfg = json.load(open("model/config.json"))

net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(
    torch.load("model/model.pt", weights_only=True)
)

net.eval()
torch.set_num_threads(1)

MU = np.array(cfg["mu"], np.float32)
SD = np.array(cfg["sd"], np.float32)

N_PARAMS = sum(p.numel() for p in net.parameters())


# --------------------------------------------------
# Input validation
# --------------------------------------------------

class Reading(BaseModel):
    battery_v: float = Field(ge=0.0, le=5.0)
    temp_c: float = Field(ge=-40.0, le=85.0)
    rings_per_day: float = Field(ge=0, le=1000)
    wifi_rssi_dbm: float = Field(ge=-100, le=0)
    age_months: float = Field(ge=0, le=240)


class Batch(BaseModel):
    rows: list[Reading] = Field(
        min_length=1,
        max_length=10000
    )


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(title="Doorbell battery API")

ms = lambda t0: round(
    (time.perf_counter() - t0) * 1000, 3
)


# --------------------------------------------------
# Reusable model memory calculation
# --------------------------------------------------

def calculate_model_info(layer_sizes):
    """
    Calculate parameters, memory storage and MACs
    for a fully connected neural network.

    layer_sizes example:
        [5, 16, 16, 1]
    """

    layers = []

    total_params = 0
    total_bytes_f32 = 0
    total_bytes_int8 = 0
    total_macs = 0

    for i in range(len(layer_sizes) - 1):

        in_features = layer_sizes[i]
        out_features = layer_sizes[i + 1]

        # Weights + biases
        params = (
            in_features * out_features
        ) + out_features

        # Theoretical storage
        bytes_f32 = params * 4
        bytes_int8 = params

        # Multiply-accumulate operations
        macs = in_features * out_features

        layers.append({
            "name": f"linear{i + 1}",
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

    # ESP32-S3 memory budgets
    SRAM_BUDGET = 300 * 1024
    PSRAM_BUDGET = 8 * 1024 * 1024

    return {
        "layers": layers,

        "total_params": total_params,
        "total_bytes_f32": total_bytes_f32,
        "total_bytes_int8": total_bytes_int8,
        "total_macs": total_macs,

        "fits_300kb_sram_f32":
            total_bytes_f32 <= SRAM_BUDGET,

        "fits_300kb_sram_int8":
            total_bytes_int8 <= SRAM_BUDGET,

        "fits_8mb_psram_f32":
            total_bytes_f32 <= PSRAM_BUDGET,

        "fits_8mb_psram_int8":
            total_bytes_int8 <= PSRAM_BUDGET
    }


# --------------------------------------------------
# Model information endpoint
# --------------------------------------------------

@app.get("/model_info")
def model_info():

    # Inspect the actual trained network architecture
    layer_sizes = []

    for layer in net.net:
        if isinstance(layer, torch.nn.Linear):
            if not layer_sizes:
                layer_sizes.append(layer.in_features)

            layer_sizes.append(layer.out_features)

    return calculate_model_info(layer_sizes)


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "team": cfg["team"],
        "params": N_PARAMS,
        "val_mae": cfg["val_mae"]
    }


# --------------------------------------------------
# Prediction processing and timing
# --------------------------------------------------

async def run(data, schema):

    t_all = time.perf_counter()
    t = {}

    # Parse
    t0 = time.perf_counter()

    rows = (
        data.rows
        if isinstance(data, Batch)
        else [data]
    )

    t["parse"] = ms(t0)

    # Normalisation
    t0 = time.perf_counter()

    x = (
        np.array(
            [
                [getattr(r, f) for f in FEATURES]
                for r in rows
            ],
            np.float32
        ) - MU
    ) / SD

    t["normalise"] = ms(t0)

    # Inference
    t0 = time.perf_counter()

    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()

    t["inference"] = ms(t0)

    # Build response
    t0 = time.perf_counter()

    days = [
        round(float(v), 2)
        for v in np.maximum(y, 0)
    ]

    body = {
        "days_left": (
            days
            if isinstance(data, Batch)
            else days[0]
        )
    }

    t["build"] = ms(t0)

    # Total server time
    t["server_total"] = ms(t_all)

    body["timings_ms"] = t

    return body


# --------------------------------------------------
# Single prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(data: Reading):

    return await run(data, Reading)


# --------------------------------------------------
# Batch prediction endpoint
# --------------------------------------------------

@app.post("/predict_batch")
async def predict_batch(data: Batch):

    return await run(data, Batch)
