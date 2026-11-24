# EDGE AI Team 13 — Complete API Specification & Contracts

## Base URL
- Local: `http://localhost:8000`
- Interactive Swagger UI: `http://localhost:8000/docs`

---

## 1. GET `/health` — Service Health & Model Status
Returns application status, assigned team number, registered team members, model parameter count, and validation MAE.

- **Method**: `GET`
- **Response**: `200 OK` (`application/json`)
- **Required Fields**:
  - `status`: string (`"ok"`)
  - `team`: integer (`13`)
  - `team_members`: list of strings (`["Suraj", "Sujal Chavan", "Nandish", "Monika"]`)
  - `params`: integer (`1281` for 32-32 candidate)
  - `val_mae`: float (`5.22`)
- **Example Response**:
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

---

## 2. GET `/model_info` — Layer Details & SRAM Budget Compliance
Provides layer-by-layer breakdown of linear layers, parameter counts, memory storage estimates (FP32 and INT8), MAC operations, and compliance with the ESP32-S3 SRAM budget (300 KiB / 307,200 bytes).

- **Method**: `GET`
- **Response**: `200 OK` (`application/json`)
- **Required Fields**:
  - `layers`: list of objects, each with:
    - `name`: string (`"linear1"`, `"linear2"`, `"linear3"`)
    - `in`: integer (input features)
    - `out`: integer (output features)
    - `params`: integer (weights + biases)
    - `bytes_f32`: integer (params * 4)
    - `macs`: integer (in * out)
  - `total_params`: integer (`1281`)
  - `total_bytes_f32`: integer (`5124`)
  - `total_bytes_int8`: integer (`1281`)
  - `total_macs`: integer (`1216`)
  - `fits_300kb_sram_f32`: boolean (`true`)
  - `fits_300kb_sram_int8`: boolean (`true`)
- **Example Response**:
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

---

## 3. POST `/predict` — Single Sensor Reading Prediction
Accepts five numerical sensor inputs, validates operational ranges, normalizes features, performs CPU neural network inference, and returns predicted remaining days with server timing breakdown.

- **Method**: `POST`
- **Content-Type**: `application/json`
- **Input Schema**:
| Field | Type | Validation Range | Unit | Description |
|---|---|---|---|---|
| `battery_v` | float | 0.0 &le; x &le; 5.0 | Volts | Battery voltage |
| `temp_c` | float | -40.0 &le; x &le; 85.0 | &deg;C | Temperature |
| `rings_per_day` | float | 0.0 &le; x &le; 1000.0 | count | Doorbell activations / day |
| `wifi_rssi_dbm` | float | -100.0 &le; x &le; 0.0 | dBm | Wi-Fi RSSI |
| `age_months` | float | 0.0 &le; x &le; 240.0 | months | Device operational age |

- **Example Request**:
```json
{
  "battery_v": 3.85,
  "temp_c": 24.5,
  "rings_per_day": 12,
  "wifi_rssi_dbm": -68,
  "age_months": 14
}
```

- **Example Success Response (200 OK)**:
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

- **Error Response (422 Unprocessable Entity)**:
Triggered if inputs are out of range or malformed.
```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "battery_v"],
      "msg": "Input should be less than or equal to 5",
      "input": 6.5,
      "ctx": {"le": 5.0}
    }
  ]
}
```

---

## 4. POST `/predict_batch` — Vectorized Batch Inference
Accepts multiple sensor readings in a single batch, validates all records, performs batch normalization, and returns predictions for each record preserving input order.

- **Method**: `POST`
- **Content-Type**: `application/json`
- **Input Schema**:
  - `rows`: list of `Reading` objects (minimum length: 1, maximum length: 10,000)
- **Example Request**:
```json
{
  "rows": [
    {
      "battery_v": 4.10,
      "temp_c": 20.0,
      "rings_per_day": 5,
      "wifi_rssi_dbm": -55,
      "age_months": 2
    },
    {
      "battery_v": 3.65,
      "temp_c": 30.0,
      "rings_per_day": 20,
      "wifi_rssi_dbm": -78,
      "age_months": 22
    }
  ]
}
```

- **Example Success Response (200 OK)**:
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
