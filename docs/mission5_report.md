# EDGE AI Team 13 — Mission 5 Technical Report: Cleaned Dataset Modeling

## 1. Executive Summary
Mission 5 investigates the impact of dataset cleaning, glitch elimination, and architectural scaling on the doorbell remaining battery-life prediction task.
By replacing confirmed hardware glitch artifacts with a rigorously sanitized dataset (1,976 records), Team 13 trained and evaluated 10 neural network configurations (1-layer and 2-layer) alongside linear and mean baselines.

The selected candidate — **Cleaned NN_2L_32_32** (5 -> 32 -> 32 -> 1, 1,281 parameters) — achieves a **Validation MAE of 5.22 days** (vs 6.88 days for 16-16), reducing error by **24.2%** while requiring only **5.00 KiB** of FP32 parameter storage.

---

## 2. Data Cleaning Rules & Audit
Sensor anomalies were classified and filtered strictly according to physical domain limits:
1. `battery_v <= 0`: 11 glitch rows removed.
2. `temp_c <= 0`: 8 rows removed (including sentinel temperature -127°C).
3. `wifi_rssi_dbm == 0`: 5 rows removed.
4. `rings_per_day < 0`: 0 rows.
5. `age_months < 0`: 0 rows.
6. `days_left < 0`: 0 rows.

**Total records removed**: 24 out of 2,000 (1.2%).
**Cleaned records retained**: 1,976.

---

## 3. Training & Validation Setup
- **Evaluation Set**: Fixed common population of 395 valid validation records (20% holdout, random seed 42).
- **Training Set**: 1,581 cleaned records (80%).
- **Feature Normalization**: `StandardScaler` fitted strictly on the 1,581 training records.
  - Mean: `[3.7718, 27.7140, 7.8646, -66.5927, 18.3238]`
  - Scale: `[0.2446, 6.7956, 5.6356, 12.1021, 10.6236]`
- **Optimizer**: Adam, learning rate = 0.01, weight decay = 1e-4, batch size = 128, maximum epochs = 1500 with early stopping patience of 150 epochs.

---

## 4. Final Selected Candidate Performance
- **Model**: `Cleaned_NN_2L_32_32`
- **Architecture**: 5 -> 32 -> 32 -> 1
- **Trainable Parameters**: 1,281
- **Training MAE**: 4.33 days
- **Validation MAE**: 5.22 days
- **Validation RMSE**: 7.45 days
- **Validation $R^2$**: 0.9888
- **Best Epoch**: 1123
