# EDGE AI Team 13 — Model Architecture & Experimental Comparison Report

## 1. Overview
This report documents the empirical evaluation and architectural selection process for Team 13's doorbell battery-life predictor. We compared the legacy Phase 12 baseline (16-16) against candidate architectures trained on both the original and glitch-cleaned datasets.

---

## 2. Dataset Hygiene: Glitch Removal Impact
In Mission 5, a strict data hygiene protocol was established:
- **Zero Battery Voltage**: 11 records (0.55% of raw data)
- **Non-positive Temperature**: 8 records (0.40% of raw data, including -127°C sensor sentinel codes)
- **Zero Wi-Fi RSSI**: 5 records (0.25% of raw data)
- **Negative Readings**: 0 records
- **Total dropped records**: 24 out of 2,000 records (1.20%)
- **Cleaned Dataset**: 1,976 records

### Split Methodology:
To eliminate data leakage, an index-based split (80% train / 20% validation, seed 42) was evaluated on a fixed population of **395 common valid validation records**.

---

## 3. Empirical Results Across Candidate Architectures

| Dataset | Model Architecture | Hidden Widths | Total Params | Train MAE (days) | Val MAE (days) | Val RMSE (days) | Val $R^2$ | Overfitting Gap (days) |
|---|---|---|---|---|---|---|---|---|
| **Cleaned** | **NN_2L_32_32 (Selected)** | **[32, 32]** | **1,281** | **4.33** | **5.22** | **7.45** | **0.9888** | **0.89** |
| Cleaned | NN_2L_64_64 | [64, 64] | 4,609 | 3.60 | 5.13 | 7.65 | 0.9882 | 1.53 |
| Cleaned | NN_2L_16_16 | [16, 16] | 385 | 6.10 | 6.88 | 9.72 | 0.9809 | 0.79 |
| Cleaned | NN_2L_8_8 | [8, 8] | 129 | 6.34 | 7.34 | 10.88 | 0.9761 | 1.00 |
| Cleaned | NN_1L_64 | [64] | 449 | 11.73 | 12.22 | 17.69 | 0.9369 | 0.50 |
| Cleaned | NN_1L_16 | [16] | 113 | 13.99 | 13.80 | 19.08 | 0.9265 | -0.19 |
| Cleaned | NN_1L_32 | [32] | 225 | 16.50 | 17.32 | 23.86 | 0.8850 | 0.82 |
| Cleaned | NN_2L_4_4 | [4, 4] | 49 | 16.75 | 17.50 | 24.16 | 0.8822 | 0.74 |
| Cleaned | Linear Regression | None | 6 | 24.18 | 26.05 | 34.97 | 0.7531 | 1.87 |
| Cleaned | Mean Baseline | None | 0 | 52.52 | 55.62 | 70.78 | -0.0114 | 3.11 |
| Original | NN_2L_64_64 | [64, 64] | 4,609 | 4.01 | 5.08 | 7.79 | 0.9878 | 1.07 |
| Original | NN_2L_32_32 | [32, 32] | 1,281 | 5.07 | 6.00 | 8.54 | 0.9853 | 0.93 |
| Original | NN_2L_16_16 (Legacy) | [16, 16] | 385 | 6.95 | 7.14 | 9.71 | 0.9810 | 0.19 |

---

## 4. Pareto Optimization & Architectural Selection Rationale

### Why 32-32 Over 16-16:
1. **Significant Error Reduction**: Validation MAE decreases from **6.88 days** (cleaned 16-16) to **5.22 days** (cleaned 32-32), representing a **24.2% reduction in prediction error** (1.66 days more accurate).
2. **Superior Generalization**: The validation $R^2$ increases from 0.9809 to 0.9888, explaining 98.88% of battery discharge variance.
3. **Head-to-head Win Rate**: Across all 395 common validation samples, the 32-32 model achieves lower absolute error on 248 samples (62.8% win rate).

### Why 32-32 Over 64-64 (Diminishing Returns):
1. **Parameter Explosion**: The 64-64 architecture requires 4,609 parameters (18.44 KiB FP32), a **3.6x increase** over 32-32 (1,281 parameters).
2. **Negligible Error Gain**: The validation MAE improves by only **0.09 days** (5.13 vs 5.22 days).
3. **Overfitting Gap**: 64-64 exhibits an overfitting gap of **1.53 days** (Train MAE 3.60 vs Val MAE 5.13), whereas 32-32 maintains a healthy gap of only **0.89 days**.
4. **Conclusion**: The 5->32->32->1 network represents the true **Pareto optimal frontier** for edge deployment on constrained microcontroller hardware.
