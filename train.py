
"""
train.py -- Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 13 --hidden 16,16

Writes model/model.pt and model/config.json, which app/main.py loads.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, BatteryNet


# --------------------------------------------------
# 1. Command-line arguments
# --------------------------------------------------

ap = argparse.ArgumentParser()

ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="16,16")
ap.add_argument("--epochs", type=int, default=1500)

a = ap.parse_args()

hidden = tuple(int(h) for h in a.hidden.split(","))


# --------------------------------------------------
# 2. Load the team's dataset
# --------------------------------------------------

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")

print("\n--- DATA QUALITY CHECK ---")
print(f"Original dataset rows: {len(df)}")


# --------------------------------------------------
# 3. Identify suspicious sensor readings
# --------------------------------------------------

zero_battery = df["battery_v"] == 0
zero_rssi = df["wifi_rssi_dbm"] == 0
temp_sentinel = df["temp_c"] == -127

# Suspicious records are flagged but NOT removed.
suspicious = zero_battery | zero_rssi | temp_sentinel


# --------------------------------------------------
# 4. Identify confirmed invalid values
# --------------------------------------------------

negative_rings = df["rings_per_day"] < 0
negative_age = df["age_months"] < 0
negative_days = df["days_left"] < 0

# Remove only confirmed invalid records.
bad = negative_rings | negative_age | negative_days


# --------------------------------------------------
# 5. Print the data-quality report
# --------------------------------------------------

print(f"Zero battery voltage: {zero_battery.sum()}")
print(f"Zero Wi-Fi RSSI: {zero_rssi.sum()}")
print(f"Temperature -127: {temp_sentinel.sum()}")

print(f"Total suspicious records: {suspicious.sum()}")

print(f"Negative daily ring counts: {negative_rings.sum()}")
print(f"Negative device ages: {negative_age.sum()}")
print(f"Negative remaining battery life: {negative_days.sum()}")

print(f"Confirmed invalid records: {bad.sum()}")


# --------------------------------------------------
# 6. Remove only confirmed invalid records
# --------------------------------------------------

print(f"Glitch rows dropped: {bad.sum()}")

df = df.loc[~bad].copy()

print(f"Rows remaining after cleaning: {len(df)}")
print("--- END DATA QUALITY CHECK ---\n")


# --------------------------------------------------
# 7. Split dataset into training and validation sets
# --------------------------------------------------

rng = np.random.default_rng(0)

idx = rng.permutation(len(df))
n_val = len(df) // 5

val = df.iloc[idx[:n_val]]
tr = df.iloc[idx[n_val:]]

X = tr[FEATURES].to_numpy(np.float32)
y = tr["days_left"].to_numpy(np.float32)

Xv = val[FEATURES].to_numpy(np.float32)
yv = val["days_left"].to_numpy(np.float32)


# --------------------------------------------------
# 8. Feature normalisation
# --------------------------------------------------

# Calculate statistics using training rows ONLY.
# The same statistics are applied to validation data.

mu = X.mean(axis=0)
sd = X.std(axis=0)

# Avoid division by zero if any feature has zero variance.
sd = np.where(sd == 0, 1.0, sd)

print("\n--- FEATURE NORMALISATION ---")
print("Training feature means:", mu)
print("Training feature standard deviations:", sd)
print("Normalisation statistics calculated successfully.")


# --------------------------------------------------
# 9. Mean baseline
# --------------------------------------------------

def mae(predictions, targets):
    return float(np.mean(np.abs(predictions - targets)))


mean_prediction = np.full_like(yv, y.mean())

mean_mae = mae(mean_prediction, yv)

print("\n--- BASELINE RESULTS ---")
print(f"Mean baseline validation MAE: {mean_mae:.2f} days")


# --------------------------------------------------
# 10. Linear regression baseline
# --------------------------------------------------

# Add an intercept column to the training and validation data.
X_linear = np.column_stack([
    np.ones(len(X)),
    X
])

Xv_linear = np.column_stack([
    np.ones(len(Xv)),
    Xv
])

# Fit linear regression using training data only.
coefficients, residuals, rank, singular_values = np.linalg.lstsq(
    X_linear,
    y,
    rcond=None
)

# Predict validation targets.
linear_predictions = Xv_linear @ coefficients

linear_mae = mae(linear_predictions, yv)

print(f"Linear regression validation MAE: {linear_mae:.2f} days")


# --------------------------------------------------
# 11. Neural network training
# --------------------------------------------------

torch.manual_seed(0)

net = BatteryNet(hidden)

opt = torch.optim.Adam(
    net.parameters(),
    lr=3e-3
)

Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.float32)

Xvt = torch.tensor((Xv - mu) / sd, dtype=torch.float32)


for ep in range(a.epochs):

    perm = torch.randperm(len(Xt))

    for i in range(0, len(Xt), 128):

        b = perm[i:i + 128]

        predictions = net(Xt[b])

        loss = torch.nn.functional.mse_loss(
            predictions,
            yt[b]
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

    if ep % 300 == 0 or ep == a.epochs - 1:

        with torch.no_grad():

            val_predictions = net(Xvt).numpy()

            print(
                f"epoch {ep:4d}  "
                f"val MAE {mae(val_predictions, yv):.2f} days"
            )


# --------------------------------------------------
# 12. Final validation result
# --------------------------------------------------

with torch.no_grad():

    val_predictions = net(Xvt).numpy()

    val_mae = mae(val_predictions, yv)

print("\n--- FINAL MODEL RESULTS ---")
print(f"Architecture: {hidden}")
print(f"Parameters: {sum(p.numel() for p in net.parameters())}")
print(f"Validation MAE: {val_mae:.2f} days")


# --------------------------------------------------
# 13. Save trained model and configuration
# --------------------------------------------------

os.makedirs("model", exist_ok=True)

torch.save(
    net.state_dict(),
    "model/model.pt"
)

config = dict(
    team=a.team,
    hidden=list(hidden),
    mu=mu.tolist(),
    sd=sd.tolist(),
    features=FEATURES,
    val_mae=round(val_mae, 2)
)

with open("model/config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\nModel weights saved: model/model.pt")
print("Model configuration saved: model/config.json")
print("Training completed successfully.")
