"""
train.py -- Clean the data, fit baselines, train the network, save it for the server.

Usage:
    python train.py --team 13 --hidden 32,32 --dataset cleaned
    python train.py --team 13 --hidden 16,16 --dataset original

Writes model/model.pt and model/config.json, which app/main.py loads.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from app.model import FEATURES, BatteryNet


def parse_args():
    ap = argparse.ArgumentParser(description="Train EDGE AI battery-life prediction network")
    ap.add_argument("--team", type=int, required=True, help="Team number (e.g. 13)")
    ap.add_argument("--hidden", default="32,32", help="Hidden layer sizes separated by comma (default: 32,32)")
    ap.add_argument("--epochs", type=int, default=1500, help="Number of training epochs (default: 1500)")
    ap.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--dataset", choices=["cleaned", "original"], default="cleaned",
                    help="Dataset to train on (default: cleaned)")
    ap.add_argument("--save-dir", default="model", help="Directory to save model.pt and config.json")
    return ap.parse_args()


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply Mission 5 cleaning rules:
    - battery_v > 0
    - temp_c > 0 (removes -127 sentinel)
    - wifi_rssi_dbm != 0
    - rings_per_day >= 0
    - age_months >= 0
    - days_left >= 0
    """
    clean_mask = (
        (df["battery_v"] > 0) &
        (df["temp_c"] > 0) &
        (df["wifi_rssi_dbm"] != 0) &
        (df["rings_per_day"] >= 0) &
        (df["age_months"] >= 0) &
        (df["days_left"] >= 0)
    )
    return df[clean_mask].copy(), clean_mask


def main():
    args = parse_args()
    hidden = tuple(int(h) for h in args.hidden.split(","))

    print(f"\n=======================================================")
    print(f"  EDGE AI Team {args.team:02d} - Model Training Pipeline")
    print(f"  Architecture: {hidden} | Dataset: {args.dataset}")
    print(f"=======================================================\n")

    # 1. Load raw dataset
    raw_path = f"data/team_{args.team:02d}_train.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset not found: {raw_path}")

    df_raw = pd.read_csv(raw_path)
    print(f"Original dataset rows: {len(df_raw)}")

    # 2. Data Cleaning & Splitting
    df_clean, clean_mask = clean_dataset(df_raw)
    dropped_count = len(df_raw) - len(df_clean)
    print(f"Cleaned dataset rows:  {len(df_clean)} (Dropped: {dropped_count} rows)")

    # Reproducible index-based split (seed 42, 20% validation)
    tr_idx, val_idx = train_test_split(df_raw.index, test_size=0.2, random_state=args.seed)

    if args.dataset == "cleaned":
        tr_clean_idx = [i for i in tr_idx if clean_mask.iloc[i]]
        val_clean_idx = sorted([i for i in val_idx if clean_mask.iloc[i]])
        df_train = df_raw.loc[tr_clean_idx].copy()
        df_val = df_raw.loc[val_clean_idx].copy()
    else:
        df_train = df_raw.loc[tr_idx].copy()
        df_val = df_raw.loc[sorted(val_idx)].copy()

    print(f"Training samples:   {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")

    X_train = df_train[FEATURES].values
    y_train = df_train["days_left"].values
    X_val = df_val[FEATURES].values
    y_val = df_val["days_left"].values

    # 3. Feature Normalisation (Fitted strictly on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    mu = scaler.mean_
    sd = scaler.scale_

    print("\n--- FEATURE NORMALISATION (Train Set Only) ---")
    for feat, m, s in zip(FEATURES, mu, sd):
        print(f"  {feat:15s} mean = {m:10.4f}, std = {s:10.4f}")

    # 4. Baselines
    def calc_metrics(y_true, y_pred):
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        return mae, rmse, r2

    mean_pred = np.full_like(y_val, y_train.mean())
    m_mae, m_rmse, m_r2 = calc_metrics(y_val, mean_pred)
    print(f"\n--- BASELINE EVALUATION ---")
    print(f"Mean Baseline:       MAE = {m_mae:6.2f} days | RMSE = {m_rmse:6.2f} days | R2 = {m_r2:6.4f}")

    X_tr_lin = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_val_lin = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])
    beta, _, _, _ = np.linalg.lstsq(X_tr_lin, y_train, rcond=None)
    lin_pred = X_val_lin @ beta
    l_mae, l_rmse, l_r2 = calc_metrics(y_val, lin_pred)
    print(f"Linear Regression:   MAE = {l_mae:6.2f} days | RMSE = {l_rmse:6.2f} days | R2 = {l_r2:6.4f}")

    # 5. Neural Network Training
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    net = BatteryNet(hidden)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"\n--- NEURAL NETWORK TRAINING ---")
    print(f"Model: {hidden} | Trainable parameters: {total_params}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_train_scaled, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xvt = torch.tensor(X_val_scaled, dtype=torch.float32)

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None

    batch_size = 128
    n_samples = len(Xt)

    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            preds = net(Xt[batch_idx])
            loss = loss_fn(preds, yt[batch_idx])
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0 or epoch == args.epochs:
            net.eval()
            with torch.no_grad():
                val_preds = net(Xvt).numpy()
                cur_val_mae = float(np.mean(np.abs(val_preds - y_val)))

                if cur_val_mae < best_val_mae:
                    best_val_mae = cur_val_mae
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

            if epoch % 300 == 0 or epoch == args.epochs:
                print(f"Epoch {epoch:4d}/{args.epochs} | Current Val MAE: {cur_val_mae:5.2f} days | Best: {best_val_mae:5.2f} days (epoch {best_epoch})")

    # 6. Evaluation with best weights
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    with torch.no_grad():
        final_tr_preds = net(Xt).numpy()
        final_val_preds = net(Xvt).numpy()

    tr_mae, tr_rmse, tr_r2 = calc_metrics(y_train, final_tr_preds)
    val_mae, val_rmse, val_r2 = calc_metrics(y_val, final_val_preds)

    print("\n--- FINAL MODEL EVALUATION ---")
    print(f"Architecture:       5 -> {' -> '.join(map(str, hidden))} -> 1")
    print(f"Parameters:         {total_params}")
    print(f"Best Epoch:         {best_epoch}")
    print(f"Training Metrics:   MAE = {tr_mae:6.2f} days | RMSE = {tr_rmse:6.2f} days | R2 = {tr_r2:6.4f}")
    print(f"Validation Metrics: MAE = {val_mae:6.2f} days | RMSE = {val_rmse:6.2f} days | R2 = {val_r2:6.4f}")

    # 7. Save model and config
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "model.pt"
    config_path = save_dir / "config.json"

    torch.save(net.state_dict(), model_path)

    config = {
        "team": args.team,
        "hidden": list(hidden),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "features": FEATURES,
        "val_mae": round(val_mae, 2),
        "val_rmse": round(val_rmse, 2),
        "val_r2": round(val_r2, 4),
        "train_mae": round(tr_mae, 2),
        "parameters": total_params,
        "best_epoch": best_epoch,
        "dataset": args.dataset
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nModel weights saved:       {model_path}")
    print(f"Model configuration saved: {config_path}")
    print("Training pipeline finished successfully.")


if __name__ == "__main__":
    main()
