"""model.py -- EDGE AI Team 13 BatteryNet architecture."""
import torch
from torch import nn

FEATURES = ["battery_v", "temp_c", "rings_per_day", "wifi_rssi_dbm", "age_months"]


class BatteryNet(nn.Module):
    def __init__(self, hidden=(32, 32)):
        super().__init__()
        layers, n_in = [], len(FEATURES)
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            n_in = h
        layers.append(nn.Linear(n_in, 1))
        self.network = nn.Sequential(*layers)

    @property
    def net(self):
        """Sequential container alias."""
        return self.network

    def forward(self, x):
        return self.network(x).squeeze(-1)
