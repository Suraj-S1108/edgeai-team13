"""
Unit tests for parameter_calculator.py
"""

import pytest
import torch
from app.model import BatteryNet
from edge_analysis.parameter_calculator import (
    count_linear_layer,
    calculate_mlp_architecture,
    calculate_two_hidden_mlp_formulas,
    compare_five_widths,
)


def test_count_linear_layer_valid():
    res = count_linear_layer(5, 32, bias=True)
    assert res["in_features"] == 5
    assert res["out_features"] == 32
    assert res["weights"] == 160
    assert res["biases"] == 32
    assert res["total_params"] == 192
    assert res["macs"] == 160

    # Without bias
    res_no_bias = count_linear_layer(5, 32, bias=False)
    assert res_no_bias["weights"] == 160
    assert res_no_bias["biases"] == 0
    assert res_no_bias["total_params"] == 160
    assert res_no_bias["macs"] == 160


def test_count_linear_layer_invalid_inputs():
    with pytest.raises(ValueError):
        count_linear_layer(0, 32)
    with pytest.raises(ValueError):
        count_linear_layer(5, -10)
    with pytest.raises(ValueError):
        count_linear_layer("5", 32)


def test_calculate_two_hidden_mlp_formulas_manual_verification():
    # 1. w = 32
    # W = 32^2 + 6*32 = 1024 + 192 = 1216
    # B = 2*32 + 1 = 65
    # P = 1216 + 65 = 1281
    # MACs = 1216
    res32 = calculate_two_hidden_mlp_formulas(32)
    assert res32["total_weights"] == 1216
    assert res32["total_biases"] == 65
    assert res32["total_params"] == 1281
    assert res32["total_macs"] == 1216

    # 2. w = 128
    # W = 128^2 + 6*128 = 16384 + 768 = 17152
    # B = 2*128 + 1 = 257
    # P = 17152 + 257 = 17409
    res128 = calculate_two_hidden_mlp_formulas(128)
    assert res128["total_weights"] == 17152
    assert res128["total_biases"] == 257
    assert res128["total_params"] == 17409
    assert res128["total_macs"] == 17152

    # 3. w = 256
    # W = 256^2 + 6*256 = 65536 + 1536 = 67072
    # B = 2*256 + 1 = 513
    # P = 67072 + 513 = 67585
    res256 = calculate_two_hidden_mlp_formulas(256)
    assert res256["total_weights"] == 67072
    assert res256["total_biases"] == 513
    assert res256["total_params"] == 67585
    assert res256["total_macs"] == 67072

    # 4. w = 512
    # W = 512^2 + 6*512 = 262144 + 3072 = 265216
    # B = 2*512 + 1 = 1025
    # P = 265216 + 1025 = 266241
    res512 = calculate_two_hidden_mlp_formulas(512)
    assert res512["total_weights"] == 265216
    assert res512["total_biases"] == 1025
    assert res512["total_params"] == 266241
    assert res512["total_macs"] == 265216

    # 5. w = 2048
    # W = 2048^2 + 6*2048 = 4194304 + 12288 = 4206592
    # B = 2*2048 + 1 = 4097
    # P = 4206592 + 4097 = 4210689
    res2048 = calculate_two_hidden_mlp_formulas(2048)
    assert res2048["total_weights"] == 4206592
    assert res2048["total_biases"] == 4097
    assert res2048["total_params"] == 4210689
    assert res2048["total_macs"] == 4206592


def test_verify_against_pytorch_battery_net():
    # Verify calculated parameter counts against PyTorch nn.Module for all benchmark widths
    for w in [32, 128, 256, 512]:
        calc = calculate_two_hidden_mlp_formulas(w)
        net = BatteryNet((w, w))
        pytorch_params = sum(p.numel() for p in net.parameters())
        assert calc["total_params"] == pytorch_params, f"Mismatch at width {w}: {calc['total_params']} vs {pytorch_params}"


def test_calculate_mlp_architecture_general():
    arch = calculate_mlp_architecture(input_dim=5, hidden_widths=(16, 32, 64), output_dim=1)
    assert len(arch["layers"]) == 4
    assert arch["layers"][0]["in_features"] == 5
    assert arch["layers"][0]["out_features"] == 16
    assert arch["layers"][1]["in_features"] == 16
    assert arch["layers"][1]["out_features"] == 32
    assert arch["layers"][2]["in_features"] == 32
    assert arch["layers"][2]["out_features"] == 64
    assert arch["layers"][3]["in_features"] == 64
    assert arch["layers"][3]["out_features"] == 1


def test_compare_five_widths_dataframe():
    df = compare_five_widths()
    assert len(df) == 5
    assert list(df["Width (w)"]) == [32, 128, 256, 512, 2048]
    assert "Total Trainable Params (w^2+8w+1)" in df.columns
    assert "Total MACs (w^2+6w)" in df.columns
    assert df.loc[df["Width (w)"] == 32, "Total Trainable Params (w^2+8w+1)"].values[0] == 1281

