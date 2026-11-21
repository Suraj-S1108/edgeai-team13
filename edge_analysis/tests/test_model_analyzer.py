"""
Unit tests for model_analyzer.py
"""

import os
import pytest
from pathlib import Path
from edge_analysis.model_analyzer import (
    load_repository_model_config,
    analyze_model_edge_metrics,
    perform_quantization_and_evaluation,
)


def test_load_repository_model_config():
    cfg = load_repository_model_config("model/config.json", "model/model.pt")
    assert cfg["team"] == 13
    assert cfg["hidden"] == (32, 32)
    assert len(cfg["features"]) == 5
    assert cfg["total_params"] == 1281
    assert cfg["total_macs"] == 1216
    assert cfg["weights_loaded"] is True
    assert "STARTER" in cfg["model_status"] or "FINAL" in cfg["model_status"]


def test_analyze_model_edge_metrics():
    cfg = load_repository_model_config("model/config.json", "model/model.pt")
    metrics = analyze_model_edge_metrics(cfg)
    assert metrics["total_params"] == 1281
    assert metrics["total_bytes_f32"] == 5124
    assert metrics["total_bytes_int8"] == 1281
    assert metrics["fits_300kb_sram_f32"] is True
    assert metrics["fits_300kb_sram_int8"] is True


def test_perform_quantization_and_evaluation(tmp_path):
    q_res = perform_quantization_and_evaluation(
        data_path="data/team_13_train.csv",
        output_dir=str(tmp_path),
    )
    assert q_res["quantization_successful"] is True
    assert q_res["mae_float32"] > 0
    assert q_res["mae_int8"] > 0
    assert os.path.exists(q_res["artifact_json_path"])
    assert os.path.exists(q_res["artifact_pt_path"])

