"""
Integration and end-to-end tests for Mission 4 pipeline.
"""

import os
from pathlib import Path
import pandas as pd
import pytest

from edge_analysis.run_analysis import run_full_mission4_pipeline


def test_full_mission4_pipeline_execution(tmp_path):
    out_dir = str(tmp_path / "edge_analysis_output")
    run_full_mission4_pipeline(output_dir=out_dir)

    # 1. Verify all 6 CSV files exist and are populated
    csv_files = [
        "width_comparison.csv",
        "memory_comparison.csv",
        "activation_memory.csv",
        "sram_feasibility.csv",
        "inference_timing.csv",
        "server_vs_edge.csv",
    ]
    for csv_name in csv_files:
        p = os.path.join(out_dir, csv_name)
        assert os.path.exists(p), f"Missing CSV: {p}"
        df = pd.read_csv(p)
        assert len(df) > 0, f"CSV {csv_name} is empty"

    # 2. Verify all 5 PNG graphics exist and are valid sizes (> 10 KB)
    png_files = [
        "params_vs_width.png",
        "f32_vs_int8_memory.png",
        "memory_vs_sram_limit.png",
        "inference_latency_vs_width.png",
        "activation_memory.png",
    ]
    for png_name in png_files:
        p = os.path.join(out_dir, png_name)
        assert os.path.exists(p), f"Missing PNG: {p}"
        size_bytes = os.path.getsize(p)
        assert size_bytes > 10_000, f"PNG {png_name} too small: {size_bytes} bytes"

    # 3. Verify Markdown reports exist and are comprehensive
    report_path = os.path.join(out_dir, "MISSION_4_TECHNICAL_REPORT.md")
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        rep_content = f.read()
    assert len(rep_content) > 5000
    assert "ESP32-S3" in rep_content
    assert "300 KB" in rep_content
    assert "INT8 Quantization" in rep_content

    summary_path = os.path.join(out_dir, "summary_of_findings.md")
    assert os.path.exists(summary_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        sum_content = f.read()
    assert len(sum_content) > 500
    assert "Key Numerical Takeaways" in sum_content

