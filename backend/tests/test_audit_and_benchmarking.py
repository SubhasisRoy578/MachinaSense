import os
import json
import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_DIR = os.path.join(BASE_DIR, "artifacts", "colab_fd001", "phase_9_5")
ORIG_EVAL_PATH = os.path.join(BASE_DIR, "artifacts", "colab_fd001", "evaluation.json")
ORIG_META_PATH = os.path.join(BASE_DIR, "artifacts", "colab_fd001", "model_metadata.json")

def test_audit_json_passes_all_checks():
    """Verify integrity_audit.json exists, is valid, and all checks passed."""
    path = os.path.join(AUDIT_DIR, "integrity_audit.json")
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["overall_audit_status"] == "PASS"
    for check_id, result in data["checks"].items():
        assert result["status"] == "PASS", f"Check {check_id} failed: {result['details']}"

def test_nasa_score_verification():
    """Verify independent NASA score matches Phase 9 reported scores within precision tolerance."""
    path = os.path.join(AUDIT_DIR, "nasa_score_verification.json")
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["verification_status"] == "PASS"
    assert data["capped_target_comparison"]["absolute_discrepancy"] < 1e-3
    assert data["uncapped_target_comparison"]["absolute_discrepancy"] < 1e-3

def test_model_benchmark_completeness():
    """Verify model_benchmark.json contains RF and LSTM comparisons on both target types."""
    path = os.path.join(AUDIT_DIR, "model_benchmark.json")
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data["benchmark_summary"]
    assert len(summary) == 6  # 3 models * 2 target types
    models = {s["model_name"] for s in summary}
    assert any("Random Forest (Colab FD001" in m for m in models)
    assert any("LSTM Neural Network" in m for m in models)
    for s in summary:
        assert s["MAE"] > 0
        assert s["RMSE"] > 0
        assert 0 < s["R2"] <= 1.0
        assert s["NASA_asymmetric_score"] > 0

def test_degradation_stage_comparison():
    """Verify degradation stage breakdown covers all 6 standard C-MAPSS intervals."""
    path = os.path.join(AUDIT_DIR, "degradation_stage_comparison.csv")
    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 6
    assert list(df["rul_range"]) == ["0-20", "21-40", "41-60", "61-80", "81-100", "101-125"]
    assert df["engine_count"].sum() == 100
    assert "stage_winner" in df.columns

def test_error_investigation():
    """Verify error investigation includes top 5 worst for RF, LSTM, and near-zero error analysis."""
    path = os.path.join(AUDIT_DIR, "error_investigation.csv")
    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) >= 15
    assert "Near-Zero Error Investigation" in df["category"].values
    assert "Worst Performer" in df["category"].values

def test_comparison_plots_exist():
    """Verify all 3 Phase 9.5 comparison plots exist and are non-empty."""
    for fn in ["actual_vs_predicted_comparison.png", "residual_comparison.png", "nasa_score_comparison.png"]:
        p = os.path.join(AUDIT_DIR, fn)
        assert os.path.exists(p)
        assert os.path.getsize(p) > 20000

def test_preservation_of_previous_artifacts():
    """Ensure original Phase 1-9 artifacts were not mutated."""
    assert os.path.exists(ORIG_EVAL_PATH)
    assert os.path.exists(ORIG_META_PATH)
    with open(ORIG_EVAL_PATH, "r") as f:
        ev = json.load(f)
    assert abs(ev["MAE"] - 10.647365) < 1e-4
    assert abs(ev["RMSE"] - 15.266790) < 1e-4

