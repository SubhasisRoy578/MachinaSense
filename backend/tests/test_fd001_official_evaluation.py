import os
import json
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "artifacts", "colab_fd001", "evaluation")
ORIGINAL_METADATA_PATH = os.path.join(BASE_DIR, "artifacts", "colab_fd001", "model_metadata.json")
ORIGINAL_EVAL_PATH = os.path.join(BASE_DIR, "artifacts", "colab_fd001", "evaluation.json")

def test_original_evaluation_artifacts_untouched():
    """Verify that original model metadata and validation evaluation files are preserved."""
    assert os.path.exists(ORIGINAL_METADATA_PATH)
    assert os.path.exists(ORIGINAL_EVAL_PATH)
    
    with open(ORIGINAL_EVAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert abs(data["MAE"] - 10.647365) < 1e-4
    assert abs(data["RMSE"] - 15.266790) < 1e-4
    assert abs(data["R2"] - 0.866064) < 1e-4
    assert data["validation_engines"] == 20
    assert data["training_engines"] == 80

def test_official_test_metrics_json():
    """Verify that official test metrics JSON is complete, valid, and meets scientific criteria."""
    metrics_path = os.path.join(EVAL_DIR, "official_test_metrics.json")
    assert os.path.exists(metrics_path)
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
        
    assert m["test_engine_count"] == 100
    assert m["feature_count"] == 19
    assert m["rul_cap"] == 125
    
    capped = m["metrics_capped_target"]
    assert 10.0 < capped["MAE"] < 15.0
    assert 14.0 < capped["RMSE"] < 20.0
    assert 0.75 < capped["R2"] < 0.90
    assert capped["NASA_asymmetric_score"] > 0
    assert len(m["top_5_best_engines"]) == 5
    assert len(m["top_5_worst_engines"]) == 5
    assert len(m["rul_range_analysis"]) == 6

def test_official_test_predictions_csv():
    """Verify official test predictions CSV integrity."""
    csv_path = os.path.join(EVAL_DIR, "official_test_predictions.csv")
    assert os.path.exists(csv_path)
    
    df = pd.read_csv(csv_path)
    assert len(df) == 100
    assert df["engine_id"].nunique() == 100
    assert set(df.columns) == {"engine_id", "actual_rul", "predicted_rul", "residual", "absolute_error"}
    assert df.isnull().sum().sum() == 0
    assert (df["predicted_rul"] >= 0).all()
    assert (df["predicted_rul"] <= 125).all()
    assert (df["actual_rul"] >= 0).all()
    assert (df["actual_rul"] <= 125).all()

def test_official_test_error_analysis_csv():
    """Verify official test error analysis CSV integrity."""
    csv_path = os.path.join(EVAL_DIR, "official_test_error_analysis.csv")
    assert os.path.exists(csv_path)
    
    df = pd.read_csv(csv_path)
    assert len(df) == 6
    assert set(df["rul_range"]) == {"0-20", "21-40", "41-60", "61-80", "81-100", "101-125"}
    assert df["engine_count"].sum() == 100

def test_evaluation_plots_exist():
    """Verify all 3 evaluation plots were generated and are non-empty."""
    for filename in ["actual_vs_predicted.png", "residual_distribution.png", "absolute_error_distribution.png"]:
        p = os.path.join(EVAL_DIR, filename)
        assert os.path.exists(p)
        assert os.path.getsize(p) > 10000  # at least 10KB

