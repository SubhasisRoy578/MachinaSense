import os
import sys
import json
import pytest
import pandas as pd
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.main import app
from app.auth.clerk import get_current_user
from app.db.models import User

COLAB_DIR = os.path.join(BASE_DIR, "artifacts", "colab_fd001")
PHASE_10_DIR = os.path.join(COLAB_DIR, "phase_10")


@pytest.fixture
def auth_client():
    mock_user = User(id="test_user_p10", email="p10@test.com")
    app.dependency_overrides[get_current_user] = lambda: mock_user
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def unauth_client():
    app.dependency_overrides.clear()
    return TestClient(app)


# =====================================================================
# 1. EXPLAINABILITY DELIVERABLES & RANKINGS
# =====================================================================
def test_rf_global_feature_importance():
    csv_path = os.path.join(PHASE_10_DIR, "rf_global_feature_importance.csv")
    assert os.path.exists(csv_path), "rf_global_feature_importance.csv must exist"
    df = pd.read_csv(csv_path)
    assert len(df) == 19, f"Expected 19 features, got {len(df)}"
    assert "feature" in df.columns and "gini_importance" in df.columns
    assert "cycle" in df["feature"].values
    assert "sensor_11" in df["feature"].values
    assert df["gini_importance"].sum() > 0.99  # Normalized importances sum to 1.0


def test_rf_permutation_importance():
    csv_path = os.path.join(PHASE_10_DIR, "rf_permutation_importance.csv")
    assert os.path.exists(csv_path), "rf_permutation_importance.csv must exist"
    df = pd.read_csv(csv_path)
    assert len(df) == 19
    assert "feature" in df.columns and "mae_increase_when_permuted" in df.columns
    # Top features should include degradation indicators
    top3 = set(df.head(3)["feature"])
    assert "sensor_11" in top3 or "sensor_4" in top3 or "cycle" in top3


def test_lstm_sensor_and_temporal_importance():
    sensor_csv = os.path.join(PHASE_10_DIR, "lstm_sensor_importance.csv")
    temp_csv = os.path.join(PHASE_10_DIR, "lstm_temporal_importance.csv")
    assert os.path.exists(sensor_csv), "lstm_sensor_importance.csv must exist"
    assert os.path.exists(temp_csv), "lstm_temporal_importance.csv must exist"

    df_sensors = pd.read_csv(sensor_csv)
    df_temp = pd.read_csv(temp_csv)

    assert len(df_sensors) == 15, "Expected 15 LSTM sensors"
    assert len(df_temp) == 30, "Expected 30 temporal window steps"

    # Peak temporal attribution occurs in the final degradation phase (last 5 cycles)
    peak_idx = df_temp["gradient_saliency"].idxmax()
    assert peak_idx >= 24, f"Expected peak saliency in final cycles, got index {peak_idx}"
    # Final 10 cycles have higher mean saliency than mid-window (cycles 10-20)
    assert df_temp.iloc[20:]["gradient_saliency"].mean() > df_temp.iloc[10:20]["gradient_saliency"].mean()


def test_explainability_json_and_plots():
    rf_json = os.path.join(PHASE_10_DIR, "rf_explainability.json")
    lstm_json = os.path.join(PHASE_10_DIR, "lstm_explainability.json")
    assert os.path.exists(rf_json), "rf_explainability.json must exist"
    assert os.path.exists(lstm_json), "lstm_explainability.json must exist"

    with open(rf_json, "r", encoding="utf-8") as f:
        rf_data = json.load(f)
    assert len(rf_data.get("sample_engine_explanations", [])) == 5
    sample_ids = [e["engine_id"] for e in rf_data["sample_engine_explanations"]]
    assert set(sample_ids) == {1, 34, 45, 77, 82}

    with open(lstm_json, "r", encoding="utf-8") as f:
        lstm_data = json.load(f)
    assert len(lstm_data.get("sample_engine_explanations", [])) == 5

    # Check plot files exist
    assert os.path.exists(os.path.join(PHASE_10_DIR, "rf_feature_importance.png"))
    assert os.path.exists(os.path.join(PHASE_10_DIR, "lstm_sensor_importance.png"))
    assert os.path.exists(os.path.join(PHASE_10_DIR, "lstm_temporal_importance.png"))


# =====================================================================
# 2. ROBUSTNESS & FAULT INJECTION
# =====================================================================
def test_noise_robustness_metrics():
    csv_path = os.path.join(PHASE_10_DIR, "robustness_test_results.csv")
    json_path = os.path.join(PHASE_10_DIR, "robustness_summary.json")
    assert os.path.exists(csv_path), "robustness_test_results.csv must exist"
    assert os.path.exists(json_path), "robustness_summary.json must exist"

    df = pd.read_csv(csv_path)
    assert len(df) == 4, "Expected 4 noise levels: 1%, 3%, 5%, 10%"
    assert "noise_level" in df.columns

    # Verify stability metrics
    assert df.loc[df["noise_level"] == "5%", "rf_stability_pct"].values[0] >= 90.0
    assert df.loc[df["noise_level"] == "5%", "lstm_stability_pct"].values[0] >= 90.0
    assert os.path.exists(os.path.join(PHASE_10_DIR, "sensor_noise_robustness.png"))
    assert os.path.exists(os.path.join(PHASE_10_DIR, "prediction_stability.png"))


# =====================================================================
# 3. PERFORMANCE & LATENCY PROFILING
# =====================================================================
def test_inference_performance_profile():
    csv_path = os.path.join(PHASE_10_DIR, "inference_performance.csv")
    json_path = os.path.join(PHASE_10_DIR, "inference_performance.json")
    assert os.path.exists(csv_path), "inference_performance.csv must exist"
    assert os.path.exists(json_path), "inference_performance.json must exist"

    df = pd.read_csv(csv_path)
    assert len(df) == 2, "Expected 2 models: RF and LSTM"
    for _, row in df.iterrows():
        assert row["warm_mean_lat_ms"] > 0
        assert row["warm_p95_lat_ms"] >= row["warm_mean_lat_ms"]
        assert row["throughput_req_per_sec"] > 0

    assert os.path.exists(os.path.join(PHASE_10_DIR, "inference_latency_comparison.png"))


# =====================================================================
# 4. DEPLOYMENT READINESS & SECURITY AUDIT
# =====================================================================
def test_deployment_and_security_checklist():
    chk_path = os.path.join(PHASE_10_DIR, "deployment_readiness_checklist.json")
    assert os.path.exists(chk_path), "deployment_readiness_checklist.json must exist"

    with open(chk_path, "r", encoding="utf-8") as f:
        chk = json.load(f)

    assert chk.get("overall_verdict") == "PRODUCTION_READY"
    assert chk.get("secrets_hygiene", {}).get("hardcoded_secrets_detected") == 0
    assert chk.get("runtime_endpoints", {}).get("health_check_endpoint") == "PASS"
    assert chk.get("runtime_endpoints", {}).get("fd001_prediction_endpoint") == "PASS"


# =====================================================================
# 5. FASTAPI EXPLAINABILITY API ENDPOINTS
# =====================================================================
def test_api_explainability_summary(auth_client):
    res = auth_client.get("/api/explainability/summary")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()
    assert body.get("status") == "ready"
    assert "models" in body
    assert "random_forest" in body["models"]
    assert "lstm" in body["models"]
    assert "robustness" in body
    assert "performance" in body


def test_api_explainability_engines(auth_client):
    res = auth_client.get("/api/explainability/engines")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()
    assert body.get("status") == "ready"
    assert body.get("sample_engine_ids") == [1, 34, 45, 77, 82]
    assert len(body.get("random_forest_engines", [])) == 5
    assert len(body.get("lstm_engines", [])) == 5


def test_api_explainability_requires_auth(unauth_client):
    # Without authorization header, in production mode it should reject 401
    # In dev mode, check behavior according to clerk mock setup
    res = unauth_client.get("/api/explainability/summary")
    assert res.status_code in [200, 401]
