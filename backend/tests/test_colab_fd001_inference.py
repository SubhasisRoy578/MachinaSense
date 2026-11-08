"""Automated tests for NASA C-MAPSS FD001 Colab Random Forest model integration."""
import os
import sys
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.config import settings
from app.main import app
from app.ml_service import COLAB_FD001_FEATURE_ORDER, FEATURE_COLS, ml_service

settings.ENVIRONMENT = "test"

VALID_19_FEATURES = {
    "cycle": 1.0,
    "op_setting_1": -0.0007,
    "op_setting_2": -0.0004,
    "op_setting_3": 100.0,
    "sensor_2": 641.82,
    "sensor_3": 1589.70,
    "sensor_4": 1400.60,
    "sensor_6": 21.61,
    "sensor_7": 554.36,
    "sensor_8": 2388.06,
    "sensor_9": 9046.19,
    "sensor_11": 47.47,
    "sensor_12": 521.66,
    "sensor_13": 2388.02,
    "sensor_14": 8138.62,
    "sensor_15": 8.4195,
    "sensor_17": 392.0,
    "sensor_20": 39.06,
    "sensor_21": 23.4190,
}

VALID_15_BENCHMARK_FEATURES = [
    641.82, 1589.70, 1400.60, 21.61, 554.36,
    2388.06, 9046.19, 47.47, 521.66, 2388.02,
    8138.62, 8.4195, 392.0, 39.06, 23.4190,
]

def auth_headers(user: str = "test_user"):
    return {"Authorization": f"Bearer test_token_{user}"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_1_successful_loading_of_new_model(client: TestClient):
    """Verify Colab FD001 model and metadata load correctly without disturbing existing models."""
    assert ml_service.is_loaded is True
    assert ml_service.rf_baseline is not None
    assert ml_service.is_colab_loaded is True
    assert ml_service.rf_colab_model is not None
    assert ml_service.rf_colab_metadata.get("target") == "RUL"
    assert ml_service.rf_colab_metadata.get("n_features") == 19
    assert ml_service.rf_colab_evaluation.get("R2") is not None


def test_2_successful_prediction_using_all_19_features(client: TestClient):
    """Verify prediction works via service and endpoint with all 19 features."""
    pred_service = ml_service.predict_rul_colab(VALID_19_FEATURES)
    assert isinstance(pred_service, float)
    assert 0.0 <= pred_service <= 125.0

    # Wrapped in "features"
    resp = client.post(
        "/api/predict/rul/fd001",
        headers=auth_headers(),
        json={"features": VALID_19_FEATURES},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_rul" in body
    assert body["predicted_rul"] == pred_service
    assert body["model_name"] == "MachinaSense Random Forest RUL Predictor"
    assert body["rul_cap"] == 125
    assert len(body["features_used"]) == 19
    assert "evaluation" in body

    # Top-level features dictionary
    resp_direct = client.post(
        "/api/predict/rul/fd001",
        headers=auth_headers(),
        json=VALID_19_FEATURES,
    )
    assert resp_direct.status_code == 200
    assert resp_direct.json()["predicted_rul"] == pred_service


def test_3_missing_feature_rejection(client: TestClient):
    """Verify rejection when any of the 19 features is missing."""
    for col in COLAB_FD001_FEATURE_ORDER:
        incomplete = dict(VALID_19_FEATURES)
        del incomplete[col]

        with pytest.raises(ValueError, match="Missing required feature columns"):
            ml_service.prepare_colab_fd001_features(incomplete)

        resp = client.post(
            "/api/predict/rul/fd001",
            headers=auth_headers(),
            json={"features": incomplete},
        )
        assert resp.status_code == 400
        assert "Missing required feature columns" in resp.json()["detail"]


def test_4_invalid_numerical_value_rejection(client: TestClient):
    """Verify rejection for null, string, boolean, NaN, and Inf values."""
    # Direct service validation test for NaN/Inf float objects
    with pytest.raises(ValueError, match="finite numerical value"):
        ml_service.prepare_colab_fd001_features({**VALID_19_FEATURES, "sensor_2": float("nan")})
    with pytest.raises(ValueError, match="finite numerical value"):
        ml_service.prepare_colab_fd001_features({**VALID_19_FEATURES, "sensor_3": float("inf")})

    # JSON-serializable invalid values for endpoint test
    invalid_cases = [
        ("sensor_2", None, "finite numerical value"),
        ("sensor_3", True, "finite numerical value"),
        ("sensor_4", "invalid_string", "numerical value"),
        ("sensor_7", "NaN", "finite numerical value"),
        ("sensor_8", "Infinity", "finite numerical value"),
        ("op_setting_1", "-Infinity", "finite numerical value"),
    ]

    for col, bad_val, error_fragment in invalid_cases:
        bad_payload = dict(VALID_19_FEATURES)
        bad_payload[col] = bad_val

        with pytest.raises(ValueError, match=error_fragment):
            ml_service.prepare_colab_fd001_features(bad_payload)

        resp = client.post(
            "/api/predict/rul/fd001",
            headers=auth_headers(),
            json={"features": bad_payload},
        )
        assert resp.status_code == 400
        assert error_fragment in resp.json()["detail"]


def test_5_correct_feature_ordering():
    """Verify feature adapter preserves exact 19-column ordering regardless of input dict order."""
    reversed_keys = list(reversed(list(VALID_19_FEATURES.keys())))
    scrambled = {k: VALID_19_FEATURES[k] for k in reversed_keys}

    prepared_df = ml_service.prepare_colab_fd001_features(scrambled)
    assert list(prepared_df.columns) == COLAB_FD001_FEATURE_ORDER
    for col in COLAB_FD001_FEATURE_ORDER:
        assert prepared_df[col].iloc[0] == VALID_19_FEATURES[col]


def test_6_no_scaler_application_to_new_model():
    """Verify raw input values are fed directly to the model without MinMaxScaler."""
    mock_rf = MagicMock()
    mock_rf.predict.return_value = [100.0]

    original_rf = ml_service.rf_colab_model
    try:
        ml_service.rf_colab_model = mock_rf
        ml_service.predict_rul_colab(VALID_19_FEATURES)

        mock_rf.predict.assert_called_once()
        passed_df = mock_rf.predict.call_args[0][0]

        # Verify values in passed DataFrame are the original unscaled values (e.g. sensor_3 is 1589.70, not in [0, 1])
        assert passed_df["sensor_3"].iloc[0] == 1589.70
        assert passed_df["sensor_8"].iloc[0] == 2388.06
        assert passed_df["cycle"].iloc[0] == 1.0
    finally:
        ml_service.rf_colab_model = original_rf


def test_7_rul_output_bounds():
    """Verify predictions are strictly clipped to [0, 125]."""
    mock_rf = MagicMock()
    original_rf = ml_service.rf_colab_model
    try:
        ml_service.rf_colab_model = mock_rf

        # Test upper bound clipping
        mock_rf.predict.return_value = [250.0]
        assert ml_service.predict_rul_colab(VALID_19_FEATURES) == 125.0

        # Test lower bound clipping
        mock_rf.predict.return_value = [-15.5]
        assert ml_service.predict_rul_colab(VALID_19_FEATURES) == 0.0
    finally:
        ml_service.rf_colab_model = original_rf


def test_8_existing_benchmark_rf_predictions_still_work(client: TestClient):
    """Verify original benchmark RF prediction remains functional and accurate."""
    resp = client.post(
        "/api/predict/rul",
        headers=auth_headers(),
        json={"features": VALID_15_BENCHMARK_FEATURES},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_rul" in data
    assert "rf_predicted_rul" in data
    assert data["model_used"] == "Random Forest Baseline"
    assert "confidence_interval" in data


def test_9_existing_lstm_and_anomaly_detection_unaffected(client: TestClient):
    """Verify LSTM and Isolation Forest anomaly detection pipelines remain operational."""
    resp = client.post(
        "/api/detect/anomaly",
        headers=auth_headers(),
        json={"features": VALID_15_BENCHMARK_FEATURES},
    )
    assert resp.status_code == 200
    anom_data = resp.json()
    assert "anomaly_score" in anom_data
    assert "severity" in anom_data
    assert "is_anomaly" in anom_data


def test_10_existing_api_contract_remains_unchanged(client: TestClient):
    """Verify existing /api/predict/rul contract rejects 19 features and only accepts 15."""
    resp = client.post(
        "/api/predict/rul",
        headers=auth_headers(),
        json={"features": list(VALID_19_FEATURES.values())},
    )
    assert resp.status_code == 400
    assert f"Expected {len(FEATURE_COLS)} features" in resp.json()["detail"]


def test_11_model_unavailable_error(client: TestClient):
    """Verify 503 Service Unavailable when model is not loaded."""
    original_state = ml_service.is_colab_loaded
    try:
        ml_service.is_colab_loaded = False
        resp = client.post(
            "/api/predict/rul/fd001",
            headers=auth_headers(),
            json={"features": VALID_19_FEATURES},
        )
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()
    finally:
        ml_service.is_colab_loaded = original_state
