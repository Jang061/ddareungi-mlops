"""
FastAPI 엔드포인트 테스트
모델 로드 없이 mock으로 대체하여 빠르게 테스트
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


VALID_PAYLOAD = {
    "station_id": "00102",
    "hour": 18,
    "day_of_week": 0,
    "month": 6,
    "is_weekend": 0,
    "temp": 24.5,
    "rainfall": 0.0,
    "wind_speed": 2.1,
    "humidity": 65.0,
    "lag_1h": 5.0,
    "lag_24h": 4.0,
    "lag_168h": 6.0,
    "rolling_mean_24h": 3.8,
}


@pytest.fixture
def client():
    """모델을 mock으로 대체한 테스트 클라이언트"""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([8.01])

    mock_mv = MagicMock()
    mock_mv.version = "1"

    with patch("src.serving.app._model", mock_model), \
         patch("src.serving.app._model_version", "v1"):
        from src.serving.app import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_version"] == "v1"


def test_predict_success(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["station_id"] == "00102"
    assert body["predicted_ride_count"] >= 0
    assert body["model_version"] == "v1"


def test_predict_missing_field(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "temp"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422  # Unprocessable Entity


def test_predict_invalid_hour(client):
    r = client.post("/predict", json={**VALID_PAYLOAD, "hour": 25})
    assert r.status_code == 422


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "predict" in r.json()
