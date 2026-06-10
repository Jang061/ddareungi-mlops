"""
Pydantic 스키마 유효성 검증 테스트
"""
import pytest
from pydantic import ValidationError

from src.serving.schema import PredictRequest, PredictResponse


VALID_INPUT = {
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


def test_valid_request():
    req = PredictRequest(**VALID_INPUT)
    assert req.station_id == "00102"
    assert req.hour == 18


def test_hour_out_of_range():
    with pytest.raises(ValidationError):
        PredictRequest(**{**VALID_INPUT, "hour": 24})


def test_negative_hour():
    with pytest.raises(ValidationError):
        PredictRequest(**{**VALID_INPUT, "hour": -1})


def test_humidity_out_of_range():
    with pytest.raises(ValidationError):
        PredictRequest(**{**VALID_INPUT, "humidity": 101})


def test_negative_rainfall_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(**{**VALID_INPUT, "rainfall": -1.0})


def test_rainfall_default_zero():
    data = {k: v for k, v in VALID_INPUT.items() if k != "rainfall"}
    req = PredictRequest(**data)
    assert req.rainfall == 0.0


def test_valid_response():
    resp = PredictResponse(
        station_id="00102",
        predicted_ride_count=8.01,
        model_version="v1",
    )
    assert resp.predicted_ride_count == 8.01
