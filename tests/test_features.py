"""
피처 엔지니어링 누수·정합성 테스트
"""
import pandas as pd
import pytest

from src.features.build import build_features


@pytest.fixture
def sample_df():
    """단일 대여소 48시간 샘플 데이터"""
    import numpy as np
    dates = pd.date_range("2025-03-01", periods=48, freq="h")
    return pd.DataFrame({
        "station_id": "00102",
        "station_name": "테스트",
        "datetime": dates,
        "ride_count": np.random.randint(1, 10, 48),
        "temp": 15.0,
        "rainfall": 0.0,
        "wind_speed": 2.0,
        "humidity": 60.0,
        "hour": dates.hour,
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "is_weekend": dates.dayofweek.isin([5, 6]).astype(int),
    })


def test_lag1_no_leakage(sample_df):
    """lag_1h가 현재 시점의 값을 참조하지 않는지 확인"""
    result = build_features(sample_df)
    result = result.reset_index(drop=True)
    # lag_1h는 이전 행의 ride_count여야 함
    for i in range(1, min(5, len(result))):
        assert result.loc[i, "lag_1h"] != result.loc[i, "ride_count"] or \
               result.loc[i - 1, "ride_count"] == result.loc[i, "ride_count"]


def test_feature_cols_exist(sample_df):
    """FEATURE_COLS 모든 컬럼이 존재하는지 확인"""
    from src.models.train import FEATURE_COLS
    result = build_features(sample_df)
    for col in FEATURE_COLS:
        assert col in result.columns, f"{col} 컬럼 없음"


def test_no_future_leakage(sample_df):
    """lag 피처가 음수 shift가 아닌지 확인 (미래 데이터 누수 방지)"""
    result = build_features(sample_df).reset_index(drop=True)
    if len(result) >= 2:
        # lag_1h는 반드시 현재 datetime보다 이전 시점 값이어야 함
        assert result["lag_1h"].iloc[1] == pytest.approx(
            sample_df[sample_df["datetime"] < result["datetime"].iloc[1]]["ride_count"].iloc[-1],
            abs=1e-6
        )
