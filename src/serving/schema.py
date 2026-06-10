"""
추론 API 입력/출력 스키마 초안 (Week 2 FastAPI에서 사용)
피처 목록은 src/models/train.py FEATURE_COLS와 반드시 동일해야 함 (학습-서빙 스큐 방지)
"""
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """단일 대여소 1시간 수요 예측 요청"""

    station_id: str = Field(..., description="대여소 ID (예: 00102)")
    hour: int = Field(..., ge=0, le=23, description="예측 시간 (0~23)")
    day_of_week: int = Field(..., ge=0, le=6, description="요일 (0=월 ~ 6=일)")
    month: int = Field(..., ge=1, le=12, description="월")
    is_weekend: int = Field(..., ge=0, le=1, description="주말 여부 (0/1)")

    # 날씨 피처 (기상청 단기예보 API에서 조회)
    temp: float = Field(..., description="기온 (°C)")
    rainfall: float = Field(0.0, ge=0, description="강수량 (mm)")
    wind_speed: float = Field(..., ge=0, description="풍속 (m/s)")
    humidity: float = Field(..., ge=0, le=100, description="습도 (%)")

    # lag/rolling 피처 (실시간 대여 이력에서 계산)
    lag_1h: float = Field(..., ge=0, description="1시간 전 대여 건수")
    lag_24h: float = Field(..., ge=0, description="24시간 전 대여 건수")
    lag_168h: float = Field(..., ge=0, description="1주 전 동시간 대여 건수")
    rolling_mean_24h: float = Field(..., ge=0, description="직전 24시간 평균 대여 건수")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictResponse(BaseModel):
    """예측 결과"""

    station_id: str
    predicted_ride_count: float = Field(..., description="예측 대여 건수")
    model_version: str = Field(..., description="사용된 모델 버전")
