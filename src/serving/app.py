"""
따릉이 수요 예측 FastAPI 서빙 앱
실행: uvicorn src.serving.app:app --reload --port 8000
"""
import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow.lightgbm
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.config import settings
from src.models.train import FEATURE_COLS
from src.serving.schema import PredictRequest, PredictResponse

# 전역 모델 저장소
_model: Any = None
_model_version: str = "unknown"


def load_model() -> None:
    """MLflow 레지스트리에서 @staging 모델 로드"""
    global _model, _model_version
    model_name = os.getenv("MODEL_NAME", "ddareungi-lgbm")
    model_alias = os.getenv("MODEL_ALIAS", "staging")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{model_name}@{model_alias}"
    _model = mlflow.lightgbm.load_model(uri)

    # 버전 정보 조회
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, model_alias)
    _model_version = f"v{mv.version}"
    print(f"모델 로드 완료: {model_name} {_model_version} (@{model_alias})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 모델 로드, 종료 시 정리"""
    load_model()
    yield
    _model = None


app = FastAPI(
    title="따릉이 수요 예측 API",
    description="서울 공공자전거 따릉이 대여소별 시간대 수요 예측",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """헬스체크"""
    return {"status": "ok", "model_version": _model_version}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """단일 대여소 1시간 수요 예측"""
    if _model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    features = pd.DataFrame([{
        "hour": req.hour,
        "day_of_week": req.day_of_week,
        "month": req.month,
        "is_weekend": req.is_weekend,
        "temp": req.temp,
        "rainfall": req.rainfall,
        "wind_speed": req.wind_speed,
        "humidity": req.humidity,
        "lag_1h": req.lag_1h,
        "lag_24h": req.lag_24h,
        "lag_168h": req.lag_168h,
        "rolling_mean_24h": req.rolling_mean_24h,
    }])[FEATURE_COLS]

    pred = float(_model.predict(features)[0])
    pred = max(0.0, round(pred, 2))  # 음수 방지

    return PredictResponse(
        station_id=req.station_id,
        predicted_ride_count=pred,
        model_version=_model_version,
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "따릉이 수요 예측 API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }
