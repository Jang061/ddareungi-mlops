"""
시계열 분할 → Naive 베이스라인 + LightGBM 학습 → 지표 비교
실행: python -m src.models.train
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import settings

PROCESSED_DIR = Path("data/processed")
RANDOM_SEED = settings.random_seed

# 피처 컬럼 (학습-서빙 스큐 방지: 서빙에서도 동일하게 사용)
FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend",
    "temp", "rainfall", "wind_speed", "humidity",
    "lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h",
]
TARGET_COL = "ride_count"


# ── 시계열 분할 ───────────────────────────────────────────
def split_data(df: pd.DataFrame):
    """
    시간 기준 분할 (랜덤 분할 금지)
    train: ~ 10월 31일
    valid: 11월
    test:  12월
    """
    train = df[df["datetime"] < "2025-11-01"]
    valid = df[(df["datetime"] >= "2025-11-01") & (df["datetime"] < "2025-12-01")]
    test  = df[df["datetime"] >= "2025-12-01"]
    return train, valid, test


# ── 평가 지표 ─────────────────────────────────────────────
def evaluate(y_true: pd.Series, y_pred: np.ndarray, name: str) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"  [{name}] RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae}


# ── Naive 베이스라인 ──────────────────────────────────────
def naive_lag1(valid: pd.DataFrame) -> np.ndarray:
    """Baseline-1: 직전 1시간 값 그대로"""
    return valid["lag_1h"].fillna(valid[TARGET_COL].mean()).values


def naive_seasonal(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    """Baseline-2: 같은 요일·시간 최근 4주 평균"""
    ref = train.copy()
    avg = (
        ref.groupby(["station_id", "day_of_week", "hour"])[TARGET_COL]
        .mean()
        .reset_index()
        .rename(columns={TARGET_COL: "pred_seasonal"})
    )
    merged = valid[["station_id", "day_of_week", "hour"]].merge(avg, on=["station_id", "day_of_week", "hour"], how="left")
    return merged["pred_seasonal"].fillna(train[TARGET_COL].mean()).values


# ── LightGBM 학습 ─────────────────────────────────────────
def train_lgbm(train: pd.DataFrame, valid: pd.DataFrame) -> lgb.Booster:
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_valid = valid[FEATURE_COLS]
    y_valid = valid[TARGET_COL]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": RANDOM_SEED,
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    return model


# ── 메인 ─────────────────────────────────────────────────
def main() -> None:
    print("=== Week 1 Day 4: 베이스라인 모델 학습 ===\n")

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    train, valid, test = split_data(df)

    print(f"train: {len(train):,} rows ({train['datetime'].min().date()} ~ {train['datetime'].max().date()})")
    print(f"valid: {len(valid):,} rows ({valid['datetime'].min().date()} ~ {valid['datetime'].max().date()})")
    print(f"test:  {len(test):,} rows  ({test['datetime'].min().date()} ~ {test['datetime'].max().date()})\n")

    results = []

    # Naive 베이스라인 (validation 기준)
    print("── Naive 베이스라인 (validation) ──")
    results.append(evaluate(valid[TARGET_COL], naive_lag1(valid), "Baseline-Lag1"))
    results.append(evaluate(valid[TARGET_COL], naive_seasonal(train, valid), "Baseline-Seasonal"))

    # LightGBM
    print("\n── LightGBM 학습 ──")
    model = train_lgbm(train, valid)
    pred_valid = model.predict(valid[FEATURE_COLS])
    results.append(evaluate(valid[TARGET_COL], pred_valid, "LightGBM-valid"))

    # 테스트셋 최종 성능
    pred_test = model.predict(test[FEATURE_COLS])
    results.append(evaluate(test[TARGET_COL], pred_test, "LightGBM-test"))

    # 결과 요약
    print("\n── 결과 요약 ──")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))

    # 피처 중요도 상위 10개
    print("\n── 피처 중요도 (상위 10) ──")
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    for feat, val in importance.head(10).items():
        print(f"  {feat:20s}: {val:.0f}")

    # 모델 저장
    model_path = PROCESSED_DIR / "lgbm_model.txt"
    model.save_model(str(model_path))
    print(f"\n모델 저장: {model_path}")


if __name__ == "__main__":
    main()
