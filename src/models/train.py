"""
시계열 분할 -> Naive 베이스라인 + LightGBM 학습 -> MLflow 실험추적
실행: python -m src.models.train
"""
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import settings

PROCESSED_DIR = Path("data/processed")
RANDOM_SEED = settings.random_seed
EXPERIMENT_NAME = "ddareungi-demand-forecast"

# 피처 컬럼 (학습-서빙 스큐 방지: 서빙에서도 동일하게 사용)
FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend",
    "temp", "rainfall", "wind_speed", "humidity",
    "lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h",
]
TARGET_COL = "ride_count"


# -- 시계열 분할 ------------------------------------------------------
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


# -- 평가 지표 --------------------------------------------------------
def calc_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    return {"rmse": rmse, "mae": mae}


# -- Naive 베이스라인 -------------------------------------------------
def naive_lag1(valid: pd.DataFrame) -> np.ndarray:
    return valid["lag_1h"].fillna(valid[TARGET_COL].mean()).values


def naive_seasonal(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    avg = (
        train.groupby(["station_id", "day_of_week", "hour"])[TARGET_COL]
        .mean()
        .reset_index()
        .rename(columns={TARGET_COL: "pred_seasonal"})
    )
    merged = valid[["station_id", "day_of_week", "hour"]].merge(
        avg, on=["station_id", "day_of_week", "hour"], how="left"
    )
    return merged["pred_seasonal"].fillna(train[TARGET_COL].mean()).values


# -- LightGBM 학습 ---------------------------------------------------
def train_lgbm(train: pd.DataFrame, valid: pd.DataFrame) -> lgb.Booster:
    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_valid, y_valid = valid[FEATURE_COLS], valid[TARGET_COL]

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
    return model, params


# -- 메인 ------------------------------------------------------------
def main() -> None:
    print("=== Week 1 Day 5: MLflow 실험추적 ===\n")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    train, valid, test = split_data(df)

    print(f"train: {len(train):,} rows  ({train['datetime'].min().date()} ~ {train['datetime'].max().date()})")
    print(f"valid: {len(valid):,} rows  ({valid['datetime'].min().date()} ~ {valid['datetime'].max().date()})")
    print(f"test:  {len(test):,} rows   ({test['datetime'].min().date()} ~ {test['datetime'].max().date()})\n")

    # ── Baseline-Lag1 run ──────────────────────────────────────────
    with mlflow.start_run(run_name="baseline-lag1"):
        preds = naive_lag1(valid)
        m = calc_metrics(valid[TARGET_COL], preds)
        mlflow.log_metrics({f"valid_{k}": v for k, v in m.items()})
        mlflow.set_tag("model_type", "baseline")
        print(f"[Baseline-Lag1]      RMSE: {m['rmse']:.4f} | MAE: {m['mae']:.4f}")

    # ── Baseline-Seasonal run ──────────────────────────────────────
    with mlflow.start_run(run_name="baseline-seasonal"):
        preds = naive_seasonal(train, valid)
        m = calc_metrics(valid[TARGET_COL], preds)
        mlflow.log_metrics({f"valid_{k}": v for k, v in m.items()})
        mlflow.set_tag("model_type", "baseline")
        print(f"[Baseline-Seasonal]  RMSE: {m['rmse']:.4f} | MAE: {m['mae']:.4f}")

    # ── LightGBM run ───────────────────────────────────────────────
    with mlflow.start_run(run_name="lightgbm-v1") as run:
        model, params = train_lgbm(train, valid)

        # params 로깅
        mlflow.log_params(params)
        mlflow.log_params({
            "num_boost_round": model.num_trees(),
            "train_rows": len(train),
            "valid_rows": len(valid),
            "test_rows": len(test),
            "features": len(FEATURE_COLS),
            "random_seed": RANDOM_SEED,
        })

        # valid metrics
        pred_valid = model.predict(valid[FEATURE_COLS])
        m_valid = calc_metrics(valid[TARGET_COL], pred_valid)
        mlflow.log_metrics({f"valid_{k}": v for k, v in m_valid.items()})

        # test metrics
        pred_test = model.predict(test[FEATURE_COLS])
        m_test = calc_metrics(test[TARGET_COL], pred_test)
        mlflow.log_metrics({f"test_{k}": v for k, v in m_test.items()})

        # 피처 중요도 저장
        importance = pd.Series(
            model.feature_importance(importance_type="gain"),
            index=FEATURE_COLS
        ).sort_values(ascending=False)
        imp_path = PROCESSED_DIR / "feature_importance.csv"
        importance.to_csv(imp_path, header=["importance"])
        mlflow.log_artifact(str(imp_path), artifact_path="feature")

        # 모델 아티팩트 저장
        mlflow.lightgbm.log_model(model, artifact_path="model")

        mlflow.set_tag("model_type", "lightgbm")
        mlflow.set_tag("split", "time-based")

        print(f"[LightGBM valid]     RMSE: {m_valid['rmse']:.4f} | MAE: {m_valid['mae']:.4f}")
        print(f"[LightGBM test]      RMSE: {m_test['rmse']:.4f}  | MAE: {m_test['mae']:.4f}")
        print(f"\nMLflow run_id: {run.info.run_id}")

    print(f"\nMLflow UI: mlflow ui --port 5000")
    print(f"실험 이름: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
