"""
MLflow 모델 레지스트리 등록 및 추론 검증
- 베스트 run(lightgbm-v1) 을 Registry에 등록
- Staging 승격
- 로드 후 샘플 추론으로 재현성 검증
실행: python -m src.models.register
"""
from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import settings
from src.models.train import FEATURE_COLS, PROCESSED_DIR, split_data

MODEL_NAME = "ddareungi-lgbm"
EXPERIMENT_NAME = "ddareungi-demand-forecast"


def get_best_run(client: MlflowClient) -> str:
    """실험에서 valid_rmse 기준 베스트 run_id 반환"""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.model_type = 'lightgbm'",
        order_by=["metrics.valid_rmse ASC"],
    )
    if not runs:
        raise ValueError("LightGBM run을 찾을 수 없습니다. 먼저 train.py를 실행하세요.")
    best = runs[0]
    print(f"베스트 run: {best.info.run_name}  (run_id: {best.info.run_id})")
    print(f"  valid_rmse: {best.data.metrics['valid_rmse']:.4f}")
    print(f"  test_rmse:  {best.data.metrics['test_rmse']:.4f}")
    return best.info.run_id


def register_model(run_id: str, client: MlflowClient) -> int:
    """run의 모델을 레지스트리에 등록하고 버전 번호 반환"""
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    version = result.version
    print(f"\n레지스트리 등록 완료: {MODEL_NAME} v{version}")
    return version


def promote_to_staging(version: int, client: MlflowClient) -> None:
    """모델을 Staging으로 승격"""
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="staging",
        version=version,
    )
    print(f"Staging 승격 완료: {MODEL_NAME} v{version} @staging")


def verify_inference(client: MlflowClient) -> None:
    """레지스트리에서 모델 로드 후 샘플 추론 검증"""
    print("\n── 추론 검증 ──")
    model_uri = f"models:/{MODEL_NAME}@staging"
    model = mlflow.lightgbm.load_model(model_uri)

    # 테스트 데이터로 샘플 추론
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    _, _, test = split_data(df)
    sample = test[FEATURE_COLS].head(5)

    preds = model.predict(sample)
    actual = test["ride_count"].head(5).values

    print("샘플 추론 결과 (실제값 vs 예측값):")
    for i, (a, p) in enumerate(zip(actual, preds)):
        print(f"  [{i+1}] 실제: {a:4.0f}  예측: {p:6.2f}")
    print("추론 검증 완료")


def main() -> None:
    print("=== Week 1 Day 6: 모델 레지스트리 등록 ===\n")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    # 1. 베스트 run 찾기
    run_id = get_best_run(client)

    # 2. 레지스트리 등록
    version = register_model(run_id, client)

    # 3. Staging 승격
    promote_to_staging(version, client)

    # 4. 로드 후 추론 검증
    verify_inference(client)

    print(f"\nMLflow UI에서 확인: http://localhost:5000/#/models/{MODEL_NAME}")


if __name__ == "__main__":
    main()
