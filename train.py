"""
따릉이 수요 예측 학습 파이프라인 진입점
실행: python train.py
"""
from src.data.collect import load_raw_files, aggregate, save as save_rides
from src.data.weather import collect_year
from src.data.join import main as join_main
from src.features.build import main as feature_main
from src.models.train import main as train_main
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def main():
    print("=" * 60)
    print("따릉이 수요 예측 학습 파이프라인")
    print("=" * 60)

    # Step 1: 따릉이 이용내역 집계
    if not (PROCESSED_DIR / "rides.parquet").exists():
        print("\n[Step 1] 따릉이 이용내역 수집·집계")
        raw = load_raw_files()
        agg = aggregate(raw)
        save_rides(agg, PROCESSED_DIR / "rides.parquet")
    else:
        print("\n[Step 1] rides.parquet 이미 존재 -스킵")

    # Step 2: 날씨 데이터 수집
    if not (PROCESSED_DIR / "weather.parquet").exists():
        print("\n[Step 2] 기상청 날씨 데이터 수집")
        import pandas as pd
        df = collect_year(2025)
        df.to_parquet(PROCESSED_DIR / "weather.parquet", index=False)
    else:
        print("\n[Step 2] weather.parquet 이미 존재 -스킵")

    # Step 3: 데이터 조인
    if not (PROCESSED_DIR / "dataset.parquet").exists():
        print("\n[Step 3] 데이터 조인")
        join_main()
    else:
        print("\n[Step 3] dataset.parquet 이미 존재 -스킵")

    # Step 4: 피처 엔지니어링
    if not (PROCESSED_DIR / "features.parquet").exists():
        print("\n[Step 4] 피처 엔지니어링")
        feature_main()
    else:
        print("\n[Step 4] features.parquet 이미 존재 -스킵")

    # Step 5: 모델 학습
    print("\n[Step 5] 모델 학습")
    train_main()

    print("\n" + "=" * 60)
    print("파이프라인 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
