"""
rides.parquet + weather.parquet → (대여소, 시간, 날씨) 통합 dataset.parquet
"""
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")


def main() -> None:
    print("=== 데이터 조인 ===")
    rides = pd.read_parquet(PROCESSED_DIR / "rides.parquet")
    weather = pd.read_parquet(PROCESSED_DIR / "weather.parquet")

    print(f"rides:   {len(rides):,} rows")
    print(f"weather: {len(weather):,} rows")

    # 날씨는 서울 전체 단일값 → 시간 기준으로 left join
    merged = rides.merge(weather, on="datetime", how="left")

    # 결측 확인
    missing = merged[["temp", "rainfall", "wind_speed", "humidity"]].isna().sum()
    print(f"\n날씨 결측치:\n{missing}")

    # 시간 피처 추가
    merged["hour"] = merged["datetime"].dt.hour
    merged["day_of_week"] = merged["datetime"].dt.dayofweek  # 0=월 ~ 6=일
    merged["month"] = merged["datetime"].dt.month
    merged["is_weekend"] = merged["day_of_week"].isin([5, 6]).astype(int)

    merged = merged.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    out = PROCESSED_DIR / "dataset.parquet"
    merged.to_parquet(out, index=False)
    print(f"\n저장 완료: {out}  ({len(merged):,} rows, {merged.shape[1]} cols)")
    print(merged.head(3).to_string())


if __name__ == "__main__":
    main()
