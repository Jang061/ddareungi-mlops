"""
피처 엔지니어링
- lag 피처: 1h, 24h, 168h(1주) 전 대여 건수
- rolling 피처: 직전 24h 평균
- 누수 주의: 예측 시점(t) 기준 과거값만 사용
입력: data/processed/dataset.parquet
출력: data/processed/features.parquet
"""
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station_id", "datetime"]).copy()

    # 대여소별로 그룹화하여 lag/rolling 계산 (누수 방지: shift(n) = n시간 전)
    grp = df.groupby("station_id", sort=False)["ride_count"]

    df["lag_1h"] = grp.shift(1)       # 1시간 전
    df["lag_24h"] = grp.shift(24)     # 24시간 전 (어제 같은 시각)
    df["lag_168h"] = grp.shift(168)   # 1주 전 같은 시각

    # 직전 24시간 평균 (현재 시점 제외 → shift(1) 후 rolling)
    df["rolling_mean_24h"] = grp.shift(1).transform(
        lambda x: x.rolling(window=24, min_periods=1).mean()
    )

    # lag 피처가 없는 초기 행 제거 (168h = 7일치)
    df = df.dropna(subset=["lag_168h"]).reset_index(drop=True)

    return df


def main() -> None:
    print("=== 피처 엔지니어링 ===")
    df = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    print(f"입력: {len(df):,} rows")

    df = build_features(df)
    print(f"출력: {len(df):,} rows (앞 168시간 제거)")
    print(f"컬럼: {df.columns.tolist()}")

    # 누수 점검: lag_1h가 미래값인지 확인
    sample = df[df["station_id"] == df["station_id"].iloc[0]].head(5)
    print("\n[누수 점검] lag_1h = 1행 전 ride_count 인지 확인:")
    print(sample[["datetime", "ride_count", "lag_1h", "lag_24h"]].to_string())

    out = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
