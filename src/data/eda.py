"""
EDA — dataset.parquet 기본 통계 및 패턴 확인
결과를 data/processed/eda_report.txt 로 저장
"""
from pathlib import Path

import pandas as pd
import numpy as np

PROCESSED_DIR = Path("data/processed")


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    lines = []

    # 1. 기본 정보
    lines.append("=" * 60)
    lines.append("1. 기본 정보")
    lines.append("=" * 60)
    lines.append(f"행 수:       {len(df):,}")
    lines.append(f"컬럼 수:     {df.shape[1]}")
    lines.append(f"기간:        {df['datetime'].min()} ~ {df['datetime'].max()}")
    lines.append(f"대여소 수:   {df['station_id'].nunique():,}")
    lines.append(f"컬럼 목록:   {df.columns.tolist()}")

    # 2. 타깃 분포
    lines.append("\n" + "=" * 60)
    lines.append("2. 타깃(ride_count) 분포")
    lines.append("=" * 60)
    rc = df["ride_count"]
    lines.append(f"평균:   {rc.mean():.2f}")
    lines.append(f"중앙값: {rc.median():.2f}")
    lines.append(f"최대:   {rc.max()}")
    lines.append(f"최소:   {rc.min()}")
    lines.append(f"표준편차: {rc.std():.2f}")
    lines.append(f"0건 비율: {(rc == 0).mean() * 100:.1f}%")

    # 3. 시간대별 평균 대여 건수
    lines.append("\n" + "=" * 60)
    lines.append("3. 시간대별 평균 대여 건수 (상위/하위 3개)")
    lines.append("=" * 60)
    by_hour = df.groupby("hour")["ride_count"].mean().sort_values(ascending=False)
    lines.append("많은 시간대:")
    for h, v in by_hour.head(3).items():
        lines.append(f"  {h:02d}시: {v:.2f}건")
    lines.append("적은 시간대:")
    for h, v in by_hour.tail(3).items():
        lines.append(f"  {h:02d}시: {v:.2f}건")

    # 4. 요일별 평균 대여 건수
    lines.append("\n" + "=" * 60)
    lines.append("4. 요일별 평균 대여 건수")
    lines.append("=" * 60)
    day_names = ["월", "화", "수", "목", "금", "토", "일"]
    by_dow = df.groupby("day_of_week")["ride_count"].mean()
    for d, v in by_dow.items():
        lines.append(f"  {day_names[d]}요일: {v:.2f}건")

    # 5. 날씨 상관관계
    lines.append("\n" + "=" * 60)
    lines.append("5. 날씨 피처와 ride_count 상관관계")
    lines.append("=" * 60)
    weather_cols = ["temp", "rainfall", "wind_speed", "humidity"]
    corr = df[weather_cols + ["ride_count"]].corr()["ride_count"].drop("ride_count")
    for col, val in corr.sort_values(ascending=False).items():
        lines.append(f"  {col:12s}: {val:+.3f}")

    # 6. 결측치
    lines.append("\n" + "=" * 60)
    lines.append("6. 결측치")
    lines.append("=" * 60)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        lines.append("  결측치 없음")
    else:
        for col, cnt in missing.items():
            lines.append(f"  {col}: {cnt:,}")

    # 7. 이상치 (ride_count 상위 0.1%)
    lines.append("\n" + "=" * 60)
    lines.append("7. ride_count 이상치 (상위 0.1%)")
    lines.append("=" * 60)
    threshold = rc.quantile(0.999)
    outliers = df[rc > threshold]
    lines.append(f"  기준값(99.9%): {threshold:.0f}건")
    lines.append(f"  이상치 행 수: {len(outliers):,}")

    report = "\n".join(lines)
    print(report)

    out = PROCESSED_DIR / "eda_report.txt"
    out.write_text(report, encoding="utf-8")
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
