"""
따릉이 이용내역 CSV → (대여소, 날짜, 시간) 단위 집계 → parquet 저장
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_raw_files() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("서울특별시 공공자전거 이용정보(시간대별)_*.csv"))
    if not files:
        raise FileNotFoundError(f"CSV 파일이 {RAW_DIR}에 없습니다.")

    frames = []
    for f in files:
        print(f"  읽는 중: {f.name}")
        df = pd.read_csv(
            f,
            encoding="cp949",
            usecols=["대여일자", "대여시간", "대여소번호", "대여소명", "이용건수"],
            dtype={"대여소번호": str},
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df["대여일자"]) + pd.to_timedelta(df["대여시간"], unit="h")

    agg = (
        df.groupby(["대여소번호", "대여소명", "datetime"], as_index=False)["이용건수"]
        .sum()
        .rename(columns={"대여소번호": "station_id", "대여소명": "station_name", "이용건수": "ride_count"})
    )

    agg = agg.sort_values(["station_id", "datetime"]).reset_index(drop=True)
    return agg


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"저장 완료: {path}  ({len(df):,} rows)")


if __name__ == "__main__":
    print("=== 따릉이 이용내역 수집·집계 ===")
    raw = load_raw_files()
    print(f"원본 행 수: {len(raw):,}")

    agg = aggregate(raw)
    print(f"집계 후 행 수: {len(agg):,}")
    print(agg.head())

    save(agg, PROCESSED_DIR / "rides.parquet")
