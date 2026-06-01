"""
기상청 종관기상관측(ASOS) 시간자료 → 서울 시간별 날씨 → parquet 저장
서울 관측소 ID: 108
"""
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.config import settings

PROCESSED_DIR = Path("data/processed")
STATION_ID = 108  # 서울 종관기상관측소
BASE_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"


def fetch_month(year: int, month: int) -> pd.DataFrame:
    start = date(year, month, 1)
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)

    url = (
        f"{BASE_URL}?serviceKey={settings.kma_api_key}"
        f"&numOfRows=800&pageNo=1&dataType=JSON"
        f"&dataCd=ASOS&dateCd=HR"
        f"&startDt={start.strftime('%Y%m%d')}&startHh=00"
        f"&endDt={end.strftime('%Y%m%d')}&endHh=23"
        f"&stnIds={STATION_ID}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    result_code = data["response"]["header"]["resultCode"]
    if result_code != "00":
        msg = data["response"]["header"]["resultMsg"]
        raise ValueError(f"API 오류: {result_code} - {msg}")

    items = data["response"]["body"]["items"]["item"]
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)
    return df


def process(df: pd.DataFrame) -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df["tm"], format="%Y-%m-%d %H:%M")

    col_map = {
        "ta": "temp",       # 기온 (°C)
        "rn": "rainfall",   # 강수량 (mm)
        "ws": "wind_speed", # 풍속 (m/s)
        "hm": "humidity",   # 습도 (%)
    }
    keep = ["datetime"] + [c for c in col_map if c in df.columns]
    df = df[keep].rename(columns=col_map)

    for col in ["temp", "rainfall", "wind_speed", "humidity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rainfall"] = df["rainfall"].fillna(0)
    return df.sort_values("datetime").reset_index(drop=True)


def collect_year(year: int) -> pd.DataFrame:
    frames = []
    for month in range(1, 13):
        print(f"  {year}-{month:02d} 수집 중...")
        try:
            raw = fetch_month(year, month)
            if not raw.empty:
                frames.append(process(raw))
        except Exception as e:
            print(f"  오류 {year}-{month:02d}: {e}")
        time.sleep(0.1)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    print("=== 기상청 ASOS 시간자료 수집 (2025년, 서울 108) ===")
    df = collect_year(2025)
    print(f"수집 행 수: {len(df):,}")
    if not df.empty:
        print(df.head())

    out = PROCESSED_DIR / "weather.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"저장 완료: {out}")
