# 모델카드 — 따릉이 수요 예측 LightGBM v1

## 모델 개요

| 항목 | 내용 |
|------|------|
| 모델명 | ddareungi-lgbm |
| 버전 | v1 |
| 알고리즘 | LightGBM (gradient boosting regression) |
| 학습일 | 2025-06 |
| 레지스트리 상태 | Staging |

## 목적

서울 따릉이 대여소별 시간대 대여 수요(ride_count)를 1시간 단위로 예측한다.
운영자가 자전거 재배치 계획을 수립하거나 수요 급증 대여소를 사전에 파악하는 데 활용한다.

## 학습 데이터

| 항목 | 내용 |
|------|------|
| 출처 | 서울 열린데이터광장 공공자전거 이용정보 + 기상청 ASOS |
| 기간 | 2025년 1월 ~ 10월 (train) |
| 그레인 | (대여소 ID, 날짜, 시간) — 1행 = 1대여소 × 1시간 |
| 행 수 | 952만 행 (train 기준) |

## 피처

| 피처 | 설명 | 종류 |
|------|------|------|
| hour | 시간 (0~23) | 시간 |
| day_of_week | 요일 (0=월~6=일) | 시간 |
| month | 월 | 시간 |
| is_weekend | 주말 여부 | 시간 |
| temp | 기온 (°C) | 날씨 |
| rainfall | 강수량 (mm) | 날씨 |
| wind_speed | 풍속 (m/s) | 날씨 |
| humidity | 습도 (%) | 날씨 |
| lag_1h | 직전 1시간 대여 건수 | lag |
| lag_24h | 24시간 전 대여 건수 | lag |
| lag_168h | 1주 전 동시간 대여 건수 | lag |
| rolling_mean_24h | 직전 24시간 평균 대여 건수 | rolling |

## 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| learning_rate | 0.05 |
| num_leaves | 63 |
| min_child_samples | 20 |
| feature_fraction | 0.8 |
| bagging_fraction | 0.8 |
| num_boost_round | 500 (early stopping 50) |
| seed | 42 |

## 성능

| 구간 | RMSE | MAE |
|------|------|-----|
| Baseline-Lag1 (valid) | 3.04 | 1.74 |
| Baseline-Seasonal (valid) | 2.05 | 1.30 |
| **LightGBM v1 (valid)** | **2.13** | **1.35** |
| **LightGBM v1 (test)** | **1.66** | **1.05** |

- 평가 기준: RMSE (건수 단위)
- test RMSE 1.66 = 예측이 실제와 평균 1.66건 차이

## 피처 중요도 (상위 5)

| 순위 | 피처 | 의미 |
|------|------|------|
| 1 | lag_1h | 직전 시간이 가장 강한 신호 |
| 2 | hour | 출퇴근 시간대 패턴 |
| 3 | rolling_mean_24h | 최근 추세 반영 |
| 4 | day_of_week | 주중/주말 패턴 |
| 5 | lag_24h | 어제 같은 시간 패턴 |

## 한계 및 주의사항

- 서울 전체 단일 날씨 사용 (대여소별 미세 날씨 차이 미반영)
- 공휴일 피처 미포함 (명절 등 특수일 예측 정확도 낮을 수 있음)
- 신규 대여소는 lag 피처 계산 불가 → 평균값으로 대체 필요
- 학습 데이터가 2025년 1년치라 장기 트렌드 변화에 취약

## 재현 방법

```bash
python train.py
```

의존성 및 시드 고정으로 동일한 결과 재현 가능 (seed=42).
