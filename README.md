# 따릉이 수요 예측 MLOps 서비스

서울 공공자전거 따릉이의 시간대별 대여 수요를 예측하고,
학습 → 레지스트리 → API 서빙 → 모니터링 → 재학습 루프를 갖춘 MLOps 파이프라인.

> **포트폴리오 목적**: 모델 정확도보다 실험추적·서빙·배포·모니터링·재현성에 초점

---

## 기술 스택

| 역할 | 도구 |
|------|------|
| 모델 | LightGBM |
| 실험추적·레지스트리 | MLflow |
| 서빙 | FastAPI + Uvicorn |
| 컨테이너 | Docker (멀티스테이지) + docker-compose |
| CI/CD | GitHub Actions |
| 모니터링 | Evidently |
| 배포 | AWS App Runner / ECS Fargate |

---

## 빠른 시작

```bash
# 1. 의존성 설치
poetry install

# 2. 환경변수 설정
cp .env.example .env
# .env에 API 키 입력 (서울 열린데이터광장, 기상청)

# 3. 따릉이 CSV 다운로드
# https://data.seoul.go.kr → "공공자전거 시간대별 이용정보"
# data/raw/ 에 저장

# 4. 전체 파이프라인 실행 (데이터 수집 → 학습)
python train.py

# 5. MLflow UI
mlflow server --host 127.0.0.1 --port 5000
# http://localhost:5000 접속
```

---

## 프로젝트 구조

```
ddareungi/
├── data/
│   ├── raw/            # 원본 CSV (git 제외)
│   └── processed/      # 전처리 결과 parquet (git 제외)
├── docs/
│   ├── problem_definition.md   # 문제 정의
│   └── model_card.md           # 모델카드
├── notebooks/          # EDA
├── src/
│   ├── config.py       # 환경변수 설정
│   ├── data/
│   │   ├── collect.py  # 따릉이 CSV 수집·집계
│   │   ├── weather.py  # 기상청 ASOS 날씨 수집
│   │   ├── join.py     # 데이터 조인
│   │   └── eda.py      # EDA
│   ├── features/
│   │   └── build.py    # 피처 엔지니어링 (lag/rolling)
│   ├── models/
│   │   ├── train.py    # 학습 + MLflow 실험추적
│   │   └── register.py # 모델 레지스트리 등록
│   └── serving/        # FastAPI 앱 (Week 2)
├── tests/
├── docker/
├── train.py            # 학습 진입점 (python train.py 한 줄 실행)
├── pyproject.toml
└── .env.example
```

---

## 파이프라인 흐름

```
[따릉이 CSV]  ──┐
                ├→ collect.py → rides.parquet
[기상청 ASOS] ──┘
                    ↓
               join.py → dataset.parquet
                    ↓
              build.py → features.parquet
                    ↓
              train.py → MLflow 실험 기록
                    ↓
            register.py → 모델 레지스트리 (Staging)
```

---

## 모델 성능

| 모델 | valid RMSE | test RMSE |
|------|-----------|----------|
| Baseline-Lag1 | 3.04 | - |
| Baseline-Seasonal | 2.05 | - |
| **LightGBM v1** | **2.13** | **1.66** |

- 평가지표: RMSE (낮을수록 좋음)
- 시계열 분할: train(1~10월) / valid(11월) / test(12월)

---

## 주차별 진행 현황

- [x] Week 1 Day 1 — 문제 정의 & 레포 스캐폴드
- [x] Week 1 Day 2 — 데이터 수집·정합
- [x] Week 1 Day 3 — EDA & 피처 엔지니어링
- [x] Week 1 Day 4 — 베이스라인 모델 학습
- [x] Week 1 Day 5 — MLflow 실험추적
- [x] Week 1 Day 6 — 모델 레지스트리 등록
- [x] Week 1 Day 7 — README & 모델카드
- [ ] Week 2 — FastAPI 추론 API + Docker
- [ ] Week 3 — GitHub Actions CI/CD + AWS 배포
- [ ] Week 4 — Evidently 모니터링 + 재학습 루프
