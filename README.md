# 따릉이 수요 예측 MLOps 서비스

서울 공공자전거 따릉이의 시간대별 대여 수요를 예측하고,
학습 → 레지스트리 → API 서빙 → 모니터링 → 재학습 루프를 갖춘 MLOps 파이프라인.

> **포트폴리오 목적**: 모델 정확도보다 실험추적·서빙·배포·모니터링·재현성에 초점

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

## 빠른 시작

```bash
# 1. 의존성 설치
poetry install

# 2. 환경변수 설정
cp .env.example .env
# .env에 API 키 입력

# 3. 학습
python train.py

# 4. MLflow UI
mlflow ui --port 5000
```

## 프로젝트 구조

```
ddareungi/
├── data/
│   ├── raw/          # 원본 데이터 (git 제외)
│   └── processed/    # 전처리 결과 parquet (git 제외)
├── docs/
│   └── problem_definition.md
├── notebooks/        # EDA
├── src/
│   ├── config.py
│   ├── features/     # 피처 엔지니어링 (학습·서빙 공유)
│   ├── models/       # 학습·평가
│   └── serving/      # FastAPI 앱
├── tests/
├── docker/
├── train.py          # 학습 진입점
├── pyproject.toml
└── .env.example
```

## 주차별 진행 현황

- [x] Week 1 Day 1 — 문제 정의 & 레포 스캐폴드
- [ ] Week 1 Day 2 — 데이터 수집·정합
- [ ] Week 1 Day 3 — EDA & 피처 엔지니어링
- [ ] Week 1 Day 4 — 베이스라인 모델
- [ ] Week 1 Day 5 — MLflow 실험추적
- [ ] Week 1 Day 6 — 모델 레지스트리
- [ ] Week 1 Day 7 — README·모델카드
