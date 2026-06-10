# ── Stage 1: 의존성 빌드 ──────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# poetry 설치
RUN pip install poetry==2.4.1

# 의존성 파일만 먼저 복사 (캐시 활용)
COPY pyproject.toml poetry.lock ./

# 가상환경을 프로젝트 내부에 생성
RUN poetry config virtualenvs.in-project true && \
    poetry install --only main --no-root --no-interaction


# ── Stage 2: 실행 이미지 ──────────────────────────────────
FROM python:3.11-slim AS runtime

# LightGBM 런타임 의존성 (libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# builder에서 venv만 복사
COPY --from=builder /app/.venv ./.venv

# 소스코드 복사
COPY src/ ./src/

# 환경변수
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONIOENCODING="utf-8" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
