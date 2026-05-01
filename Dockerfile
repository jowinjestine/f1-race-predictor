FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra serve

COPY src/ src/
RUN uv pip install --no-deps -e .

COPY data/raw/model/Model_H_LightGBM_GOSS_Delta.pkl data/raw/model/
COPY data/raw/model/Model_E_LightGBM_shallow.pkl data/raw/model/
COPY data/raw/laps/all_laps.parquet data/raw/laps/
COPY data/raw/race/all_races.parquet data/raw/race/

ENV PORT=8080

CMD ["uv", "run", "uvicorn", "f1_predictor.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
