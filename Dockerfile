FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra serve

COPY src/ src/
RUN uv pip install --no-deps -e .

ENV PORT=8080
ENV F1_LOAD_FROM_GCS=true

CMD ["uv", "run", "uvicorn", "f1_predictor.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
