FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev && uv pip install --no-deps -e .
COPY samples ./samples
ENV WHITEOUT_JOBS=/tmp/jobs WHITEOUT_CACHE=/tmp/cache PORT=8080
CMD ["sh", "-c", "uv run --no-sync uvicorn whiteout.server:app --host 0.0.0.0 --port ${PORT}"]
