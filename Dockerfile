FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_INDEX_PATH=/app/deployment/northstar-index.json \
    EVALUATION_REPORT_PATH=/app/deployment/evaluation-report.json

WORKDIR /app

COPY pyproject.toml ./
COPY backend ./backend
COPY evals ./evals
RUN python -m pip install --no-cache-dir .

COPY frontend ./frontend
COPY deployment ./deployment

RUN addgroup --system --gid 10001 app \
    && adduser --system --uid 10001 --ingroup app --home /app app

USER app

CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port \"${PORT:?PORT is required}\" --workers 1"]
