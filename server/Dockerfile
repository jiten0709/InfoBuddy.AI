# Multi-stage build for FastAPI server
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (curl for healthcheck/debug)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy app
COPY . .

EXPOSE 8000

# Default to uvicorn; adjust "app:app" if your FastAPI instance name differs
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]