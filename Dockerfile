# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRAUD_DATA_DIR=/app/data \
    FRAUD_MODEL_DIR=/app/models

WORKDIR /app

# libgomp1 : requis par XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/

EXPOSE 8001 8002

# Service par défaut : l'API de scoring (le modèle est monté depuis ./models).
CMD ["python", "src/API_Score.py"]
