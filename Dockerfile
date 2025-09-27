# syntax=docker/dockerfile:1
FROM python:3.11-slim

# --- base env ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_WORKERS=2

# --- system deps ---
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# --- app dir ---
WORKDIR /app

# --- python deps layer (better cache) ---
COPY requirements.txt ./ 
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- copy app ---
COPY . .

# --- Spaces sets $PORT dynamically; honor it ---
ARG PORT=7860
ENV PORT=${PORT}
EXPOSE ${PORT}

# Optional: run as non-root (safer)
# RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
# USER appuser

# --- start ---
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
