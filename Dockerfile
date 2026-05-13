# ──────────────────────────────────────────────────────────
# Ministros Voice Agent — Production Dockerfile
# Single-stage build — installs into system Python directly
# ──────────────────────────────────────────────────────────

FROM python:3.12-slim

# Install system deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency specification
COPY voice-agent/pyproject.toml voice-agent/uv.lock voice-agent/.python-version ./voice-agent/

# Export requirements and install into system Python
RUN uv export --project voice-agent --frozen --no-dev --no-hashes > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Copy server source code
COPY server/ /app/server/
COPY .env.example /app/.env.example

ENV PYTHONUNBUFFERED=1

# Server listens on this port
EXPOSE 8805

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8805/health')" || exit 1

# Start the voice agent server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8805", "--app-dir", "/app/server"]
