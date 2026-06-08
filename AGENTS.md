# Project Context — Ministros Voice Agent

## Overview
Real-time browser-to-server voice assistant using Pipecat AI framework.

## Architecture
- **Client**: Browser-based mic capture + audio playback (`client/`)
- **Server**: FastAPI + Pipecat backend (`server/`)
  - STT: Sarvam AI (`saaras:v3`)
  - LLM: Cerebras (`llama3.1-8b`)
  - TTS: Sarvam AI (`bulbul:v3`, voice: `shubh`)
- **Transport**: WebSocket with raw PCM audio (16kHz)
- **Dependencies**: Managed via `uv` in `voice-agent/pyproject.toml`

## Key Files
- `server/main.py` — FastAPI app, WebSocket endpoint `/ws`, security gates
- `server/pipeline.py` — Pipecat pipeline: transport → STT → LLM → TTS → output
- `server/config.py` — Env loading (.env at project root)
- `server/processors/naturalizer.py` — Post-LLM text cleanup
- `server/processors/pivot_detector.py` — Topic change detection
- `server/serializers/raw_pcm.py` — PCM audio serializer
- `server/middleware/rate_limiter.py` — IP-based rate limiting
- `server/middleware/security.py` — Origin validation, message limits

## Key Decisions
- Cerebras chosen over Groq for LLM (faster TTFB)
- `jose` library not used — auth is HMAC-based when enabled
- Raw PCM transport (no WAV headers) for lower latency
- Silero VAD with aggressive 150ms endpointing
- LLM max 150 tokens for concise spoken responses
- TTS pace 1.15x for faster synthesis
- Supports Hindi (hi-IN) and English (en-IN)

## Running
```
# Install deps
uv sync --project voice-agent

# Server
cd voice-agent && uv run uvicorn main:app --host 0.0.0.0 --port 8805 --app-dir ../server

# Client
cd client && python -m http.server 3000
```

## Deployment
Azure Container Apps (Central India region). See `azure-deploy.sh`.

## Environment Variables
- `SARVAM_API_KEY` — STT/TTS
- `CEREBRAS_API_KEY` — LLM
- `GROQ_API_KEY` — fallback LLM (kept for compat)
- `HOST` / `PORT` — server bind (default 0.0.0.0:8805)

## Conventions
- Logging: `loguru`
- Async throughout
- Security middleware is in-memory (no Redis)
- Pipeline params tuned for sub-second voice-to-voice latency (~635ms)
