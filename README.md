# Aiden — Restaurant Grand Chennai voice agent

Voice + RAG: microphone → STT → Qdrant → LLM → TTS → speaker.

## What you need

| Item | Required? | Why |
|------|-----------|-----|
| **`.venv/`** | Auto-created locally | Not in git. Run `uv sync` once; you don't manage it by hand. |
| **`docs/`** | Only to **build or update** the knowledge index | Runtime uses **`qdrant_storage/`**, not docx. Keep `docs/` if menu/policies change. |
| **`qdrant_storage/`** | Yes for RAG | Built by ingest (gitignored). |
| **`.env`** | Yes | Copy from `.env.example` and add API keys. |

## Layout

```text
├── frontend/          # Reference client (hand off to frontend team)
├── server/            # FastAPI + WebSocket API
├── ingestion/         # Index docs/ → qdrant_storage/
├── docs/              # Source .docx (ingest only)
├── pyproject.toml
└── qdrant_storage/    # Created by ingest
```

## Setup and run

```powershell
copy .env.example .env
# Set CEREBRAS_API_KEY and SARVAM_API_KEY
uv sync
uv run python -m ingestion.indexer
uv run uvicorn main:app --host 0.0.0.0 --port 8854 --app-dir server
```

Demo UI (second terminal):

```powershell
cd frontend
python -m http.server 3000
```

Set `frontend/config.js` → `wsBaseUrl` (e.g. `ws://localhost:8854/ws`).

---

## Frontend integration

Reference: `frontend/` (`config.js`, `agent.js`, `audio-processor.js`, `index.html`).

### HTTP

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | `{ "status": "ok" }` |
| `GET` | `/ready` | `503` if API keys missing |
| `WS` | `/ws?lang=en-IN` | Voice session (`hi-IN` also supported) |

### WebSocket

**Connect:** `ws://<host>:<port>/ws?lang=en-IN`

**Client → server:** binary PCM **16-bit LE**, mono, **16 kHz**; JSON `{ "type": "interrupt" }` for barge-in.

**Server → client:** binary PCM audio; JSON RTVI events (`bot-started-speaking`, `bot-stopped-speaking`, etc.).

### Embed `VoiceAgent`

```html
<script src="config.js"></script>
<script src="agent.js"></script>
```

```javascript
const agent = new VoiceAgent(window.VOICE_AGENT_CONFIG.wsBaseUrl);
await agent.connect("en-IN");
```

Host `audio-processor.js` on the same origin as your page.

### Production

- Use `wss://` behind TLS
- Restrict CORS in `server/main.py`
- Do not commit `.env` or `qdrant_storage/`
