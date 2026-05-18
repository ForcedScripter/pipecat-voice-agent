import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pipecat.pipeline.runner import PipelineRunner
from config import CEREBRAS_API_KEY, SARVAM_API_KEY
from pipeline import create_pipeline
from middleware.rate_limiter import WebSocketRateLimiter
from middleware.security import WebSocketSecurity

# ─── Configuration from environment ──────────────────────

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

ALLOWED_WS_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_WS_ORIGINS", "*").split(",")
    if o.strip()
]

MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", "10"))
SESSION_TIMEOUT_SECS = int(os.getenv("SESSION_TIMEOUT_SECS", "600"))
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"

# ─── Middleware instances ─────────────────────────────────

rate_limiter = WebSocketRateLimiter(
    max_connections_per_minute=5,
    max_concurrent_per_ip=MAX_CONCURRENT_SESSIONS,
)

ws_security = WebSocketSecurity(
    allowed_origins=ALLOWED_WS_ORIGINS,
    max_message_bytes=10240,  # 10KB — enough for 5120 PCM16 samples
    session_timeout_secs=SESSION_TIMEOUT_SECS,
    require_auth=REQUIRE_AUTH,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Voice Agent server starting up...")
    logger.info(
        "Security config | cors={} ws_origins={} max_sessions={} timeout={}s auth={}",
        CORS_ORIGINS,
        ALLOWED_WS_ORIGINS,
        MAX_CONCURRENT_SESSIONS,
        SESSION_TIMEOUT_SECS,
        REQUIRE_AUTH,
    )
    yield
    logger.info("Voice Agent server shutting down.")


app = FastAPI(title="Live Voice Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "ministros-voice-agent",
        "version": "1.1.0",
        **rate_limiter.stats,
    }


@app.get("/ready")
async def ready():
    missing = []
    if not CEREBRAS_API_KEY:
        missing.append("CEREBRAS_API_KEY")
    if not SARVAM_API_KEY:
        missing.append("SARVAM_API_KEY")
    if missing:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "missing": missing},
        )
    return {"status": "ready"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = str(uuid.uuid4())
    client_ip = websocket.client.host if websocket.client else "unknown"

    # ── Security Gate 1: Origin validation ────────────────
    if not ws_security.validate_origin(websocket):
        logger.warning(
            "Rejected connection: bad origin | session_id={} ip={}",
            session_id,
            client_ip,
        )
        await websocket.close(code=4403)
        return

    # ── Security Gate 2: Optional token auth ──────────────
    if not ws_security.validate_token(websocket):
        logger.warning(
            "Rejected connection: auth failed | session_id={} ip={}",
            session_id,
            client_ip,
        )
        await websocket.close(code=4401)
        return

    # ── Security Gate 3: Rate limiting ────────────────────
    if not await rate_limiter.check_connection(client_ip, session_id):
        logger.warning(
            "Rejected connection: rate limited | session_id={} ip={}",
            session_id,
            client_ip,
        )
        await websocket.close(code=4429)
        return

    # ── All gates passed — accept ─────────────────────────
    await websocket.accept()
    session_start = time.monotonic()
    language = websocket.query_params.get("lang", "hi-IN")
    voice = websocket.query_params.get("voice", "shubh")

    logger.info(
        "Client connected | session_id={} ip={} lang={} voice={}",
        session_id,
        client_ip,
        language,
        voice,
    )

    try:
        transport, task = await create_pipeline(
            websocket,
            language=language,
            voice=voice,
            session_id=session_id,
        )

        @transport.event_handler("on_client_connected")
        async def on_connected(t, ws):
            logger.info("Pipeline running | session_id={}", session_id)

        @transport.event_handler("on_client_disconnected")
        async def on_disconnected(t, ws):
            logger.info(
                "Client disconnected — stopping pipeline | session_id={}", session_id
            )
            await task.cancel()

        runner = PipelineRunner()
        await runner.run(task)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected cleanly | session_id={}", session_id)
    except Exception as e:
        error_msg = str(e)
        logger.error(
            "Pipeline error | session_id={} err={}",
            session_id,
            e,
            exc_info=True,
        )
        try:
            # Send error to frontend before closing so the UI can show a message
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"message": error_msg},
            }))
            await websocket.close()
        except Exception:
            pass
    finally:
        # Always release the rate-limiter session
        elapsed = time.monotonic() - session_start
        await rate_limiter.release_session(client_ip, session_id)
        logger.info(
            "Session ended | session_id={} duration={:.1f}s",
            session_id,
            elapsed,
        )


if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info",
    )
