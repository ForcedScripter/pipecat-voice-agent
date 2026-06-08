import asyncio
import uuid
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import LLMMessagesAppendFrame
from config import (
    CARTESIA_API_KEY,
    CARTESIA_VOICES,
    CEREBRAS_API_KEY,
    DEFAULT_CARTESIA_VOICE,
    DEFAULT_LANGUAGE,
    LANGUAGE_OPTIONS,
    SARVAM_API_KEY,
)
from pipeline import create_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Voice Agent server starting up...")
    yield
    logger.info("Voice Agent server shutting down.")


app = FastAPI(title="Live Voice Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/languages")
async def list_languages():
    return {
        "default": DEFAULT_LANGUAGE,
        "languages": [
            {"code": code, "label": cfg["label"]}
            for code, cfg in LANGUAGE_OPTIONS.items()
        ],
    }


@app.get("/voices")
async def list_voices():
    return {
        "default": DEFAULT_CARTESIA_VOICE,
        "voices": [
            {"id": slug, "name": slug.capitalize(), "voice_id": vid}
            for slug, vid in CARTESIA_VOICES.items()
        ],
    }


@app.get("/ready")
async def ready():
    missing = []
    if not CEREBRAS_API_KEY:
        missing.append("CEREBRAS_API_KEY")
    if not SARVAM_API_KEY:
        missing.append("SARVAM_API_KEY")
    if not CARTESIA_API_KEY:
        missing.append("CARTESIA_API_KEY")
    if missing:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "missing": missing},
        )
    return {"status": "ready"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = str(uuid.uuid4())

    await websocket.accept()
    logger.info(
        "Client connected | session_id={} client={}",
        session_id,
        websocket.client,
    )

    # ?lang=en-IN|hi-IN|ta-IN|...  &  ?voice=netra|parvaty|...
    language = websocket.query_params.get("lang", DEFAULT_LANGUAGE)
    voice = websocket.query_params.get("voice", DEFAULT_CARTESIA_VOICE)

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
            # Trigger Aiden's greeting immediately
            # One-shot nudge; context sanitizer removes this after the customer speaks
            await task.queue_frames([LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": "[call connected — greet briefly as Aiden, one sentence]"}]
            )])

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
        logger.error(
            "Pipeline error | session_id={} err={}",
            session_id,
            e,
            exc_info=True,
        )
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
