import asyncio
import uuid
import io
import docx
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pathlib import Path
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import LLMMessagesAppendFrame
from config import CEREBRAS_API_KEY, SARVAM_API_KEY
from pipeline import create_pipeline
from rag_service import RAGService

# Qdrant Database Path (stores temporary collection data)
_ROOT_DIR = Path(__file__).resolve().parent.parent
QDRANT_PATH = str(_ROOT_DIR / "qdrant_db")

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


def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract plain text from uploaded txt or docx file."""
    if filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_content))
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    else:
        # assume plain text
        return file_content.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 600) -> list[str]:
    """Simple text chunker by paragraph and word count."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            chunks.append(p)
        else:
            words = p.split()
            current_chunk = []
            current_len = 0
            for w in words:
                current_chunk.append(w)
                current_len += len(w) + 1
                if current_len >= chunk_size:
                    chunks.append(" ".join(current_chunk))
                    # implement overlap of last 15 words
                    overlap_words = current_chunk[-15:]
                    current_chunk = list(overlap_words)
                    current_len = sum(len(x) + 1 for x in current_chunk)
            if current_chunk and current_len > len(" ".join(current_chunk[-15:])):
                chunks.append(" ".join(current_chunk))
    return [c for c in chunks if len(c.strip()) > 10]


@app.post("/upload")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    if not session_id or not session_id.strip():
        return JSONResponse(status_code=400, content={"error": "session_id is required"})
    
    # Sanitize session_id to strip any query parameter suffix
    session_id = session_id.strip()
    if "?" in session_id:
        session_id = session_id.split("?")[0]
    if "&" in session_id:
        session_id = session_id.split("&")[0]

    try:
        content = await file.read()
        text = extract_text_from_file(content, file.filename)
        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "Document contains no readable text"})
        
        chunks = chunk_text(text)
        if not chunks:
            return JSONResponse(status_code=400, content={"error": "Document content is too short to index"})

        collection_name = f"session_{session_id}"
        
        # Index document chunks using temporary RAGService
        rag = RAGService(qdrant_path=QDRANT_PATH, collection_name=collection_name)
        await rag.index_documents(chunks)
        rag.close()
        
        logger.info("Successfully indexed {} chunks for session {}", len(chunks), session_id)
        return {"status": "ok", "chunks": len(chunks)}
        
    except Exception as e:
        logger.error("Failed to process document upload: {}", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to upload document: {str(e)}"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = websocket.query_params.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    else:
        # Sanitize session_id to strip any query parameter suffix
        session_id = session_id.strip()
        if "?" in session_id:
            session_id = session_id.split("?")[0]
        if "&" in session_id:
            session_id = session_id.split("&")[0]

    await websocket.accept()
    logger.info(
        "Client connected | session_id={} client={}",
        session_id,
        websocket.client,
    )

    # Accept ?lang=hi-IN or ?lang=en-IN
    language = websocket.query_params.get("lang", "hi-IN")
    
    # Initialize dynamic RAGService for this session
    collection_name = f"session_{session_id}"
    rag_service = RAGService(qdrant_path=QDRANT_PATH, collection_name=collection_name)


    try:
        transport, task = await create_pipeline(
            websocket, 
            language=language, 
            session_id=session_id,
            rag_service=rag_service,
        )

        @transport.event_handler("on_client_connected")
        async def on_connected(t, ws):
            logger.info("Pipeline running | session_id={}", session_id)
            # Trigger Louie's greeting immediately with dynamic instructions
            await task.queue_frames([LLMMessagesAppendFrame(
                messages=[{"role": "system", "content": "The user just connected. Greet them warmly in one sentence, and let them know you are ready to answer questions based on the uploaded document, if any. Keep it natural."}]
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
    finally:
        # Guarantee session vector data cleanup
        logger.info("Cleaning up temporary RAG session data for session_id={}", session_id)
        rag_service.delete_collection()
        rag_service.close()


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
