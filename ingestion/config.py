"""Settings for building the Qdrant index (ingest only)."""

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

_VOICE_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_VOICE_ROOT / ".env")

QDRANT_PATH = os.getenv("QDRANT_PATH", str(_VOICE_ROOT / "qdrant_storage"))
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "restaurant_dynamic_rag")
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

EMBED_MODEL = HuggingFaceEmbedding(
    model_name=EMBED_MODEL_NAME,
    embed_batch_size=16,
)
