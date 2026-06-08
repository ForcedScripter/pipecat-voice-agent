import os
from pathlib import Path

from dotenv import load_dotenv
from pipecat.transcriptions.language import Language

# .env lives at repo root (one level above server/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8845))

LLM_MODEL = "gpt-oss-120b"

SAMPLE_RATE = 16000

# RAG — local Qdrant index (built with: uv run python -m ingestion.indexer)
_VOICE_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    str(_VOICE_ROOT / "qdrant_storage"),
)
RAG_COLLECTION_NAME = "restaurant_dynamic_rag"
RAG_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RAG_TOP_K = 4
RAG_SCORE_THRESHOLD = 0.25
RAG_MAX_CONTEXT_CHARS = 1000
LLM_MAX_COMPLETION_TOKENS = 320

# Cartesia TTS voices (slug → voice_id)
CARTESIA_VOICES: dict[str, str] = {
    "netra": "faf0731e-dfb9-4cfc-8119-259a79b27e12",
    "parvaty": "bec003e2-3cb3-429c-8468-206a393c67ad",
    "rohan": "4877b818-c7fe-4c89-b1cf-eadf8e23da72",
    "sneha": "6b02ffe5-e3cb-48c0-a023-c72f85953375",
    "vishal": "098fb15d-2597-4186-8b74-25340050b6e7",
    "sagar": "6303e5fb-a0a7-48f9-bb1a-dd42c216dc5d",
}
DEFAULT_CARTESIA_VOICE = "netra"

# BCP-47 style codes sent from frontend ?lang=
LANGUAGE_OPTIONS: dict[str, dict] = {
    "en-IN": {
        "label": "English",
        "pipecat": Language.EN_IN,
        "cartesia": "en",
    },
    "hi-IN": {
        "label": "Hindi",
        "pipecat": Language.HI_IN,
        "cartesia": "hi",
    },
    "ta-IN": {
        "label": "Tamil",
        "pipecat": Language.TA_IN,
        "cartesia": "ta",
    },
    "te-IN": {
        "label": "Telugu",
        "pipecat": Language.TE_IN,
        "cartesia": "te",
    },
    "ml-IN": {
        "label": "Malayalam",
        "pipecat": Language.ML_IN,
        "cartesia": "ml",
    },
    "kn-IN": {
        "label": "Kannada",
        "pipecat": Language.KN_IN,
        "cartesia": "kn",
    },
    "or-IN": {
        "label": "Odia",
        "pipecat": Language.OR_IN,
        "cartesia": "hi",  # Cartesia has no Odia code; closest TTS fallback
    },
    "pa-IN": {
        "label": "Punjabi",
        "pipecat": Language.PA_IN,
        "cartesia": "pa",
    },
}
DEFAULT_LANGUAGE = "en-IN"

# Fixed policy — overrides RAG/documents for opening-hours questions
OPENING_HOURS_FACT = (
    "Restaurant Grand Chennai is open twenty-four hours a day, seven days a week, every day."
)
OPENING_HOURS_SPOKEN = (
    "We're open twenty-four seven, every single day of the week."
)


def resolve_language(lang: str | None) -> dict:
    """Return language config for STT/TTS/prompts."""
    if not lang:
        return LANGUAGE_OPTIONS[DEFAULT_LANGUAGE]
    key = lang.strip()
    if key in LANGUAGE_OPTIONS:
        return {"code": key, **LANGUAGE_OPTIONS[key]}
    return {"code": DEFAULT_LANGUAGE, **LANGUAGE_OPTIONS[DEFAULT_LANGUAGE]}


def resolve_voice_id(voice: str | None) -> str:
    """Map voice slug or raw UUID to a Cartesia voice_id."""
    if not voice or not voice.strip():
        return CARTESIA_VOICES[DEFAULT_CARTESIA_VOICE]

    key = voice.strip().lower()
    if key in CARTESIA_VOICES:
        return CARTESIA_VOICES[key]

    raw = voice.strip()
    if len(raw) == 36 and raw.count("-") == 4:
        return raw

    return CARTESIA_VOICES[DEFAULT_CARTESIA_VOICE]
