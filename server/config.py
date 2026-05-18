import os
from pathlib import Path
from dotenv import load_dotenv

# .env lives at the project root (one level above server/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8805))

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1-8b")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

# Latency tuning (see README latency section)
VAD_START_SECS = float(os.getenv("VAD_START_SECS", "0.12"))
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.15"))
VAD_MIN_VOLUME = float(os.getenv("VAD_MIN_VOLUME", "0.15"))
USER_SPEECH_TIMEOUT = float(os.getenv("USER_SPEECH_TIMEOUT", "0.35"))
USE_SMART_TURN = os.getenv("USE_SMART_TURN", "false").lower() == "true"
TTS_ENABLE_PREPROCESSING = os.getenv("TTS_ENABLE_PREPROCESSING", "false").lower() == "true"
TTS_PACE = float(os.getenv("TTS_PACE", "1.15"))
