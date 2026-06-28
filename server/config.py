import os
from pathlib import Path
from dotenv import load_dotenv

# .env lives at the project root (one level above server/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
RINGG_API_KEY = os.getenv("RINGG_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8845))

LLM_MODEL = "gpt-oss-120b"

SAMPLE_RATE = 16000
