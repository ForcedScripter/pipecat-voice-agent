"""Smoke test: pipeline imports and turn strategies build without starting a server."""

import asyncio
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, str(SERVER_DIR))


async def main() -> None:
    from pipeline import _build_turn_strategies
    from config import (
        VAD_STOP_SECS,
        USER_SPEECH_TIMEOUT,
        USE_SMART_TURN,
        TTS_ENABLE_PREPROCESSING,
    )

    strategies = _build_turn_strategies()
    assert strategies.stop, "expected at least one stop strategy"
    assert strategies.start, "expected at least one start strategy"

    print("pipeline import: OK")
    print(f"  VAD stop_secs={VAD_STOP_SECS}")
    print(f"  user_speech_timeout={USER_SPEECH_TIMEOUT}")
    print(f"  use_smart_turn={USE_SMART_TURN}")
    print(f"  tts_preprocessing={TTS_ENABLE_PREPROCESSING}")
    print(f"  stop={[type(s).__name__ for s in strategies.stop]}")
    print(f"  start={[type(s).__name__ for s in strategies.start]}")


if __name__ == "__main__":
    asyncio.run(main())
