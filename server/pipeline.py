from loguru import logger

# -- Turn strategies (low-latency path uses VAD + speech timeout, not Smart Turn)
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

# -- VAD
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

# -- Pipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask

# -- Context aggregators
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    LLMAssistantAggregatorParams,
)

# -- RTVI
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver

# -- Services
from pipecat.services.cerebras.llm import CerebrasLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService

# -- Transport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)

# -- Language
from pipecat.transcriptions.language import Language

# -- Config and custom processors
from config import (
    CEREBRAS_API_KEY,
    SARVAM_API_KEY,
    LLM_MODEL,
    SAMPLE_RATE,
    VAD_START_SECS,
    VAD_STOP_SECS,
    VAD_MIN_VOLUME,
    USER_SPEECH_TIMEOUT,
    USE_SMART_TURN,
    TTS_ENABLE_PREPROCESSING,
    TTS_PACE,
)
from serializers.raw_pcm import RawPCMSerializer
from processors.pivot_detector import PivotDetectorProcessor
from processors.naturalizer import ResponseNaturalizerProcessor

SYSTEM_PROMPT = """You are a warm, efficient voice support assistant.

Speak like a calm, capable human support agent -- clear, conversational, and helpful.

Rules:
- Keep responses to one short sentence unless more detail is needed
- Ask only ONE question at a time
- Support English, Hindi, and Hinglish naturally
- Match the user's tone and language style
- Use contractions naturally (I'm, you're, that's, let's)
- Speak like audio, not written text
- If interrupted, stop and continue from new context
- If uncertain, say so briefly and guide forward

Never:
- Say "As an AI" or "I'm just a language model"
- Use markdown, bullet points, emojis, or formal formatting
- Spell words letter-by-letter unless asked
- Read punctuation or symbols aloud
- Use filler like "Certainly" or "I'd be happy to help"
- Guess facts, policies, or outcomes

Goal: Help users quickly, clearly, and naturally in realtime voice.
"""


def _build_turn_strategies():
    """Prefer VAD + speech-timeout over Smart Turn for lower end-of-utterance latency."""
    if USE_SMART_TURN:
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
            LocalSmartTurnAnalyzerV3,
        )

        turn_stop = TurnAnalyzerUserTurnStopStrategy(
            turn_analyzer=LocalSmartTurnAnalyzerV3()
        )
        logger.info("Turn stop: Smart Turn v3 (higher quality, higher latency)")
    else:
        turn_stop = SpeechTimeoutUserTurnStopStrategy(
            user_speech_timeout=USER_SPEECH_TIMEOUT,
        )
        logger.info(
            "Turn stop: speech timeout ({:.2f}s after VAD stop)",
            USER_SPEECH_TIMEOUT,
        )

    # VAD starts the user turn immediately; transcription is a soft-speech fallback.
    turn_start = [
        VADUserTurnStartStrategy(),
        TranscriptionUserTurnStartStrategy(use_interim=False),
    ]
    return UserTurnStrategies(start=turn_start, stop=[turn_stop])


async def create_pipeline(
    websocket,
    language: str = "hi-IN",
    voice: str = "shubh",
    session_id: str | None = None,
):
    sid = session_id or "-"
    logger.info("Creating pipeline | session_id={} language={} voice={}", sid, language, voice)

    # -- Transport
    # Silero VAD handles endpointing locally. Sarvam STT must NOT run its own VAD
    # (vad_signals=False) or you get double-endpointing and hundreds of ms extra delay.
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
            audio_in_stream_on_start=True,
            audio_in_passthrough=True,
            serializer=RawPCMSerializer(),
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    start_secs=VAD_START_SECS,
                    stop_secs=VAD_STOP_SECS,
                    min_volume=VAD_MIN_VOLUME,
                )
            ),
        ),
    )
    logger.info(
        "Transport created | session_id={} vad start={:.2f}s stop={:.2f}s",
        sid,
        VAD_START_SECS,
        VAD_STOP_SECS,
    )

    turn_strategies = _build_turn_strategies()

    # -- STT
    stt = SarvamSTTService(
        api_key=SARVAM_API_KEY,
        mode="transcribe",
        sample_rate=SAMPLE_RATE,
        settings=SarvamSTTService.Settings(
            model="saaras:v3",
            language=Language.HI_IN if language == "hi-IN" else Language.EN_IN,
            vad_signals=False,
            high_vad_sensitivity=False,
        ),
    )
    logger.info("STT service created | session_id={} vad_signals=False", sid)

    # -- LLM
    llm = CerebrasLLMService(
        api_key=CEREBRAS_API_KEY,
        settings=CerebrasLLMService.Settings(
            model=LLM_MODEL,
            temperature=0.6,
            max_completion_tokens=150,
        ),
    )
    logger.info("LLM service created | session_id={}", sid)

    # -- TTS
    tts = SarvamTTSService(
        api_key=SARVAM_API_KEY,
        sample_rate=SAMPLE_RATE,
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            voice=voice,
            language=Language.HI_IN if language == "hi-IN" else Language.EN_IN,
            pace=TTS_PACE,
            pitch=0.0,
            enable_preprocessing=TTS_ENABLE_PREPROCESSING,
            temperature=0.7,
        ),
    )
    logger.info(
        "TTS service created | session_id={} preprocessing={}",
        sid,
        TTS_ENABLE_PREPROCESSING,
    )

    # -- Custom processors
    pivot_detector = PivotDetectorProcessor()
    naturalizer = ResponseNaturalizerProcessor(add_starters=True)

    # -- RTVI: no transport= so it does NOT gate audio behind client-ready handshake
    rtvi = RTVIProcessor()

    # -- LLM context
    lang_name = "Hindi" if language == "hi-IN" else "English"
    dynamic_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"IMPORTANT: The user has selected {lang_name}. "
        f"You MUST respond entirely in {lang_name}."
    )
    context = LLMContext(messages=[{"role": "system", "content": dynamic_prompt}])

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=turn_strategies,
        ),
        assistant_params=LLMAssistantAggregatorParams(),
    )
    logger.info("Context aggregators created | session_id={}", sid)

    # -- Pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            pivot_detector,
            llm,
            naturalizer,
            tts,
            rtvi,
            assistant_aggregator,
            transport.output(),
        ]
    )
    logger.info("Pipeline assembled | session_id={}", sid)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )
    logger.info("Pipeline task created -- ready | session_id={}", sid)

    return transport, task
