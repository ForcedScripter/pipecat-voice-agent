from loguru import logger

# -- Smart Turn
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.turns.user_stop import (
    TurnAnalyzerUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_start import (
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

# -- VAD
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor

# -- Pipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import LLMMessagesAppendFrame

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
# from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat_ringg import RinggSTTService, RinggSTTParams
from pipecat.services.cartesia.tts import CartesiaTTSService

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
    RINGG_API_KEY,
    CARTESIA_API_KEY,
    LLM_MODEL,
    SAMPLE_RATE,
)
from serializers.raw_pcm import RawPCMSerializer
from processors.pivot_detector import PivotDetectorProcessor
from processors.naturalizer import ResponseNaturalizerProcessor
from processors.context_sanitizer import ContextSanitizerProcessor
from processors.llm_empty_guard import LLMEmptyGuardProcessor

from processors.audio_gate import AudioGateProcessor
from processors.client_interrupt import ClientInterruptProcessor
from processors.turn_reset import TurnResetProcessor
from processors.turn_logger import TurnLifecycleProcessor
from processors.rag_injector import RAGContextInjectorProcessor
from rag_service import RAGService


def get_system_prompt(language: str) -> str:
    lang_name = "Hindi" if language == "hi-IN" else "English"
    return f"""You are a warm, efficient, and natural-sounding voice support assistant. You speak {lang_name}.

Speak like a calm, capable human support agent -- clear, conversational, and helpful.

Behavior:
- Keep responses short and easy to understand
- Default to one short spoken sentence unless more detail is required
- Ask only ONE question at a time
- React naturally, but stay professional and composed
- Match the user's tone and language style naturally
- Support English, Hindi, and Hinglish naturally
- If the user changes topic, adapt smoothly and continue naturally
- If interrupted, stop naturally and continue from the new context
- Briefly acknowledge frustrations or confusion when appropriate

Accuracy:
- NEVER guess facts, policies, balances, actions, or outcomes
- If information is missing or unclear, ask a short clarifying question
- If you are uncertain, say so briefly and guide the user forward
- Avoid overexplaining unless the user asks for details

Voice Style:
- Use contractions naturally (I'm, you're, that's, let's)
- Speak like audio, not written text
- Prefer natural spoken phrasing over perfectly written grammar
- Keep replies concise to reduce response latency in conversation
- Avoid long explanations and unnecessary filler
- Avoid repetitive phrases like "Certainly" or "I'd be happy to help"
- NEVER say "As an AI" or "I'm just a language model"
- NEVER use markdown, bullet points, emojis, or formal formatting
- NEVER sound robotic, scripted, or overly corporate
- NEVER spell words, names, codes, or sentences letter-by-letter unless the user explicitly asks
- NEVER read punctuation, symbols, markdown, URLs, or formatting aloud
- NEVER say "dot" while speaking naturally unless the user explicitly asks for an email, URL, or spelling

Conversation Style:
- Focus on helping the user quickly and naturally
- Keep the conversation flowing smoothly in realtime
- Prioritize clarity over perfect grammar
- Sound confident, calm, and human
- Avoid unnecessary apologies or excessive politeness

Short Inputs and Closers:
- ALWAYS respond to short or one-word inputs like "okay", "hmm", "yeah", "alright", "sure", "fine", "thanks", "thank you", "bye", "goodbye", "no", "nope", "what else" and similar words
- NEVER return an empty response — always say something, even if it's just a brief acknowledgement
- For "thank you" or "thanks": respond warmly and ask if there's anything else
- For "bye" or "goodbye": say a brief, friendly goodbye
- For "okay", "hmm", "alright", "sure": briefly check if they need anything else or acknowledge naturally
- For vague or unclear one-word inputs: ask a short clarifying question
- Keep these responses to one short sentence

Frustrated or Rude Users:
- If the user is frustrated, rude, or uses harsh language, stay calm and professional
- NEVER refuse to respond or go silent — always say something brief and helpful
- Acknowledge their frustration briefly, then redirect to how you can help
- Example responses: "I understand, let me know if there's anything I can help with" or "No worries, I'm here whenever you need"
- If the user says "shut up", "go away", "stop talking", etc., briefly acknowledge and offer to stay available
- NEVER lecture the user about their tone or language
- NEVER apologize excessively — one brief acknowledgment is enough

Your goal is to help users quickly, clearly, and naturally and professional in realtime voice conversation."""


async def create_pipeline(
    websocket,
    language: str = "en-IN",
    session_id: str | None = None,
    rag_service: RAGService | None = None,
):
    sid = session_id or "-"
    logger.info("Creating pipeline | session_id={} language={}", sid, language)

    # -- Transport
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
        ),
    )
    logger.info("Transport created | session_id={}", sid)

    # -- Smart Turn
    smart_turn_stop = TurnAnalyzerUserTurnStopStrategy(
        turn_analyzer=LocalSmartTurnAnalyzerV3(),
    )
    speech_timeout_stop = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=1.2)
    vad_turn_start = VADUserTurnStartStrategy()
    logger.info("Turn strategies created | session_id={}", sid)

    # Silero VAD on the audio stream — required for barge-in (transport vad_analyzer is unused in 1.1)
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.7,
                start_secs=0.2,
                stop_secs=0.4,
                min_volume=0.6,
            )
        ),
    )
    client_interrupt = ClientInterruptProcessor()

    # -- STT (keeping Sarvam STT — best for Indian languages)
    # stt = SarvamSTTService(
    #     api_key=SARVAM_API_KEY,
    #     mode="transcribe",
    #     sample_rate=SAMPLE_RATE,
    #     settings=SarvamSTTService.Settings(
    #         model="saaras:v3",
    #         language=Language.HI_IN if language == "hi-IN" else Language.EN_IN,
    #         vad_signals=False,
    #         high_vad_sensitivity=False,
    #         positive_speech_threshold=0.4,
    #         negative_speech_threshold=0.3,
    #         start_speech_volume_threshold=-45,
    #     ),
    # )

    # -- STT (Ringg AI Integration)
    ringg_lang = "hi" if language == "hi-IN" else "en"
    stt = RinggSTTService(
        base_url=None,
        params=RinggSTTParams(
            api_key=RINGG_API_KEY,
            language=ringg_lang,
            mode="stream",
            enable_cap_punc=True
        )
    )
    logger.info("STT service created | session_id={}", sid)

    # -- LLM
    llm = CerebrasLLMService(
        api_key=CEREBRAS_API_KEY,
        settings=CerebrasLLMService.Settings(
            model=LLM_MODEL,
            temperature=0.7,
            max_completion_tokens=1000,
        ),
    )
    logger.info("LLM service created | session_id={}", sid)

    # -- TTS (Cartesia Sonic-3 — ~40ms TTFB vs Sarvam's ~300ms)
    tts = CartesiaTTSService(
        api_key=CARTESIA_API_KEY,
        voice_id="faf0731e-dfb9-4cfc-8119-259a79b27e12",
        model="sonic-3",
        language="hi" if language == "hi-IN" else "en",
        sample_rate=SAMPLE_RATE,
    )
    logger.info("TTS service created | session_id={}", sid)

    # -- Custom processors
    pivot_detector = PivotDetectorProcessor()
    naturalizer = ResponseNaturalizerProcessor(add_starters=True, language=language)
    llm_empty_guard = LLMEmptyGuardProcessor()

    # -- RAG context injector
    rag_injector = RAGContextInjectorProcessor(
        rag_service=rag_service,
        max_context_chars=1500,
    )

    audio_gate = AudioGateProcessor(barge_in_rms=0.04, decay_secs=0.35)
    logger.info("Custom processors created | session_id={}", sid)

    # -- RTVI
    rtvi = RTVIProcessor()
    logger.info("RTVI processor created | session_id={}", sid)

    # -- LLM context
    context = LLMContext(
        messages=[{"role": "system", "content": get_system_prompt(language)}]
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[vad_turn_start],
                stop=[smart_turn_stop, speech_timeout_stop],
            ),
            user_turn_stop_timeout=3.0,
        ),
        assistant_params=LLMAssistantAggregatorParams(),
    )
    logger.info("Context aggregators created | session_id={}", sid)

    context_sanitizer = ContextSanitizerProcessor(context=context)
    turn_reset = TurnResetProcessor(context=context)
    turn_logger = TurnLifecycleProcessor()

    # -- Pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # raw audio from browser
            client_interrupt,
            # browser {type: interrupt} -> InterruptionFrame
            audio_gate,  # drop echo while bot speaks, allow loud barge-in
            # denoise — clean audio before VAD sees it
            vad,  # VAD -> user turn start + pipeline interruption
            turn_reset,  # drop truncated assistant from context on interrupt
            stt,  # audio -> TranscriptionFrame
            user_aggregator,  # turn detection + LLM context
            turn_logger,  # [Turn] lifecycle logs
            context_sanitizer,  # sanitize + trim before LLM
            pivot_detector,  # topic-change detection
            rag_injector,  # inject user-uploaded context chunks
            llm,  # text -> streaming response
            naturalizer,  # clean robotic phrases
            llm_empty_guard,  # inject fallback if LLM produced nothing
            tts,  # text -> audio chunks (Cartesia ~40ms TTFB)
            rtvi,  # RTVI events to browser
            assistant_aggregator,  # store reply in context
            transport.output(),  # stream audio to browser
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
