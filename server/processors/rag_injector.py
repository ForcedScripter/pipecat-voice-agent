"""
RAGContextInjectorProcessor
────────────────────────────
Pipecat FrameProcessor that intercepts LLMContextFrame, retrieves relevant
restaurant context from Qdrant, and injects it as a system message before
the LLM processes the user's query.

Pipeline position: after pivot_detector, before llm.

Key behaviors:
- Intercepts LLMContextFrame (the frame pipecat 1.1.0 sends to the LLM)
- Skips retrieval for short/filler inputs (greetings, "okay", "bye", etc.)
- Runs retrieval async and off the hot path
- Formats retrieved chunks as a [RESTAURANT CONTEXT] system message
"""

import re
import asyncio
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from rag_service import RAGService
from config import OPENING_HOURS_FACT, resolve_language
from processors.topic_guard import looks_like_code_output

RESTAURANT_CONTEXT_MARKER = "[RESTAURANT CONTEXT]"
MAX_INJECT_CHUNKS = 3

# Menu docs are in English — map Indian-language / romanized terms → English retrieval.
_TOPIC_HINTS: list[tuple[str, str]] = [
    # Parking
    ("parking", "parking car bike valet availability"),
    ("park", "parking availability"),
    ("valet", "valet parking"),
    ("பார்க்கிங்", "parking availability"),
    ("वाहन", "parking"),
    ("पार्किंग", "parking availability"),
    ("పార్కింగ్", "parking"),
    ("പാർക്കിംഗ്", "parking"),
    ("ಪಾರ್ಕಿಂಗ್", "parking"),
    ("ପାର୍କିଂ", "parking"),
    ("ਪਾਰਕਿੰਗ", "parking"),
    # Hours
    ("hour", "restaurant opening hours Monday Sunday"),
    ("hours", "restaurant opening hours"),
    ("open", "opening hours restaurant"),
    ("close", "closing hours restaurant"),
    ("timing", "restaurant working hours"),
    ("நேரம்", "opening hours restaurant"),
    ("திற", "opening hours"),
    ("மணி", "opening hours time"),
    ("समय", "opening hours restaurant"),
    ("खुला", "opening hours"),
    ("बंद", "closing hours"),
    ("సమయం", "opening hours"),
    ("സമയം", "opening hours"),
    ("ಸಮಯ", "opening hours"),
    ("ସମୟ", "opening hours"),
    ("ਸਮਾਂ", "opening hours"),
    # Menu / food
    ("menu", "menu dishes food"),
    ("dish", "menu dishes"),
    ("food", "menu dishes"),
    ("biryani", "biryani menu price"),
    ("மெனு", "menu dishes"),
    ("உணவு", "menu food dishes"),
    ("मेनू", "menu dishes"),
    ("खाना", "menu food dishes"),
    ("भोजन", "menu food"),
    ("మెనూ", "menu dishes"),
    ("മെനു", "menu dishes"),
    ("ಮೆನು", "menu dishes"),
    # Price
    ("price", "dish price menu"),
    ("cost", "price menu"),
    ("விலை", "dish price"),
    ("कीमत", "dish price menu"),
    ("दाम", "price menu"),
    ("ధర", "price menu"),
    ("വില", "price menu"),
    ("ಬೆಲೆ", "price menu"),
    # Reservation
    ("reserv", "table reservation booking"),
    ("book", "table booking reservation"),
    ("table", "table reservation seating"),
    ("முன்பதிவு", "table reservation booking"),
    ("बुकिंग", "table reservation"),
    ("रिज़र्व", "table reservation"),
    ("బుకింగ్", "table reservation"),
    ("ബുക്കിംഗ്", "table reservation"),
    # Allergy
    ("allerg", "allergen dairy gluten nuts"),
    ("dairy", "dairy allergen dishes"),
    ("அலர்ஜி", "food allergy allergen"),
    ("एलर्जी", "food allergy allergen"),
    ("अलर्जी", "allergy allergen"),
]

# Chunks from training/ops docs — not facts for callers (clog retrieval).
_META_CHUNK_PATTERNS = [
    r"example\s+customer\s+queries",
    r"example\s+complaint",
    r"example\s+\d",
    r"sample\s+ai\s+responses",
    r"ai\s+assistant\s+(behavioral|should|must|rules|responsibilities)",
    r"system\s+notes\s+for\s+ai",
    r"mandatory\s+allergy\s+safety\s+flow",
    r"the\s+ai\s+assistant\s+should",
    r"the\s+ai\s+assistant\s+must",
    r"complaint\s+(ticket|categories|handling)",
    r"escalation\s+rules",
    r"feedback\s+collection",
    r"personalization\s+features",
    r"privacy\s*&\s*data",
    r"low\s+priority",
    r"medium\s+priority",
    r"high\s+priority",
    r"ai\s+recommendation\s+rules",
    r"ai\s+navigation\s+assistance",
    r"ai\s+complaint\s+handling",
    r"ai\s+behavioral\s+rules",
    r"ai\s+escalation\s+rules",
    r"repeat\s+customer\s+recognition",
    r"welcome\s+back,\s+\w+",
    r"personalized\s+offer",
    r"ai\s+assistant\s+may\s+suggest",
]
_META_CHUNK_RE = re.compile("|".join(_META_CHUNK_PATTERNS), re.IGNORECASE)

# Likely real menu / policy facts.
_FACT_SIGNALS = re.compile(
    r"(price:\s*₹|category:|contains allergens:|description:|type:\s*(vegetarian|non-vegetarian)|"
    r"preparation time:|q\d+\.|opening hours|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d{1,2}\s*(am|pm|:00))",
    re.IGNORECASE,
)

# Common STT mis-hearings → better retrieval query.
_STT_QUERY_FIXES = [
    (re.compile(r"\bopening\s+hands\b", re.I), "opening hours"),
    (re.compile(r"\bwho\s+is\s+the\s+chief\b", re.I), "chef"),
    (re.compile(r"\bdairy\s+free\s+products?\b", re.I), "dairy free dishes without dairy allergen"),
    (re.compile(r"\bhalf\s+footed\b", re.I), "dairy allergy"),
    (re.compile(r"\bgood\s+allergy\b", re.I), "food allergy dairy"),
    (re.compile(r"\blogic\b", re.I), "allergy"),
    (re.compile(r"\bwatch\s+the\s+price\b", re.I), "what is the price"),
    (re.compile(r"\bsuccess\s+baton\b", re.I), "what can you help with"),
]


def _normalize_query(text: str) -> str:
    q = text.strip()
    for pattern, replacement in _STT_QUERY_FIXES:
        q = pattern.sub(replacement, q)
    return q


def _is_meta_chunk(text: str) -> bool:
    """Training scripts and example dialogues — not customer-facing facts."""
    if _META_CHUNK_RE.search(text):
        return True
    # Example dialogue blocks without menu structure
    if "customer:" in text.lower() and "ai response:" in text.lower():
        if not _FACT_SIGNALS.search(text):
            return True
    stripped = text.strip().strip("- ").lower()
    # Title-only chunks with no actual times or facts
    if not re.search(r"\d", text):
        if stripped in (
            "opening hours",
            "restaurant working hours",
            "lunch peak hours",
            "dinner peak hours",
        ):
            return True
        if "hours" in stripped and len(stripped) < 50:
            return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 2 and not _FACT_SIGNALS.search(text):
        if lines and (lines[0].startswith("-") or len(lines[0]) < 80):
            return True
    return False


def _normalize_rag_chunk(text: str) -> str:
    """Flatten day: time lines so the LLM/TTS handle them better."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(
            r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*:\s*(.+)$",
            line,
            re.I,
        )
        if m:
            out.append(f"On {m.group(1)}, {m.group(2).strip()}")
        else:
            out.append(line.lstrip("- ").strip())
    return "\n".join(out)


def _chunk_priority(text: str) -> int:
    """Higher = prefer for voice answers."""
    score = 0
    if _FACT_SIGNALS.search(text):
        score += 10
    if "price:" in text.lower():
        score += 5
    if "contains allergens:" in text.lower():
        score += 3
    if _is_meta_chunk(text):
        score -= 20
    return score


def _filter_and_rank_chunks(chunks: list[str]) -> list[str]:
    usable = [
        c
        for c in chunks
        if c.strip() and not _is_meta_chunk(c) and not looks_like_code_output(c)
    ]
    usable.sort(key=_chunk_priority, reverse=True)
    # Deduplicate near-identical chunks
    seen: set[str] = set()
    unique: list[str] = []
    for c in usable:
        key = c.strip()[:120]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _strip_old_restaurant_context(messages: list) -> None:
    """Keep only the base system prompt; drop stale RAG blocks."""
    kept = []
    for msg in messages:
        content = msg.get("content") or ""
        if msg.get("role") == "system" and RESTAURANT_CONTEXT_MARKER in content:
            continue
        kept.append(msg)
    messages[:] = kept


# Short inputs that should skip RAG retrieval entirely
SKIP_PATTERNS = [
    r"^(hi|hello|hey|hola|namaste|namaskar)$",
    r"^(ok|okay|hmm|hm|yeah|yep|yes|no|nope|nah|sure|fine|alright|acha|haan|theek)$",
    r"^(thanks|thank you|thankyou|shukriya|dhanyavaad)$",
    r"^(bye|goodbye|good bye|see you|tata|alvida)$",
    r"^(what|huh|sorry|pardon)$",
]


def _is_connect_placeholder(text: str) -> bool:
    return "[call connected" in (text or "").lower()


def _should_skip_retrieval(text: str) -> bool:
    """Return True if the text is too short/generic to benefit from RAG."""
    if _is_connect_placeholder(text):
        return True

    cleaned = text.strip().lower()

    # Very short inputs (1-2 words) that match filler patterns
    if len(cleaned.split()) <= 2:
        for pattern in SKIP_PATTERNS:
            if re.match(pattern, cleaned):
                return True

    # Extremely short text unlikely to be a real question
    if len(cleaned) < 3:
        return True

    return False


def _english_search_hints(text: str, language: str) -> list[str]:
    """English queries so embeddings match English menu docs."""
    if language == "en-IN":
        return []

    hints: list[str] = []
    seen: set[str] = set()
    lower = text.lower()

    def add(q: str) -> None:
        if q not in seen:
            seen.add(q)
            hints.append(q)

    for needle, english in _TOPIC_HINTS:
        if needle in text or needle.lower() in lower:
            add(english)

    if not hints:
        add("restaurant grand chennai menu dishes prices hours reservation parking")

    return hints


_OPENING_HOURS_INTENT_RE = re.compile(
    r"(?i)(\bopening\s+hands\b|"
    r"\b(open|opening|close|closing|closed)\b.*\b(hour|hours|time|timing|timings)\b|"
    r"\b(hour|hours|timing|timings)\b.*\b(open|close|restaurant)\b|"
    r"\bwhat\s+time\b|\bwhen\s+(are\s+you|do\s+you)\s+open\b|"
    r"\bare\s+you\s+open\b|\bworking\s+hours\b|\brestaurant\s+hours\b|"
    r"நேரம்|समय|సమయం|സമയം|ಸಮಯ)",
)


def _is_opening_hours_question(text: str) -> bool:
    """Opening hours use a fixed policy — do not pull day-by-day times from docs."""
    if _OPENING_HOURS_INTENT_RE.search(text):
        return True
    lower = text.lower()
    if "opening hands" in lower:
        return True
    if "hour" in lower and any(w in lower for w in ("open", "close", "timing", "when")):
        return True
    return False


def _format_context(chunks: list[str], max_chars: int = 2000) -> str:
    """Format retrieved chunks into a concise system message."""
    combined = ""
    for chunk in chunks:
        normalized = _normalize_rag_chunk(chunk.strip())
        entry = f"- {normalized}\n"
        if len(combined) + len(entry) > max_chars:
            break
        combined += entry

    return combined.strip()


class RAGContextInjectorProcessor(FrameProcessor):
    """
    Intercepts LLMContextFrame and injects retrieved restaurant context.

    On LLMContextFrame:
    1. Reads the message list from the LLMContext
    2. Extracts the latest user message text
    3. Checks if it's a substantive query (skip greetings/fillers)
    4. Retrieves relevant chunks from Qdrant via RAGService
    5. Injects a [RESTAURANT CONTEXT] system message into the context
    6. Pushes the frame downstream to the LLM
    """

    def __init__(
        self,
        rag_service: RAGService,
        max_context_chars: int = 2000,
        language: str = "en-IN",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._rag = rag_service
        self._max_chars = max_context_chars
        self._language = language
        self._lang_label = resolve_language(language)["label"]
        logger.info(
            "[RAGInjector] Initialized | lang={} max_context_chars={}",
            language,
            max_context_chars,
        )

    async def _retrieve_chunks(self, user_text: str) -> list[str]:
        """Retrieve from Qdrant; add English queries when docs are English."""
        if _is_opening_hours_question(user_text):
            logger.info("[RAGInjector] Opening-hours question — using fixed 24/7 policy")
            return [OPENING_HOURS_FACT]

        queries = [_normalize_query(user_text)]
        queries.extend(_english_search_hints(user_text, self._language))

        all_chunks: list[str] = []
        for q in queries:
            all_chunks.extend(await self._rag.retrieve(q))

        return _filter_and_rank_chunks(all_chunks)[:MAX_INJECT_CHUNKS]

    def _extract_user_text(self, messages: list) -> Optional[str]:
        """Extract the latest user message text from the context messages."""
        if not messages:
            return None

        # Walk messages in reverse to find the latest user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Content can be a string or a list of content parts
                if isinstance(content, str):
                    text = content.strip()
                elif isinstance(content, list):
                    # Extract text from content parts
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    text = " ".join(text_parts).strip()
                else:
                    continue

                if text:
                    return text
        return None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            # Get the messages from the LLMContext
            messages = frame.context.get_messages()
            user_text = self._extract_user_text(messages)

            if user_text and not _should_skip_retrieval(user_text):
                logger.debug(
                    "[RAGInjector] Retrieving | lang={} user='{}'",
                    self._language,
                    user_text[:60],
                )

                chunks = await self._retrieve_chunks(user_text)

                if chunks:
                    _strip_old_restaurant_context(messages)
                    context_text = _format_context(chunks, self._max_chars)
                    context_msg = {
                        "role": "system",
                        "content": (
                            f"{RESTAURANT_CONTEXT_MARKER}\n"
                            f'The customer asked: "{user_text}"\n\n'
                            f"Answer in **{self._lang_label}** only. "
                            f"The facts below may be in English — translate them naturally into "
                            f"{self._lang_label}; do not say you lack details if the answer is here.\n"
                            f"Use only lines that answer the question. Do not read unrelated facts.\n\n"
                            f"{context_text}\n"
                            f"[END RESTAURANT CONTEXT]"
                        ),
                    }

                    # Insert context message just before the last user message
                    # Find the index of the last user message
                    insert_idx = len(messages) - 1
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "user":
                            insert_idx = i
                            break

                    messages.insert(insert_idx, context_msg)
                    frame.context.set_messages(messages)

                    logger.info(
                        "[RAGInjector] Injected {} chars of context ({} chunks)",
                        len(context_text),
                        len(chunks),
                    )
                else:
                    logger.debug("[RAGInjector] No relevant context found")
            elif user_text:
                logger.debug(
                    "[RAGInjector] Skipped retrieval for short input: '{}'",
                    user_text[:40],
                )

            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
