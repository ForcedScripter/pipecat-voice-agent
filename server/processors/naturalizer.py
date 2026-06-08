import re
import random
from collections import deque
from typing import List, Dict

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMFullResponseEndFrame,
    InterruptionFrame,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)


# ==================== SEMANTIC STARTERS ====================

STARTER_CATEGORIES_EN: Dict[str, List[str]] = {
    "affirmative": ["Yeah, ", "Sure, ", "Absolutely, ", "Right, "],
    "explanatory": ["So, ", "Basically, ", "Actually, ", "Well, "],
    "insightful": ["Ah, ", "Oh, ", "Interesting, ", "Right, "],
    "casual": ["Okay, ", "Got it — ", "Hmm, "],
    "neutral": ["", "", "", "", "Well, ", "So, "],  # empty has higher weight
}

STARTER_CATEGORIES_HI: Dict[str, List[str]] = {
    "affirmative": ["Haan, ", "Bilkul, ", "Sure, ", "Absolutely, "],
    "explanatory": ["Dekho, ", "Toh, ", "Actually, ", "Basically, "],
    "insightful": ["Ah, ", "Oh, ", "Acha, "],
    "casual": ["Okay, ", "Hmm, ", "Acha, "],
    "neutral": ["", "", "", "", "Toh, "],
}

_HOUR_WORDS = {
    0: "twelve", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}
_TEEN_MINUTES = {
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
}
_TENS_MINUTES = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}

SEMANTIC_TRIGGERS = {
    "affirmative": [
        r"\b(yes|yeah|sure|correct|right|agree|exactly|indeed)\b",
        r"\b(of course|definitely|absolutely)\b",
        r"^(Yes|Yeah|Sure|Correct|Right)",
    ],
    "explanatory": [
        r"\b(because|since|so|therefore|thus|means|implies)\b",
        r"\b(for example|like|such as)\b",
        r"\b(how|why|what|when)\b",
    ],
    "insightful": [
        r"\b(interesting|surprising|actually|turns out|realize|notice)\b",
        r"\b(oh|wow|ah)\b",
    ],
}


class ResponseNaturalizerProcessor(FrameProcessor):
    def __init__(
        self,
        add_starters: bool = True,
        language: str = "en-IN",
        min_chunk_length: int = 12,
        starter_cooldown: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._starter_categories = (
            STARTER_CATEGORIES_HI if language == "hi-IN" else STARTER_CATEGORIES_EN
        )

        self._add_starters = add_starters
        self._min_chunk_length = min_chunk_length
        self._recent_starters = deque(maxlen=starter_cooldown)

        # Buffer accumulates streaming chunks until we have enough
        # text to pick a meaningful starter. Once starter is injected,
        # subsequent chunks pass through immediately.
        self._buffer = ""
        self._starter_injected = False

    # ── Semantic starter logic ───────────────────────────────────────────────

    def _get_semantic_category(self, text: str) -> str:
        text_lower = text.lower()
        for category, patterns in SEMANTIC_TRIGGERS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return category
        return "casual" if len(text.split()) < 15 else "explanatory"

    def _pick_semantic_starter(self, text: str) -> str:
        if not self._add_starters:
            return ""

        category = self._get_semantic_category(text)
        candidates = self._starter_categories.get(category, self._starter_categories["neutral"])
        available = [s for s in candidates if s not in self._recent_starters]

        if not available:
            available = candidates

        if random.random() < 0.45 and "" in self._starter_categories["neutral"]:
            starter = ""
        else:
            starter = random.choice(available)

        if starter:
            self._recent_starters.append(starter)

        return starter

    # ── Text cleaning ────────────────────────────────────────────────────────

    @staticmethod
    def _speak_clock_time(hour: int, minute: int, ampm: str | None) -> str:
        """11:30 AM -> eleven thirty AM"""
        h12 = hour % 12 or 12
        spoken_h = _HOUR_WORDS.get(h12, str(h12))
        if minute == 0:
            spoken_m = ""
        elif minute < 10:
            spoken_m = f"oh {_HOUR_WORDS[minute]}"
        elif minute in _TEEN_MINUTES:
            spoken_m = _TEEN_MINUTES[minute]
        else:
            tens, ones = divmod(minute, 10)
            spoken_m = _TENS_MINUTES.get(tens, str(tens))
            if ones:
                spoken_m = f"{spoken_m} {_HOUR_WORDS[ones]}"
        parts = [spoken_h]
        if spoken_m:
            parts.append(spoken_m)
        if ampm:
            parts.append(ampm.upper())
        return " ".join(parts)

    def _format_times_for_speech(self, text: str) -> str:
        def repl(m: re.Match) -> str:
            h, mi, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
            return self._speak_clock_time(h, mi, ampm)

        return re.sub(
            r"\b(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)?\b",
            repl,
            text,
        )

    def _strip_markdown_tables(self, text: str) -> str:
        """Remove markdown tables — pipe rows often become 'to to to' after dash cleanup."""
        if "|" not in text:
            return text

        parts: list[str] = []
        for line in text.splitlines():
            if line.count("|") < 2:
                parts.append(line)
                continue
            cells = [c.strip() for c in line.split("|") if c.strip()]
            cells = [
                c
                for c in cells
                if not re.fullmatch(r"[-:\s]+", c)
                and not re.search(r"\bto\s+to\b", c, re.I)
                and len(c) > 1
            ]
            if cells:
                parts.append(", ".join(cells))
        return " ".join(parts) if parts else ""

    def _format_for_speech(self, text: str) -> str:
        """Make LLM output sound natural in TTS — not like reading a document."""
        if not text:
            return ""

        text = self._strip_markdown_tables(text)
        text = text.replace("|", " ")
        text = re.sub(r"(\bto\s+){3,}", " ", text, flags=re.I)

        # Fix glued words from streaming (Aidenhappy, timesLunch, Saturday1)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"times([A-Za-z])", r"times \1", text, flags=re.I)
        text = re.sub(
            r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(\d)",
            r"\1 \2",
            text,
            flags=re.I,
        )
        text = re.sub(r"(\d)\s*(PM|AM)\b", r"\1 \2", text, flags=re.I)
        # Common glued phrases from streaming
        text = re.sub(
            r"(menu|prices|allergens|hours|reservation|calling|today|help)([a-z]{3,})",
            r"\1 \2",
            text,
            flags=re.I,
        )

        # Prices
        text = re.sub(r"₹\s*(\d+)", r"\1 rupees", text)

        # Time ranges before stripping colons
        text = self._format_times_for_speech(text)

        # Day/time range dashes -> spoken "to"
        text = re.sub(
            r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*[-–]\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
            r"\1 through \2",
            text,
            flags=re.I,
        )
        # Only spoken range dashes (not markdown --- separators)
        text = re.sub(r"\s+[–—]\s+", " to ", text)
        text = re.sub(r"(\d)\s*-\s*(\d)", r"\1 to \2", text)

        # Section labels the model might echo (Opening Hours:, Category:)
        text = re.sub(
            r"(?i)\b(opening hours|category|type|description|contains allergens|preparation time|price|spice level)\s*:",
            "",
            text,
        )

        # Colons and semicolons — TTS often says "colon"
        text = re.sub(r":\s*", ", ", text)
        text = re.sub(r";\s*", ", ", text)

        # Bullets / list markers
        text = re.sub(r"^\s*[-•*]\s*", "", text, flags=re.M)
        text = re.sub(r"\s+[-•*]\s+", ", ", text)

        # Symbols TTS mishandles
        text = text.replace("(", " ").replace(")", " ")
        text = text.replace("/", " or ")
        text = text.replace("&", " and ")
        text = text.replace("—", ", ")
        text = text.replace("–", ", ")

        # Collapse duplicate commas / spaces
        text = re.sub(r",\s*,+", ", ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip(" ,")

    def _strip_spellings(self, text: str) -> str:
        SPELLING_PATTERNS = [
            re.compile(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b"),
            re.compile(r"\b(?:[A-Za-z][\-\s]){3,}[A-Za-z]\b"),
            re.compile(r"\b(?:[A-Za-z]\.\s*){3,}"),
            re.compile(r"(?:\b[A-Z]\b[\s\-\.,]*){4,}"),
        ]
        for pattern in SPELLING_PATTERNS:
            text = pattern.sub("", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = re.sub(r"^[\s,\-—]+", "", text)
        text = re.sub(r"[\s,\-—]+$", "", text)
        return text

    def _clean(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        # Model sometimes ignores "no code" — strip before TTS
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`\n]+`", "", text)
        text = re.sub(r"(?m)^\s*(def |class |import |from \w+ import|#include).*$", "", text)
        if re.search(
            r"(?i)\b(def |class |import |function |console\.log|here'?s the code)\b", text
        ):
            return (
                "I can only help with Restaurant Grand Chennai — menu, reservations, "
                "hours, parking, and allergies. What would you like to know?"
            )

        ROBOTIC_SUBS = [
            (r"(?i)^(As an AI[,.]?\s*)", ""),
            (r"(?i)^(Certainly!\s*I'?d be happy to help\.?\s*)", ""),
            (r"(?i)^(Of course!\s*I'?d be happy to assist\.?\s*)", ""),
            (r"(?i)\bAs a language model\b", ""),
            (r"(?i)\bI don't have personal opinions\b", "From what I can tell"),
            (r"(?i)\bI'm just an AI\b", ""),
            (r"(?i)\bI cannot provide\b", "I can't"),
        ]
        for pattern, repl in ROBOTIC_SUBS:
            text = re.sub(pattern, repl, text)

        text = self._strip_spellings(text)
        if not text:
            return ""

        TTS_PUNCTUATION_SUBS = [
            (r"\.{2,}", ", "),  # ellipsis → pause (single periods are fine for TTS)
            (r"\*+", ""),
            (r"#+\s?", ""),
            (r"_+", ""),
            (r"\s-\s", " — "),
            (r",\s*,+", ", "),
            (r"\s{2,}", " "),
        ]
        for pattern, repl in TTS_PUNCTUATION_SUBS:
            text = re.sub(pattern, repl, text)

        text = self._format_for_speech(text)
        return text.strip()

    # ── State reset ──────────────────────────────────────────────────────────

    def _reset(self):
        """
        Full reset for a fresh response turn.
        Called on both normal end-of-response and interruptions.
        Clears the buffer so no partial text from the previous turn
        bleeds into the next one — this is what prevented garbling.
        """
        self._buffer = ""
        self._starter_injected = False

    # ── Frame processing ─────────────────────────────────────────────────────

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            text = self._clean(frame.text)

            if not text:
                await self.push_frame(frame, direction)
                return

            # Fast path: no starter buffering — lower latency to first TTS audio
            if not self._add_starters:
                await self.push_frame(TextFrame(text=text), direction)
                return

            if not self._starter_injected:
                # Still buffering — accumulate until we have enough text
                # to pick a meaningful starter category.
                self._buffer += (" " + text) if self._buffer else text

                if len(self._buffer) >= self._min_chunk_length:
                    # We have enough — pick starter on the full buffered text
                    # so the category detection is accurate, not based on
                    # a single partial token like "Int" or "o".
                    buffered = self._buffer
                    self._buffer = ""
                    self._starter_injected = True

                    starter = self._pick_semantic_starter(buffered)
                    if starter:
                        buffered = starter + buffered[0].lower() + buffered[1:]

                    await self.push_frame(TextFrame(text=buffered), direction)
                # else: still buffering, don't push yet

            else:
                # Starter already injected — pass chunks through immediately
                await self.push_frame(TextFrame(text=text), direction)

        elif isinstance(frame, LLMFullResponseEndFrame):
            # Flush any remaining buffer before resetting.
            # This handles the rare case where the entire LLM response
            # was shorter than min_chunk_length.
            if self._buffer:
                buffered = self._buffer
                self._reset()
                starter = self._pick_semantic_starter(buffered)
                if starter:
                    buffered = starter + buffered[0].lower() + buffered[1:]
                await self.push_frame(TextFrame(text=buffered), direction)
            else:
                self._reset()

            await self.push_frame(frame, direction)

        elif isinstance(frame, InterruptionFrame):
            # User interrupted — hard reset, discard buffer entirely.
            # Do NOT flush buffer here: partial text from an interrupted
            # response should never reach TTS.
            self._reset()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
