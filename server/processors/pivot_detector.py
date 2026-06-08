import re
from typing import Optional
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMMessagesAppendFrame,
    InterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


PIVOT_PATTERNS = [
    r"\bactually\b",
    r"\bwait\b",
    r"\bnever mind\b",
    r"\bforget that\b",
    r"\bby the way\b",
    r"\boh wait\b",
    r"\bsomething else\b",
    r"\bswitching gears\b",
    r"\blet me ask you\b",
    r"\bquick question\b",
    r"\bone more thing\b",
    r"\bruko\b",
    r"\bek second\b",
    r"\balag baat\b",
    r"\bwaise\b",
]
class PivotDetectorProcessor(FrameProcessor):
    """
    Detects mid-conversation topic changes (pivots) and handles them gracefully.

    On pivot detection:
    1. Fires StartInterruptionFrame upstream → stops current bot audio immediately
    2. Injects a system hint into LLMMessagesFrame → LLM acknowledges the shift naturally
    """

    def __init__(self, *, context_window: int = 3, **kwargs):
        super().__init__(**kwargs)
        self._recent_topics: list[str] = []
        self._context_window = context_window
        self._last_bot_topic: Optional[str] = None
        self._pivot_detected = False
        self._pivot_text: Optional[str] = None

    def _pattern_pivot(self, text: str) -> bool:
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in PIVOT_PATTERNS)

    def _semantic_pivot(self, new_text: str) -> bool:
        if not self._last_bot_topic:
            return False
        stop = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "i",
            "you",
            "we",
            "it",
            "of",
            "to",
            "in",
            "and",
            "or",
            "that",
            "this",
        }
        last = set(self._last_bot_topic.lower().split()) - stop
        new = set(new_text.lower().split()) - stop
        if not last or not new:
            return False
        overlap = len(last & new) / max(len(last), len(new))
        return overlap < 0.15

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._pivot_detected = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if self._pattern_pivot(text) or self._semantic_pivot(text):
                self._pivot_detected = True
                self._pivot_text = text
                logger.info(f"[PivotDetector] Topic shift detected: '{text}'")
                # Stop bot mid-speech
                await self.push_frame(InterruptionFrame(), FrameDirection.UPSTREAM)
            else:
                self._pivot_text = None

            self._recent_topics.append(text)
            if len(self._recent_topics) > self._context_window:
                self._recent_topics.pop(0)
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMMessagesAppendFrame):
            if self._pivot_detected and self._pivot_text:
                pivot_hint = {
                    "role": "system",
                    "content": (
                        f"The user just changed the topic. Their new message is: '{self._pivot_text}'. "
                        f"Smoothly acknowledge the shift in one short phrase, then answer. "
                        f"Keep your overall response brief."
                    ),
                }
                messages = list(frame.messages)
                messages.insert(-1, pivot_hint)
                frame = LLMMessagesAppendFrame(messages=messages)
                self._pivot_detected = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame):
            # Track last bot topic for semantic pivot comparison
            if frame.text:
                self._last_bot_topic = frame.text[:120]
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
