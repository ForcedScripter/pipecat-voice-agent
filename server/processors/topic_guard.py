"""
TopicGuardProcessor
───────────────────
Blocks code and off-topic LLM output before TTS.
Prompts alone are not enough — this enforces restaurant-only replies in code.
"""

import re

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    TextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

REFUSAL_MESSAGE = (
    "I can only help with Restaurant Grand Chennai — the menu, reservations, "
    "opening hours, parking, and allergies. What would you like to know about the restaurant?"
)

_CODE_REQUEST_RE = re.compile(
    r"(?i)("
    r"\b(write|generate|create|build|make|give|show)\b.{0,40}\b(code|script|program|app|api|website|file)s?\b|"
    r"\b(code|script|program)\b.{0,30}\b(for me|example|sample|file|python|java)\b|"
    r"\b(python|javascript|typescript|java|c\+\+|c#|react|node\.?js|html|css|sql|flutter|kotlin)\b|"
    r"\b(function|algorithm|debug|compile|leetcode|github|repository|repo)\b|"
    r"\b(homework|assignment|coursework|project)\b.{0,20}\b(code|program)\b|"
    r"\bdef\s+\w+\s*\(|\bclass\s+\w+|#include\s*<|import\s+\w+"
    r")",
)

_OFF_TOPIC_RE = re.compile(
    r"(?i)("
    r"\b(weather|forecast)\b|"
    r"\b(news|politic|election|president)\b|"
    r"\b(stock|crypto|bitcoin|invest)\b|"
    r"\b(write|draft)\b.{0,20}\b(essay|email|letter|story|poem|resume|cv)\b|"
    r"\b(solve|calculate)\b.{0,20}\b(math|equation|physics|chemistry)\b|"
    r"\b(who is|what is the capital|tell me about)\b.{0,30}\b(?!chef|restaurant|menu|dish)"
    r")",
)

_RESTAURANT_TOPIC_RE = re.compile(
    r"(?i)\b("
    r"menu|dish|dishes|food|meal|price|cost|rupee|allerg|reserv|book|table|seat|"
    r"parking|park|valet|hour|hours|open|close|timing|restaurant|chennai|order|"
    r"delivery|takeaway|biryani|veg|non[- ]?veg|combo|chef|dining|party|guest"
    r")\b",
)


def _is_connect_placeholder(text: str) -> bool:
    return "[call connected" in (text or "").lower()


def is_off_topic_user_request(text: str) -> bool:
    """True when the user is not asking about restaurant support."""
    if not text or _is_connect_placeholder(text):
        return False
    if _RESTAURANT_TOPIC_RE.search(text):
        return False
    if _CODE_REQUEST_RE.search(text):
        return True
    if _OFF_TOPIC_RE.search(text):
        return True
    return False


def looks_like_code_output(text: str) -> bool:
    """True when assistant text looks like source code or technical instructions."""
    if not text or len(text.strip()) < 6:
        return False

    lower = text.lower()

    if "```" in text or text.count("`") >= 2:
        return True
    if re.search(
        r"(?m)^\s*(def |class |import |from \w+ import|#include|public static|"
        r"private |package |using namespace)",
        text,
    ):
        return True
    if re.search(
        r"(?i)\b(console\.log|printf\s*\(|System\.out|fn main|async function|"
        r"=>|\.py\b|\.js\b|\.ts\b|\.java\b|\.cpp\b)",
        text,
    ):
        return True
    if text.count("{") >= 2 and text.count("}") >= 2:
        return True
    if text.count(";") >= 4 and "rupee" not in lower and "menu" not in lower:
        return True
    # Step-by-step programming instructions
    if re.search(r"(?i)\b(step \d+|line \d+|copy this code|paste the following)\b", text):
        return True
    if re.search(r"(?i)\b(here'?s the code|below is the code|code snippet)\b", text):
        return True
    return False


class TopicGuardProcessor(FrameProcessor):
    """
    Placed after the LLM, before naturalizer/TTS.
    Drops code/off-topic streams and speaks a fixed restaurant-only refusal instead.
    """

    def __init__(self, context: LLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._refuse_turn = False
        self._suppress_output = False
        self._buffer: list[str] = []

    def _latest_user_text(self) -> str:
        for msg in reversed(self._context.messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content") or ""
            if isinstance(content, str):
                return content.strip()
        return ""

    def _refresh_refusal_flag(self) -> None:
        user = self._latest_user_text()
        self._refuse_turn = is_off_topic_user_request(user)
        if self._refuse_turn:
            logger.warning(
                "[TopicGuard] Off-topic / code request blocked | user='{}'",
                user[:100],
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (LLMContextFrame, LLMRunFrame)):
            self._refresh_refusal_flag()

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = []
            self._suppress_output = self._refuse_turn

        elif isinstance(frame, TextFrame):
            if direction != FrameDirection.DOWNSTREAM:
                await self.push_frame(frame, direction)
                return

            self._buffer.append(frame.text or "")

            if self._refuse_turn or looks_like_code_output("".join(self._buffer)):
                self._suppress_output = True
                return

            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._suppress_output or looks_like_code_output("".join(self._buffer)):
                logger.warning(
                    "[TopicGuard] Replaced LLM output with restaurant-only refusal"
                )
                await self.push_frame(TextFrame(text=REFUSAL_MESSAGE), direction)
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
