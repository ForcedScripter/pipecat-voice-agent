"""
LLMEmptyGuardProcessor
──────────────────────
Detects when the LLM produces an empty response (no TextFrames) and
injects a fallback message to keep the conversation going.

Also enforces a max-wait timeout: if the LLM hasn't produced any text
within `timeout_secs` of starting, the fallback fires immediately
instead of waiting for LLMFullResponseEndFrame.
"""

import asyncio
import random

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


DEFAULT_FALLBACKS = [
    "i'm here if you need anything",
    "let me know how I can help",
    "i'm here whenever you're ready",
    "no worries, just let me know what you need",
    "take your time, i'm here to help",
    "sure thing, what can I do for you?",
    "i understand, let me know if there's anything I can help with",
    "alright, i'm here if you need me",
]


class LLMEmptyGuardProcessor(FrameProcessor):
    """
    Injects a fallback TextFrame when the LLM produces an empty response.
    Rotates through a list of fallback messages so the user never hears
    the same phrase twice in a row.

    Also starts a timer when the LLM begins processing. If no TextFrame
    arrives within `timeout_secs`, the fallback fires immediately —
    preventing long silences when the LLM's safety filter blocks output.

    Args:
        fallback_texts: List of fallback strings to rotate through.
                        Defaults to DEFAULT_FALLBACKS.
        timeout_secs:   Max seconds to wait for the first TextFrame after
                        LLMFullResponseStartFrame before injecting fallback.
    """

    def __init__(
        self,
        fallback_texts: list[str] | None = None,
        timeout_secs: float = 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._fallbacks = fallback_texts or DEFAULT_FALLBACKS
        self._timeout_secs = timeout_secs
        self._last_index: int = -1
        self._has_text = False
        self._interrupted = False
        self._timeout_fired = False
        self._timeout_task: asyncio.Task | None = None
        logger.info(
            "[LLMEmptyGuard] Initialized | {} fallback messages loaded | timeout={}s",
            len(self._fallbacks),
            timeout_secs,
        )

    def _pick_fallback(self) -> str:
        """Pick a random fallback, never repeating the last one."""
        available = [
            (i, t) for i, t in enumerate(self._fallbacks) if i != self._last_index
        ]
        index, text = random.choice(available)
        self._last_index = index
        return text

    def _cancel_timeout(self):
        """Cancel any pending timeout task."""
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            self._timeout_task = None

    async def _on_timeout(self):
        """Fires when the LLM hasn't produced text within timeout_secs."""
        try:
            await asyncio.sleep(self._timeout_secs)
        except asyncio.CancelledError:
            return

        # Timer expired and no text arrived — inject fallback now
        if not self._has_text and not self._interrupted:
            self._timeout_fired = True
            fallback = self._pick_fallback()
            logger.warning(
                "[LLMEmptyGuard] LLM timeout ({}s) — no text received, injecting fallback: '{}'",
                self._timeout_secs,
                fallback,
            )
            await self.push_frame(
                TextFrame(text=fallback),
                FrameDirection.DOWNSTREAM,
            )
            await self.push_frame(
                LLMFullResponseEndFrame(),
                FrameDirection.DOWNSTREAM,
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            # LLM just started — reset state and start the timeout clock
            self._has_text = False
            self._interrupted = False
            self._timeout_fired = False
            self._cancel_timeout()
            self._timeout_task = asyncio.create_task(self._on_timeout())
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame):
            self._has_text = True
            self._cancel_timeout()  # Got text — no need for timeout
            if not self._timeout_fired:
                await self.push_frame(frame, direction)
            # If timeout already fired, silently drop late-arriving text
            # to avoid double-speaking

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._cancel_timeout()
            if not self._has_text and not self._interrupted and not self._timeout_fired:
                # End of response with no text and timeout didn't fire
                # (response finished quickly but empty)
                fallback = self._pick_fallback()
                logger.warning(
                    "[LLMEmptyGuard] LLM returned empty response — injecting fallback: '{}'",
                    fallback,
                )
                await self.push_frame(
                    TextFrame(text=fallback),
                    direction,
                )
            elif not self._has_text and self._interrupted:
                logger.debug(
                    "[LLMEmptyGuard] Empty response after interruption — suppressed fallback"
                )
            # Reset for next response
            self._has_text = False
            self._interrupted = False
            self._timeout_fired = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            self._cancel_timeout()
            was_streaming_llm_text = self._has_text
            self._has_text = False
            self._timeout_fired = False
            if was_streaming_llm_text:
                self._interrupted = True
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
