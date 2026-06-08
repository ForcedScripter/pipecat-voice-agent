"""
AudioGateProcessor
──────────────────
Prevents the bot's TTS output from being picked up by the mic and
re-fed into the pipeline (echo loopback).

While the bot is speaking, this processor only passes audio frames
to downstream processors (VAD, STT) if the RMS energy exceeds a
high threshold — meaning the user must speak noticeably louder than
the ambient echo to break through.

After the bot stops speaking, a short grace period allows the echo
to decay before returning to normal sensitivity.

Place this processor BEFORE the VAD in the pipeline.
"""

import math
import asyncio
import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioGateProcessor(FrameProcessor):
    """
    Gate mic audio while the bot is speaking to prevent echo loopback.

    Modes:
        OPEN   — all audio passes through (normal listening)
        GATED  — only loud audio passes (bot is speaking, echo rejection)
        DECAY  — brief transition after bot stops, echo dying out

    Args:
        barge_in_rms:   RMS threshold to pass audio while gated.
                        Must be loud enough that echo doesn't reach it,
                        but low enough that a real voice does.
                        Typical values:
                          0.04–0.08  → too permissive, echo leaks through
                          0.10–0.15  → good for laptop speakers
                          0.15–0.25  → aggressive, only close/loud speech
        decay_secs:     Seconds to stay in DECAY after bot stops speaking.
                        Allows residual echo to die before reopening.
                        0.35s is often too short — 0.5–0.6s recommended.
    """

    def __init__(
        self,
        barge_in_rms: float = 0.15,
        decay_secs: float = 0.55,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._barge_in_rms = barge_in_rms
        self._decay_secs = decay_secs
        self._bot_speaking = False
        self._in_decay = False
        self._decay_task: asyncio.Task | None = None
        self._dropped_frames = 0
        self._passed_frames = 0
        logger.info(
            "[AudioGate] Initialized | barge_in_rms={} decay_secs={}",
            barge_in_rms,
            decay_secs,
        )

    def _rms(self, audio_bytes: bytes) -> float:
        """Calculate RMS energy of PCM-16 LE audio using numpy (fast path)."""
        if len(audio_bytes) < 2:
            return 0.0
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(samples**2))) / 32768.0

    async def _start_decay(self):
        """Transition from GATED to OPEN after a brief decay period."""
        self._in_decay = True
        await asyncio.sleep(self._decay_secs)
        self._in_decay = False
        if self._dropped_frames > 0:
            logger.debug(
                "[AudioGate] Decay complete — reopened. "
                "Dropped {} frames, passed {} during gated period.",
                self._dropped_frames,
                self._passed_frames,
            )
        self._dropped_frames = 0
        self._passed_frames = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._in_decay = False
            self._dropped_frames = 0
            self._passed_frames = 0
            # Cancel any pending decay
            if self._decay_task and not self._decay_task.done():
                self._decay_task.cancel()
            logger.debug("[AudioGate] GATED — bot started speaking")
            await self.push_frame(frame, direction)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            # Start decay timer
            self._decay_task = asyncio.create_task(self._start_decay())
            logger.debug(
                "[AudioGate] Bot stopped — entering decay ({:.0f}ms)",
                self._decay_secs * 1000,
            )
            await self.push_frame(frame, direction)

        elif (
            isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM
        ):
            if self._bot_speaking or self._in_decay:
                rms = self._rms(frame.audio)
                if rms >= self._barge_in_rms:
                    # Loud enough — likely real user speech (barge-in)
                    self._passed_frames += 1
                    await self.push_frame(frame, direction)
                else:
                    # Too quiet — likely echo, drop it
                    self._dropped_frames += 1
                    # Don't push — frame is silently dropped
            else:
                # OPEN mode — pass everything
                await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
