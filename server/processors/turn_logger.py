"""Log user-turn and LLM lifecycle events for debugging silence issues."""

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TurnLifecycleProcessor(FrameProcessor):
    """Emit [Turn] logs when speech turns start/stop and the LLM runs."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.info("[Turn] user started speaking")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info("[Turn] user stopped speaking")
        elif isinstance(frame, LLMRunFrame):
            logger.info("[Turn] LLM triggered")
        elif isinstance(frame, LLMFullResponseEndFrame):
            logger.info("[Turn] LLM response ended")

        await self.push_frame(frame, direction)
