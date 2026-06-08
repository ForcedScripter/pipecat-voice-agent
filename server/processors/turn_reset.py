"""Drop truncated assistant context when the user barges in."""

from loguru import logger

from pipecat.frames.frames import Frame, InterruptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from processors.context_sanitizer import ContextSanitizerProcessor


class TurnResetProcessor(FrameProcessor):
    """Remove cut-off assistant replies from context after an interruption."""

    def __init__(self, context: LLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            if ContextSanitizerProcessor.drop_truncated_last_assistant(self._context):
                logger.info("[TurnReset] dropped truncated assistant after interruption")

        await self.push_frame(frame, direction)
