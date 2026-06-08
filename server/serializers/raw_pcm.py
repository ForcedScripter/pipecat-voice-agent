import json

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

_TARGET_BYTES = 1024  # 512 samples × 2 bytes = 1024 bytes for Silero at 16 kHz


class RawPCMSerializer(FrameSerializer):
    """Per-connection PCM re-chunker: buffers incoming bytes until _TARGET_BYTES."""

    def __init__(self, **kwargs):
        # RTVI control messages must reach the browser (bot-started-speaking, etc.)
        params = FrameSerializer.InputParams(ignore_rtvi_messages=False)
        super().__init__(params=params, **kwargs)
        self._accumulator = bytearray()
        self._count = 0

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if self.should_ignore_frame(frame):
            return None
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            try:
                return json.dumps(frame.message)
            except Exception:
                return None
        return None

    async def deserialize(self, data: bytes | str):
        if isinstance(data, bytes):
            self._accumulator.extend(data)
            if len(self._accumulator) < _TARGET_BYTES:
                return None  # not enough yet, wait for more
            chunk = bytes(self._accumulator[:_TARGET_BYTES])
            del self._accumulator[:_TARGET_BYTES]
            self._count += 1
            if self._count == 1 or self._count % 50 == 0:
                print(
                    f"[RawPCMSerializer] emitting chunk bytes={len(chunk)}, count={self._count}",
                    flush=True,
                )
            return InputAudioRawFrame(
                audio=chunk,
                sample_rate=16000,
                num_channels=1,
            )
        if isinstance(data, str):
            # A text/JSON message from the browser signals a control event (e.g.
            # start-speaking, interruption). Flush any partially accumulated audio
            # so stale bytes from the previous turn don't bleed into the next one
            # and cause garbled STT transcriptions.
            if self._accumulator:
                print(
                    f"[RawPCMSerializer] flushing {len(self._accumulator)} stale bytes on control message",
                    flush=True,
                )
                self._accumulator.clear()
            try:
                msg = json.loads(data)
                return InputTransportMessageFrame(message=msg)
            except Exception:
                return None
        return None

