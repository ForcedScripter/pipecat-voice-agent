"""
ContextSanitizerProcessor
─────────────────────────
Merges or replaces consecutive same-role messages in the LLM context before
every LLM call.
"""

import re

from loguru import logger

from pipecat.frames.frames import Frame, LLMContextFrame, LLMMessagesAppendFrame, LLMRunFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ContextSanitizerProcessor(FrameProcessor):
    """
    Sanitizes the LLM context before each LLM call:
    - Consecutive USER messages: keep only the latest
    - Consecutive ASSISTANT messages: replace if truncated, merge if complete
    - Trims history to max_history_messages (non-system)
    """

    TRUNCATION_LENGTH_THRESHOLD = 60
    MAX_HISTORY_MESSAGES = 6
    EPHEMERAL_SYSTEM_MARKERS = (
        "customer just connected",
        "greet as aiden in one short sentence",
    )
    AIDEN_SYSTEM_MARKER = "you are aiden"

    def __init__(self, context: LLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (LLMContextFrame, LLMMessagesAppendFrame, LLMRunFrame)):
            self._prepare_context()

        await self.push_frame(frame, direction)

    def _prepare_context(self):
        """Sanitize and trim before the LLM reads context."""
        self._sanitize()
        self._drop_ephemeral_system_messages()
        self._dedupe_user_messages()
        self._drop_bad_assistant_turns()
        self._trim_history()

    @classmethod
    def drop_truncated_last_assistant(cls, context: LLMContext) -> bool:
        """Remove the last assistant message if it looks cut off by barge-in."""
        messages = context.messages
        if len(messages) < 2:
            return False

        last = messages[-1]
        if last.get("role") != "assistant":
            return False

        content = (last.get("content") or "").strip()
        if not cls._is_truncated_assistant(content):
            return False

        messages.pop()
        logger.info(
            "[ContextSanitizer] Removed truncated assistant: '{}'",
            content[:60],
        )
        return True

    @classmethod
    def _is_truncated_assistant(cls, content: str) -> bool:
        content = content.strip()
        return len(content) < cls.TRUNCATION_LENGTH_THRESHOLD and not content.endswith(
            (".", "?", "!", ",", "—")
        )

    def _sanitize(self):
        """Sanitize self._context.messages in-place before LLM sees it."""
        messages = self._context.messages
        if not messages:
            return

        merged: list[dict] = [dict(messages[0])]

        for msg in messages[1:]:
            role = msg.get("role", "")
            prev_role = merged[-1].get("role", "")

            if role == prev_role and role == "user":
                prev_content = (merged[-1].get("content") or "").strip()
                new_content = (msg.get("content") or "").strip()
                # Keep whichever message is longer/more complete —
                # Sarvam STT sometimes delivers a fuller partial first,
                # then a shorter correction; always taking the latest
                # would lose context.
                if len(new_content) >= len(prev_content):
                    merged[-1] = dict(msg)
                    logger.info(
                        "[ContextSanitizer] Replaced user message '{}' → '{}'",
                        prev_content[:60],
                        new_content[:60],
                    )
                else:
                    logger.info(
                        "[ContextSanitizer] Kept longer user message '{}' (discarded '{}')",
                        prev_content[:60],
                        new_content[:60],
                    )

            elif role == prev_role and role == "assistant":
                prev_content = (merged[-1].get("content") or "").strip()
                new_content = (msg.get("content") or "").strip()

                if self._is_truncated_assistant(prev_content):
                    merged[-1] = dict(msg)
                    logger.info(
                        "[ContextSanitizer] Replaced truncated assistant '{}' → '{}'",
                        prev_content[:60],
                        new_content[:60],
                    )
                else:
                    merged[-1] = dict(merged[-1])
                    merged[-1]["content"] = (prev_content + " " + new_content).strip()
                    logger.info(
                        "[ContextSanitizer] Merged consecutive assistant messages → '{}'",
                        merged[-1]["content"][:80],
                    )

            else:
                merged.append(dict(msg))

        self._context.messages[:] = merged

    def _drop_ephemeral_system_messages(self):
        """Remove one-shot connect prompts; keep the main Aiden system prompt."""
        messages = self._context.messages
        if not any(m.get("role") == "user" for m in messages):
            return

        aiden_prompt: dict | None = None
        rag_msgs: list[dict] = []
        for m in messages:
            if m.get("role") != "system":
                continue
            content = m.get("content") or ""
            lower = content.lower()
            if "[RESTAURANT CONTEXT]" in content:
                rag_msgs.append(dict(m))
                continue
            if any(marker in lower for marker in self.EPHEMERAL_SYSTEM_MARKERS):
                logger.info("[ContextSanitizer] Dropped ephemeral connect system message")
                continue
            if aiden_prompt is None or self.AIDEN_SYSTEM_MARKER in lower:
                aiden_prompt = dict(m)

        dialog = [m for m in messages if m.get("role") != "system"]
        rebuilt: list[dict] = []
        if aiden_prompt:
            rebuilt.append(aiden_prompt)
        if rag_msgs:
            rebuilt.append(rag_msgs[-1])
        rebuilt.extend(dialog)
        self._context.messages[:] = rebuilt

    def _normalize_user_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @staticmethod
    def _is_connect_placeholder(text: str) -> bool:
        return "[call connected" in (text or "").lower()

    def _dedupe_user_messages(self):
        """Drop repeated or overlapping user transcripts (STT duplicates)."""
        messages = self._context.messages
        seen: set[str] = set()
        kept: list[dict] = []
        for m in messages:
            if m.get("role") != "user":
                kept.append(m)
                continue
            raw = m.get("content", "") or ""
            if self._is_connect_placeholder(raw):
                logger.info("[ContextSanitizer] Dropped connect placeholder user message")
                continue
            key = self._normalize_user_text(raw)
            if not key:
                continue
            if key in seen:
                logger.info("[ContextSanitizer] Dropped duplicate user: '{}'", key[:50])
                continue
            seen.add(key)
            kept.append(m)
        self._context.messages[:] = kept

    def _drop_bad_assistant_turns(self):
        """Remove cut-off assistant replies from context (they confuse the next turn)."""
        messages = self._context.messages
        if not messages:
            return
        cleaned: list[dict] = []
        for m in messages:
            if m.get("role") == "assistant":
                content = (m.get("content") or "").strip()
                if self._is_truncated_assistant(content):
                    logger.info(
                        "[ContextSanitizer] Dropped truncated assistant: '{}'",
                        content[:60],
                    )
                    continue
                # Mid-sentence list dumps without ending punctuation
                if len(content) > 40 and not content.endswith((".", "?", "!")):
                    if re.search(r"(peak hours|following times)\s*$", content, re.I):
                        logger.info(
                            "[ContextSanitizer] Dropped incomplete list assistant reply"
                        )
                        continue
                # Markdown tables / "to to to" hallucinations (break TTS)
                if content.count("|") >= 2 or re.search(r"\bto\s+to\s+to\b", content, re.I):
                    logger.info(
                        "[ContextSanitizer] Dropped malformed table assistant reply"
                    )
                    continue
            cleaned.append(m)
        self._context.messages[:] = cleaned

    def _trim_history(self):
        """Keep base system prompt, latest RAG block, and recent conversation."""
        messages = self._context.messages
        if len(messages) < 1:
            return

        aiden_system: dict | None = None
        rag_system: list[dict] = []
        for m in messages:
            if m.get("role") != "system":
                continue
            content = m.get("content") or ""
            if "[RESTAURANT CONTEXT]" in content:
                rag_system.append(m)
            elif self.AIDEN_SYSTEM_MARKER in content.lower():
                aiden_system = m
            elif aiden_system is None:
                aiden_system = m

        if len(rag_system) > 1:
            rag_system = [rag_system[-1]]

        other_msgs = [m for m in messages if m.get("role") != "system"]
        if len(other_msgs) > self.MAX_HISTORY_MESSAGES:
            other_msgs = other_msgs[-self.MAX_HISTORY_MESSAGES :]

        rebuilt: list[dict] = []
        if aiden_system:
            rebuilt.append(aiden_system)
        rebuilt.extend(rag_system)
        rebuilt.extend(other_msgs)
        self._context.messages[:] = rebuilt
