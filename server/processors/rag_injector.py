import json
import asyncio
from typing import Optional
from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from rag_service import RAGService

CONTEXT_MARKER = "[DOCUMENT CONTEXT]"

class RAGContextInjectorProcessor(FrameProcessor):
    """
    Generalized Pipecat FrameProcessor that injects context retrieved from 
    the session-level Qdrant vector database into the LLM context frame.
    """

    def __init__(
        self,
        rag_service: RAGService,
        max_context_chars: int = 1500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._rag = rag_service
        self._max_chars = max_context_chars
        logger.info("[RAGInjector] Initialized | max_context_chars={}", max_context_chars)

    def _extract_user_text(self, messages: list) -> Optional[str]:
        """Extract the latest user message text from the context messages."""
        if not messages:
            return None

        # Search backward for the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    return " ".join(text_parts).strip()
        return None

    def _strip_old_context(self, messages: list) -> None:
        """Remove previously injected RAG context to keep the message list clean."""
        messages[:] = [msg for msg in messages if not (msg.get("role") == "system" and CONTEXT_MARKER in (msg.get("content") or ""))]

    def _format_context(self, chunks: list[str]) -> str:
        """Format retrieved text chunks as a bulleted list within character constraints."""
        combined = ""
        for chunk in chunks:
            entry = f"- {chunk.strip()}\n"
            if len(combined) + len(entry) > self._max_chars:
                break
            combined += entry
        return combined.strip()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            messages = frame.context.get_messages()
            user_text = self._extract_user_text(messages)

            # Skip retrieval for extremely short inputs (filler words/greetings)
            if user_text and len(user_text.strip()) > 3:
                logger.debug("[RAGInjector] Retrieving context for query: '{}'", user_text[:60])
                
                # Fetch chunks from the dynamic RAG service
                chunks = await self._rag.retrieve(user_text)

                if chunks:
                    self._strip_old_context(messages)
                    context_text = self._format_context(chunks)
                    
                    context_msg = {
                        "role": "system",
                        "content": (
                            f"{CONTEXT_MARKER}\n"
                            f"The user is asking: \"{user_text}\"\n\n"
                            f"You MUST use the facts below to help formulate your answer. "
                            f"If the answer can be derived from these facts, prioritize them over general knowledge. "
                            f"If the facts are irrelevant, proceed naturally, but stay aligned with any documents provided.\n\n"
                            f"{context_text}\n"
                            f"[END DOCUMENT CONTEXT]"
                        ),
                    }

                    # Insert context message just before the latest user message
                    insert_idx = len(messages) - 1
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "user":
                            insert_idx = i
                            break

                    messages.insert(insert_idx, context_msg)
                    frame.context.set_messages(messages)
                    
                    logger.info("[RAGInjector] Injected {} chars of context ({} chunks)", len(context_text), len(chunks))
                else:
                    logger.debug("[RAGInjector] No context chunks found for query")

            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
