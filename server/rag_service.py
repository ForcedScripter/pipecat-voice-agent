"""
RAGService
──────────
Async retrieval-only service for the pipecat voice pipeline.

Loads a pre-built Qdrant index and a local embedding model once at init.
Exposes a single `retrieve(query)` method that returns relevant text chunks.

All blocking work (embedding + Qdrant search) runs in `asyncio.to_thread()`
so the pipecat event loop is never blocked.

Usage:
    rag = RAGService(qdrant_path="./qdrant_storage", ...)
    chunks = await rag.retrieve("what soups do you have?")
"""

import asyncio
import json
from typing import Optional

from loguru import logger
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


class RAGService:
    """Async wrapper around Qdrant vector search + local embedding model."""

    def __init__(
        self,
        qdrant_path: str,
        collection_name: str,
        embed_model_name: str = "BAAI/bge-base-en-v1.5",
        top_k: int = 5,
        score_threshold: float = 0.3,
    ):
        logger.info(
            "[RAG] Initializing | qdrant={} collection={} model={} top_k={} threshold={}",
            qdrant_path,
            collection_name,
            embed_model_name,
            top_k,
            score_threshold,
        )

        self._collection = collection_name
        self._top_k = top_k
        self._score_threshold = score_threshold

        # -- Load Qdrant client (local file-based storage)
        self._client = QdrantClient(path=qdrant_path)
        info = self._client.get_collection(collection_name)
        logger.info(
            "[RAG] Qdrant loaded | {} points in '{}'",
            info.points_count,
            collection_name,
        )

        # -- Load embedding model (one-time ~2-3s cost)
        self._embed_model = SentenceTransformer(embed_model_name)
        logger.info("[RAG] Embedding model loaded | {}", embed_model_name)

    def _embed_sync(self, text: str) -> list[float]:
        """Embed a single query string (synchronous, runs in thread pool)."""
        return self._embed_model.encode(text, normalize_embeddings=True).tolist()

    def _search_sync(self, embedding: list[float]) -> list:
        """Search Qdrant for similar vectors (synchronous, runs in thread pool)."""
        results = self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=self._top_k,
            score_threshold=self._score_threshold,
            with_payload=True,
        )
        return results.points

    @staticmethod
    def _extract_text(payload: dict) -> str:
        """
        Extract the actual text content from a Qdrant point payload.

        LlamaIndex stores the text inside `_node_content` as a JSON string
        with a `text` field. We parse that. Falls back to the `section` or
        `doc_type` metadata fields if parsing fails.
        """
        node_content = payload.get("_node_content", "")
        if node_content:
            try:
                node = json.loads(node_content)
                text = node.get("text", "").strip()
                if text:
                    return text
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: use section name or raw payload
        return payload.get("section", "").strip()

    async def retrieve(self, query: str) -> list[str]:
        """
        Retrieve relevant text chunks for a query.

        Returns a list of text strings (empty list if nothing relevant found).
        Runs embedding + search in thread pool — non-blocking.
        """
        try:
            # Embed in thread pool (sentence-transformers is synchronous)
            embedding = await asyncio.to_thread(self._embed_sync, query)

            # Search Qdrant in thread pool
            results = await asyncio.to_thread(self._search_sync, embedding)

            if not results:
                logger.debug("[RAG] No results for query: '{}'", query[:80])
                return []

            chunks = []
            for hit in results:
                text = self._extract_text(hit.payload)
                if text:
                    chunks.append(text)
                    logger.debug(
                        "[RAG] Hit score={:.3f} | '{}'",
                        hit.score,
                        text[:80],
                    )

            logger.info(
                "[RAG] Retrieved {} chunks for query: '{}'",
                len(chunks),
                query[:60],
            )
            return chunks

        except Exception as e:
            logger.error("[RAG] Retrieval failed: {}", e)
            return []

    def close(self):
        """Close the Qdrant client (cleanup on shutdown)."""
        try:
            self._client.close()
            logger.info("[RAG] Qdrant client closed")
        except Exception:
            pass
