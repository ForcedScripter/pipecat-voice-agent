import asyncio
import json
from typing import Optional
from loguru import logger
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Shared thread-safe in-memory QdrantClient instance to avoid file locks
_GLOBAL_QDRANT_CLIENT = None
_GLOBAL_EMBED_MODEL = None

def get_shared_qdrant_client() -> QdrantClient:
    global _GLOBAL_QDRANT_CLIENT
    if _GLOBAL_QDRANT_CLIENT is None:
        logger.info("[RAG] Initializing shared in-memory QdrantClient")
        _GLOBAL_QDRANT_CLIENT = QdrantClient(location=":memory:")
    return _GLOBAL_QDRANT_CLIENT

def get_shared_embed_model(model_name: str) -> SentenceTransformer:
    global _GLOBAL_EMBED_MODEL
    if _GLOBAL_EMBED_MODEL is None:
        logger.info("[RAG] Loading shared SentenceTransformer model | {}", model_name)
        _GLOBAL_EMBED_MODEL = SentenceTransformer(model_name)
    return _GLOBAL_EMBED_MODEL

class RAGService:
    """Dynamic RAG service supporting Qdrant session-level vector collections."""

    def __init__(
        self,
        qdrant_path: str,
        collection_name: str,
        embed_model_name: str = "BAAI/bge-base-en-v1.5",
        top_k: int = 4,
        score_threshold: float = 0.25,
    ):
        logger.info(
            "[RAG] Initializing | collection={} model={} top_k={} threshold={}",
            collection_name,
            embed_model_name,
            top_k,
            score_threshold,
        )

        self._collection = collection_name
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._qdrant_path = qdrant_path

        # Connect to shared in-memory Qdrant Client to avoid locking
        self._client = get_shared_qdrant_client()
        
        # Load shared embedding model (free, offline, no API key needed)
        self._embed_model = get_shared_embed_model(embed_model_name)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed list of query strings (synchronous, runs in thread pool)."""
        return self._embed_model.encode(texts, normalize_embeddings=True).tolist()

    def _search_sync(self, embedding: list[float]) -> list:
        """Search Qdrant for similar vectors (synchronous, runs in thread pool)."""
        try:
            if not self._client.collection_exists(self._collection):
                return []
            results = self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=self._top_k,
                score_threshold=self._score_threshold,
                with_payload=True,
            )
            return results.points
        except Exception as e:
            logger.warning("[RAG] Query failed for '{}': {}", self._collection, e)
            return []

    @staticmethod
    def _extract_text(payload: dict) -> str:
        """Extract text from Qdrant payload."""
        node_content = payload.get("_node_content", "")
        if node_content:
            try:
                node = json.loads(node_content)
                text = node.get("text", "").strip()
                if text:
                    return text
            except (json.JSONDecodeError, TypeError):
                pass
        return payload.get("text", "").strip()

    async def retrieve(self, query: str) -> list[str]:
        """Retrieve relevant text chunks for a query (runs in thread pool)."""
        try:
            # Check dynamically if collection exists
            exists = await asyncio.to_thread(self._client.collection_exists, self._collection)
            if not exists:
                logger.debug("[RAG] Collection '{}' does not exist yet. Skipping retrieval.", self._collection)
                return []

            # Embed the query
            embeddings = await asyncio.to_thread(self._embed_sync, [query])
            query_vector = embeddings[0]

            # Search in database
            results = await asyncio.to_thread(self._search_sync, query_vector)

            if not results:
                logger.debug("[RAG] No results found in '{}' for: '{}'", self._collection, query[:80])
                return []

            chunks = []
            for hit in results:
                text = self._extract_text(hit.payload)
                if text:
                    chunks.append(text)
                    logger.debug("[RAG] Hit score={:.3f} | '{}'", hit.score, text[:80])

            logger.info("[RAG] Retrieved {} chunks from '{}' for: '{}'", len(chunks), self._collection, query[:60])
            return chunks

        except Exception as e:
            logger.error("[RAG] Retrieval failed: {}", e)
            return []

    async def index_documents(self, documents: list[str]):
        """Index a list of text chunks dynamically into Qdrant (runs in thread pool)."""
        await asyncio.to_thread(self._index_sync, documents)

    def _index_sync(self, documents: list[str]):
        """Synchronous indexing implementation."""
        logger.info("[RAG] Creating collection and embedding {} chunks for '{}'", len(documents), self._collection)
        
        # 1. Create or recreate collection
        try:
            if self._client.collection_exists(self._collection):
                self._client.delete_collection(self._collection)
        except Exception:
            pass

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
            ),
        )

        # 2. Embed all chunks
        embeddings = self._embed_sync(documents)

        # 3. Upsert vectors
        points = [
            models.PointStruct(
                id=i,
                vector=embeddings[i],
                payload={"text": documents[i]},
            )
            for i in range(len(documents))
        ]
        
        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )
        logger.info("[RAG] Indexing complete. Collection '{}' is now live.", self._collection)

    def delete_collection(self):
        """Purge the session collection from the local database."""
        try:
            if self._client.collection_exists(self._collection):
                self._client.delete_collection(self._collection)
                logger.info("[RAG] Temporary collection '{}' successfully deleted", self._collection)
        except Exception as e:
            logger.warning("[RAG] Failed to delete collection '{}': {}", self._collection, e)

    def close(self):
        """Close Qdrant client connection (no-op for shared client)."""
        pass

