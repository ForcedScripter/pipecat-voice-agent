from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
)

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)

from llama_index.core.ingestion import (
    IngestionPipeline,
)

from llama_index.vector_stores.qdrant import (
    QdrantVectorStore,
)

from ingestion.config import (
    COLLECTION_NAME,
    EMBED_MODEL,
    QDRANT_PATH,
)

from ingestion.parser import (
    StructuralDocxReader,
)

from ingestion.chunker import (
    SimpleChunker,
)

from ingestion.metadata import (
    RestaurantMetadataExtractor,
)


# =========================================================
# CREATE COLLECTION
# =========================================================

def create_collection(client):

    collections = client.get_collections().collections

    names = [c.name for c in collections]

    if COLLECTION_NAME in names:
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE,
        ),
    )


# =========================================================
# BUILD INDEX
# =========================================================

def build_index(docs_dir="./docs"):

    client = QdrantClient(
        path=QDRANT_PATH
    )

    create_collection(client)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    # =====================================================
    # LOAD DOCUMENTS
    # =====================================================

    reader = StructuralDocxReader()

    all_docs = []

    docs_path = Path(docs_dir)

    print("Looking for docs in:", docs_path.resolve())

    for docx_path in docs_path.rglob("*.docx"):

        print("Loading:", docx_path)

        docs = reader.load_data(docx_path)

        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} document blocks")

    # =====================================================
    # INGESTION PIPELINE
    # =====================================================

    pipeline = IngestionPipeline(
        transformations=[

            # chunking
            SimpleChunker(),

            # metadata extraction
            RestaurantMetadataExtractor(),

            # embeddings
            EMBED_MODEL,
        ],
        vector_store=vector_store,
    )

    pipeline.run(
        documents=all_docs,
        show_progress=True,
    )

    print("Ingestion complete.")

    # =====================================================
    # RETURN INDEX
    # =====================================================

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=EMBED_MODEL,
    )


if __name__ == "__main__":
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    build_index(str(docs_dir))