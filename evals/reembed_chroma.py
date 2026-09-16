"""
One-time full rebuild of the Consiva Chroma vector database.

Purpose:
    Rebuild data/vectorstore from the current production baseline:
        data/structured_final/

Important:
    - This is NOT part of the production ingestion pipeline.
    - It is a maintenance/rebuild utility.
    - It reuses the exact chunking, embedding, metadata, and Chroma
      functions from ingestion/vector_updater.py.
    - The resulting database uses the same collection and path as production.

Run from project root:

    python evals/reembed_chroma.py
"""

from pathlib import Path
import sys
import shutil
import logging


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PRODUCTION FUNCTIONS
# ============================================================

from ingestion.vector_updater import (
    create_embedding_model,
    create_vectorstore,
    load_and_chunk_page,
    calculate_content_hash,
    build_chunk_id,
    build_metadata,
    embed_chunks,
    upsert_vectors,
    COLLECTION_NAME,
    VECTORSTORE_DIR,
)


# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURED_DIR = PROJECT_ROOT / "data" / "structured_final"

EXPECTED_COLLECTION = COLLECTION_NAME


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# VALIDATION
# ============================================================

def validate_project():
    """Validate that the required project structure exists."""

    logger.info("Validating project...")

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"Project root not found: {PROJECT_ROOT}"
        )

    if not STRUCTURED_DIR.exists():
        raise FileNotFoundError(
            f"Structured data directory not found: {STRUCTURED_DIR}"
        )

    files = sorted(STRUCTURED_DIR.glob("*.txt"))

    if not files:
        raise ValueError(
            f"No .txt files found in: {STRUCTURED_DIR}"
        )

    logger.info(
        f"Found {len(files)} structured pages."
    )

    return files


def validate_vectorstore_is_empty():
    """
    Make sure we don't accidentally overwrite an existing database.

    The old vectorstore should already have been deleted manually.
    """

    if not VECTORSTORE_DIR.exists():
        logger.info(
            "Vectorstore directory does not exist. "
            "A new database will be created."
        )
        return

    contents = list(VECTORSTORE_DIR.iterdir())

    if contents:
        raise RuntimeError(
            "\n"
            "Vectorstore directory is not empty:\n"
            f"  {VECTORSTORE_DIR}\n\n"
            "For safety, this rebuild script will NOT overwrite it.\n"
            "Delete the existing vectorstore directory first, then rerun."
        )

    logger.info(
        "Vectorstore directory exists but is empty."
    )


# ============================================================
# BUILD DOCUMENTS
# ============================================================

def build_all_vectors(page_files):
    """
    Chunk every production baseline page and prepare:

        IDs
        documents
        metadata

    Uses the exact production chunker and metadata format.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 1: CHUNKING PRODUCTION BASELINE")
    logger.info("=" * 70)

    all_ids = []
    all_documents = []
    all_metadatas = []

    page_chunk_counts = {}

    for page_number, file_path in enumerate(page_files, start=1):

        logger.info(
            f"[{page_number}/{len(page_files)}] "
            f"Processing: {file_path.name}"
        )

        try:
            chunks = load_and_chunk_page(file_path)

            if not chunks:
                raise ValueError(
                    f"No valid chunks generated for {file_path.name}"
                )

            content_hash = calculate_content_hash(file_path)

            page_id = file_path.stem
            source = file_path.name

            page_ids = []

            for index, chunk in enumerate(chunks, start=1):

                content = str(chunk["content"]).strip()

                if not content:
                    continue

                chunk_id = build_chunk_id(
                    page_id=page_id,
                    index=index,
                )

                metadata = build_metadata(
                    chunk=chunk,
                    chunk_id=chunk_id,
                    page_id=page_id,
                    source=source,
                    content_hash=content_hash,
                )

                all_ids.append(chunk_id)
                all_documents.append(content)
                all_metadatas.append(metadata)

                page_ids.append(chunk_id)

            if not page_ids:
                raise ValueError(
                    f"No usable chunks generated for {file_path.name}"
                )

            page_chunk_counts[source] = len(page_ids)

            logger.info(
                f"  → {len(page_ids)} chunks"
            )

        except Exception as exc:
            raise RuntimeError(
                f"Failed while processing {file_path.name}: {exc}"
            ) from exc

    logger.info("")
    logger.info(
        f"Total pages processed: {len(page_files)}"
    )

    logger.info(
        f"Total chunks created: {len(all_ids)}"
    )

    return (
        all_ids,
        all_documents,
        all_metadatas,
        page_chunk_counts,
    )


# ============================================================
# VALIDATE PREPARED DATA
# ============================================================

def validate_prepared_data(
    ids,
    documents,
    metadatas,
):
    """Validate all vectors before embedding."""

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: VALIDATING CHUNKS")
    logger.info("=" * 70)

    if not ids:
        raise ValueError("No chunk IDs generated.")

    if not documents:
        raise ValueError("No documents generated.")

    if not metadatas:
        raise ValueError("No metadata generated.")

    if not (
        len(ids)
        == len(documents)
        == len(metadatas)
    ):
        raise ValueError(
            "IDs, documents and metadata counts do not match:\n"
            f"IDs: {len(ids)}\n"
            f"Documents: {len(documents)}\n"
            f"Metadata: {len(metadatas)}"
        )

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate chunk IDs detected."
        )

    for index, chunk_id in enumerate(ids):

        if not chunk_id:
            raise ValueError(
                f"Empty chunk ID at index {index}."
            )

        if not documents[index]:
            raise ValueError(
                f"Empty document for chunk {chunk_id}."
            )

        metadata = metadatas[index]

        required_fields = [
            "chunk_id",
            "page_id",
            "source",
            "content_hash",
        ]

        for field in required_fields:
            if field not in metadata:
                raise ValueError(
                    f"Missing metadata field '{field}' "
                    f"for chunk {chunk_id}"
                )

    logger.info(
        f"Validation successful: {len(ids)} chunks."
    )


# ============================================================
# EMBEDDING
# ============================================================

def create_all_embeddings(
    embedding_model,
    documents,
):
    """
    Create embeddings using the exact production embedding
    function from vector_updater.py.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: CREATING BGE-M3 EMBEDDINGS")
    logger.info("=" * 70)

    logger.info(
        "Embedding model: BAAI/bge-m3"
    )

    logger.info(
        f"Documents to embed: {len(documents)}"
    )

    embeddings = embed_chunks(
        embedding_model=embedding_model,
        chunks=[
            {"content": document}
            for document in documents
        ],
    )

    if not embeddings:
        raise RuntimeError(
            "Embedding generation returned no embeddings."
        )

    if len(embeddings) != len(documents):
        raise RuntimeError(
            "Embedding count does not match document count:\n"
            f"Documents: {len(documents)}\n"
            f"Embeddings: {len(embeddings)}"
        )

    logger.info(
        f"Successfully generated {len(embeddings)} embeddings."
    )

    return embeddings


# ============================================================
# INSERT INTO CHROMA
# ============================================================

def insert_into_chroma(
    vectorstore,
    ids,
    documents,
    metadatas,
    embeddings,
):
    """
    Insert all vectors into the exact production Chroma
    collection using the production upsert function.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 4: INSERTING INTO CHROMA")
    logger.info("=" * 70)

    logger.info(
        f"Collection: {EXPECTED_COLLECTION}"
    )

    logger.info(
        f"Database: {VECTORSTORE_DIR}"
    )

    logger.info(
        f"Vectors to insert: {len(ids)}"
    )

    upsert_vectors(
        vectorstore=vectorstore,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    logger.info(
        "Chroma upsert completed."
    )


# ============================================================
# VERIFY DATABASE
# ============================================================

def verify_chroma(
    vectorstore,
    expected_ids,
    expected_page_counts,
):
    """
    Verify that the rebuilt Chroma database contains exactly
    the expected vectors and page coverage.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 5: VERIFYING CHROMA DATABASE")
    logger.info("=" * 70)

    collection = vectorstore._collection

    actual_count = collection.count()
    expected_count = len(expected_ids)

    logger.info(
        f"Expected vectors: {expected_count}"
    )

    logger.info(
        f"Actual vectors:   {actual_count}"
    )

    if actual_count != expected_count:
        raise RuntimeError(
            "Vector count mismatch:\n"
            f"Expected: {expected_count}\n"
            f"Actual:   {actual_count}"
        )

    # --------------------------------------------------------
    # Get stored IDs and metadata
    # --------------------------------------------------------

    stored = collection.get(
        include=["metadatas"]
    )

    stored_ids = set(stored.get("ids", []))
    expected_id_set = set(expected_ids)

    missing_ids = expected_id_set - stored_ids
    extra_ids = stored_ids - expected_id_set

    if missing_ids:
        sample = sorted(missing_ids)[:10]

        raise RuntimeError(
            "Missing vector IDs detected.\n"
            f"Sample: {sample}"
        )

    if extra_ids:
        sample = sorted(extra_ids)[:10]

        raise RuntimeError(
            "Unexpected vector IDs detected.\n"
            f"Sample: {sample}"
        )

    # --------------------------------------------------------
    # Verify page coverage
    # --------------------------------------------------------

    actual_page_counts = {}

    for metadata in stored.get("metadatas", []):

        if not metadata:
            raise RuntimeError(
                "A stored vector has no metadata."
            )

        source = metadata.get("source")

        if not source:
            raise RuntimeError(
                "A stored vector has no source metadata."
            )

        actual_page_counts[source] = (
            actual_page_counts.get(source, 0) + 1
        )

    if actual_page_counts != expected_page_counts:

        logger.error(
            f"Expected page counts: {expected_page_counts}"
        )

        logger.error(
            f"Actual page counts: {actual_page_counts}"
        )

        raise RuntimeError(
            "Page/chunk coverage verification failed."
        )

    logger.info(
        f"Verified {len(stored_ids)} unique vector IDs."
    )

    logger.info(
        f"Verified {len(actual_page_counts)} source pages."
    )

    logger.info(
        "Chroma verification successful."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("")
    logger.info("=" * 70)
    logger.info("CONSIVA CHROMA FULL REBUILD")
    logger.info("=" * 70)
    logger.info("")

    logger.info(
        f"Project root: {PROJECT_ROOT}"
    )

    logger.info(
        f"Source:       {STRUCTURED_DIR}"
    )

    logger.info(
        f"Destination:  {VECTORSTORE_DIR}"
    )

    logger.info(
        f"Collection:   {EXPECTED_COLLECTION}"
    )

    logger.info("")

    # --------------------------------------------------------
    # 1. Validate source files
    # --------------------------------------------------------

    page_files = validate_project()

    # --------------------------------------------------------
    # 2. Safety check
    # --------------------------------------------------------

    validate_vectorstore_is_empty()

    # --------------------------------------------------------
    # 3. Build chunks + metadata
    # --------------------------------------------------------

    (
        ids,
        documents,
        metadatas,
        page_chunk_counts,
    ) = build_all_vectors(page_files)

    # --------------------------------------------------------
    # 4. Validate everything before embedding
    # --------------------------------------------------------

    validate_prepared_data(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    # --------------------------------------------------------
    # 5. Load exact production embedding model
    # --------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("LOADING EMBEDDING MODEL")
    logger.info("=" * 70)

    embedding_model = create_embedding_model()

    # --------------------------------------------------------
    # 6. Create embeddings
    # --------------------------------------------------------

    embeddings = create_all_embeddings(
        embedding_model=embedding_model,
        documents=documents,
    )

    # --------------------------------------------------------
    # 7. Create production Chroma database
    # --------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("CREATING CHROMA DATABASE")
    logger.info("=" * 70)

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore = create_vectorstore(
        embedding_model=embedding_model,
    )

    # --------------------------------------------------------
    # 8. Insert vectors
    # --------------------------------------------------------

    insert_into_chroma(
        vectorstore=vectorstore,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    # --------------------------------------------------------
    # 9. Verify database
    # --------------------------------------------------------

    verify_chroma(
        vectorstore=vectorstore,
        expected_ids=ids,
        expected_page_counts=page_chunk_counts,
    )

    # --------------------------------------------------------
    # 10. Final result
    # --------------------------------------------------------

    final_count = vectorstore._collection.count()

    logger.info("")
    logger.info("=" * 70)
    logger.info("REBUILD COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info("")
    logger.info(
        f"Pages:        {len(page_files)}"
    )
    logger.info(
        f"Vectors:      {final_count}"
    )
    logger.info(
        f"Collection:   {EXPECTED_COLLECTION}"
    )
    logger.info(
        f"Vectorstore:  {VECTORSTORE_DIR}"
    )
    logger.info("")
    logger.info(
        "The rebuilt database is ready for the existing RAG system."
    )
    logger.info(
        "The production ingestion pipeline was not modified."
    )
    logger.info("")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        logger.error(
            "\nRebuild interrupted by user."
        )
        sys.exit(1)

    except Exception as exc:
        logger.error("")
        logger.error("=" * 70)
        logger.error("REBUILD FAILED")
        logger.error("=" * 70)
        logger.error(str(exc))
        logger.error("")

        logger.error(
            "If a partial vectorstore was created, "
            "delete data/vectorstore and rerun the rebuild."
        )

        sys.exit(1)