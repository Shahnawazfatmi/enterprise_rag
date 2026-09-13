from pathlib import Path
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# IMPORTANT:
# Reuse the EXACT existing chunking logic from the current RAG.
from chunker.chunking import process_file

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PATHS & CONFIGURATION
# ============================================================


NEW_STRUCTURED_DIR = BASE_DIR / "data" / "new_structured"
CHANGES_FILE = BASE_DIR / "data" / "comparison" / "changes.json"

VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

COLLECTION_NAME = "consiva_knowledge"

MODEL_NAME = "BAAI/bge-m3"

# Embedding batch size.
EMBED_BATCH_SIZE = 32

# Chroma upsert/delete batch size.
CHROMA_BATCH_SIZE = 100


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# CONTENT NORMALIZATION
# ============================================================

def normalize_content(text: str) -> str:
    """
    Normalize content exactly enough to produce stable hashes.

    This should remain consistent with compare.py.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    normalized_lines = []

    for line in text.split("\n"):
        line = line.strip()

        # Collapse repeated spaces/tabs.
        line = re.sub(r"[ \t]+", " ", line)

        # Ignore empty lines.
        if line:
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def calculate_content_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash using the same normalization
    strategy used by compare.py.
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    normalized = normalize_content(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# CHANGE FILE
# ============================================================

def load_changes() -> dict:
    """
    Load and validate changes.json.

    Expected structure:

    {
        "pages": {
            "new": [...],
            "changed": [...],
            "unchanged": [...],
            "deleted": [...]
        },
        "details": {
            "page_id": {
                "status": "...",
                "baseline": {...},
                "new": {...}
            }
        }
    }
    """

    if not CHANGES_FILE.exists():
        raise FileNotFoundError(
            f"Changes file not found: {CHANGES_FILE}"
        )

    try:
        with CHANGES_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            changes = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in changes file: {error}"
        ) from error

    if not isinstance(changes, dict):
        raise ValueError(
            "changes.json must contain a JSON object."
        )

    pages = changes.get("pages")

    if not isinstance(pages, dict):
        raise ValueError(
            "Invalid changes.json: 'pages' must be an object."
        )

    details = changes.get("details")

    if not isinstance(details, dict):
        raise ValueError(
            "Invalid changes.json: 'details' must be an object."
        )

    for key in ("new", "changed", "unchanged", "deleted"):
        if key not in pages:
            raise ValueError(
                f"Invalid changes.json: missing pages.{key}"
            )

        if not isinstance(pages[key], list):
            raise ValueError(
                f"Invalid changes.json: pages.{key} must be a list."
            )

    return changes


# ============================================================
# ACTION EXTRACTION
# ============================================================

def build_actions(changes: dict) -> list[dict]:
    """
    Convert changes.json into normalized update actions.

    Example:

    {
        "page_id": "consiva.ai_pricing",
        "status": "CHANGED",
        "filename": "consiva.ai_pricing.txt"
    }
    """

    pages = changes["pages"]
    details = changes["details"]

    actions = []

    # --------------------------------------------------------
    # NEW
    # --------------------------------------------------------

    for page_id in pages["new"]:

        if page_id not in details:
            raise ValueError(
                f"Missing details for NEW page: {page_id}"
            )

        detail = details[page_id]

        new_data = detail.get("new")

        if not isinstance(new_data, dict):
            raise ValueError(
                f"Missing 'new' details for page: {page_id}"
            )

        filename = new_data.get("file_name")

        if not filename:
            raise ValueError(
                f"Missing new.file_name for page: {page_id}"
            )

        actions.append(
            {
                "page_id": page_id,
                "status": "NEW",
                "filename": filename,
            }
        )

    # --------------------------------------------------------
    # CHANGED
    # --------------------------------------------------------

    for page_id in pages["changed"]:

        if page_id not in details:
            raise ValueError(
                f"Missing details for CHANGED page: {page_id}"
            )

        detail = details[page_id]

        new_data = detail.get("new")

        if not isinstance(new_data, dict):
            raise ValueError(
                f"Missing 'new' details for page: {page_id}"
            )

        filename = new_data.get("file_name")

        if not filename:
            raise ValueError(
                f"Missing new.file_name for page: {page_id}"
            )

        actions.append(
            {
                "page_id": page_id,
                "status": "CHANGED",
                "filename": filename,
            }
        )

    # --------------------------------------------------------
    # DELETED
    # --------------------------------------------------------

    for page_id in pages["deleted"]:

        if page_id not in details:
            raise ValueError(
                f"Missing details for DELETED page: {page_id}"
            )

        detail = details[page_id]

        baseline_data = detail.get("baseline")

        if not isinstance(baseline_data, dict):
            raise ValueError(
                f"Missing 'baseline' details for deleted page: {page_id}"
            )

        filename = baseline_data.get("file_name")

        if not filename:
            raise ValueError(
                f"Missing baseline.file_name for deleted page: {page_id}"
            )

        actions.append(
            {
                "page_id": page_id,
                "status": "DELETED",
                "filename": filename,
            }
        )

    return actions


# ============================================================
# EMBEDDING MODEL
# ============================================================

def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create the SAME embedding model/configuration used
    by the existing embedder.py.
    """

    logger.info(
        "Loading embedding model: %s",
        MODEL_NAME,
    )

    model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    logger.info("Embedding model loaded successfully.")

    return model


# ============================================================
# VECTOR STORE
# ============================================================

def create_vectorstore(
    embedding_model: HuggingFaceEmbeddings,
) -> Chroma:
    """
    Open the existing persistent Chroma collection.

    This does NOT create a new collection.
    """

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embedding_model,
    )

    return vectorstore


# ============================================================
# CHUNKING
# ============================================================

def load_and_chunk_page(
    file_path: Path,
) -> list[dict]:
    """
    Use the EXACT existing RAG chunking implementation.

    This calls chunker/chunking.py -> process_file()
    so the nightly updater cannot accidentally develop
    different chunking behavior from the existing RAG.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Structured page not found: {file_path}"
        )

    logger.info(
        "Chunking: %s",
        file_path.name,
    )

    chunks = process_file(file_path)

    if not chunks:
        raise ValueError(
            f"No chunks generated for: {file_path.name}"
        )

    for index, chunk in enumerate(chunks, start=1):

        if not isinstance(chunk, dict):
            raise ValueError(
                f"Invalid chunk generated for {file_path.name}"
            )

        content = chunk.get("content")

        if not content or not str(content).strip():
            raise ValueError(
                f"Empty chunk detected in {file_path.name}"
            )

    logger.info(
        "Generated %d chunks: %s",
        len(chunks),
        file_path.name,
    )

    return chunks


# ============================================================
# CHUNK IDS
# ============================================================

def build_chunk_id(
    page_id: str,
    index: int,
) -> str:
    """
    Generate deterministic page-specific chunk IDs.

    Example:

        consiva.ai_pricing::chunk_000001
        consiva.ai_pricing::chunk_000002

    This is safer than the old global chunk_000001 scheme
    because each page owns its own chunk namespace.
    """

    return f"{page_id}::chunk_{index:06d}"


# ============================================================
# METADATA
# ============================================================

def build_metadata(
    chunk: dict,
    chunk_id: str,
    page_id: str,
    source: str,
    content_hash: str,
) -> dict:
    """
    Build Chroma metadata.

    Existing RAG metadata fields are preserved.
    Additional page_id/content_hash fields are useful
    for future maintenance and auditing.
    """

    return {
        "chunk_id": chunk_id,
        "page_id": page_id,
        "source": source,
        "content_hash": content_hash,

        "H1": chunk.get("H1") or "",
        "H2": chunk.get("H2") or "",
        "H3": chunk.get("H3") or "",
        "H4": chunk.get("H4") or "",
        "H5": chunk.get("H5") or "",
        "H6": chunk.get("H6") or "",
    }


# ============================================================
# EMBEDDING
# ============================================================

def embed_chunks(
    embedding_model: HuggingFaceEmbeddings,
    chunks: list[dict],
) -> list[list[float]]:
    """
    Embed chunks in batches.

    Embeddings are generated BEFORE deleting old vectors.
    This protects the existing data if embedding fails.
    """

    texts = [
        str(chunk["content"])
        for chunk in chunks
    ]

    all_embeddings = []

    total = len(texts)

    for start in range(
        0,
        total,
        EMBED_BATCH_SIZE,
    ):

        end = min(
            start + EMBED_BATCH_SIZE,
            total,
        )

        batch_texts = texts[start:end]

        logger.info(
            "Embedding chunks %d-%d/%d",
            start + 1,
            end,
            total,
        )

        batch_embeddings = (
            embedding_model.embed_documents(
                batch_texts
            )
        )

        if len(batch_embeddings) != len(batch_texts):
            raise ValueError(
                "Embedding count mismatch: "
                f"expected {len(batch_texts)}, "
                f"received {len(batch_embeddings)}"
            )

        all_embeddings.extend(
            batch_embeddings
        )

    if len(all_embeddings) != len(chunks):
        raise ValueError(
            "Final embedding count mismatch: "
            f"chunks={len(chunks)}, "
            f"embeddings={len(all_embeddings)}"
        )

    if not all_embeddings:
        raise ValueError(
            "No embeddings were generated."
        )

    return all_embeddings


# ============================================================
# EXISTING VECTOR IDS
# ============================================================

def get_existing_ids_for_source(
    vectorstore: Chroma,
    source: str,
) -> list[str]:
    """
    Find every vector belonging to a page.

    Source remains the filename because that matches
    the existing RAG metadata structure.
    """

    result = vectorstore._collection.get(
        where={
            "source": source,
        },
        include=[],
    )

    ids = result.get("ids", [])

    if ids is None:
        return []

    return list(ids)


# ============================================================
# CHROMA UPSERT
# ============================================================

def upsert_vectors(
    vectorstore: Chroma,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    embeddings: list[list[float]],
) -> None:
    """
    Upsert vectors in safe batches.
    """

    if not (
        len(ids)
        == len(documents)
        == len(metadatas)
        == len(embeddings)
    ):
        raise ValueError(
            "Vector data length mismatch."
        )

    total = len(ids)

    for start in range(
        0,
        total,
        CHROMA_BATCH_SIZE,
    ):

        end = min(
            start + CHROMA_BATCH_SIZE,
            total,
        )

        logger.info(
            "Upserting vectors %d-%d/%d",
            start + 1,
            end,
            total,
        )

        vectorstore._collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )


# ============================================================
# CHROMA DELETE
# ============================================================

def delete_vectors(
    vectorstore: Chroma,
    ids: list[str],
) -> None:
    """
    Delete vectors in batches.
    """

    if not ids:
        return

    total = len(ids)

    for start in range(
        0,
        total,
        CHROMA_BATCH_SIZE,
    ):

        end = min(
            start + CHROMA_BATCH_SIZE,
            total,
        )

        batch_ids = ids[start:end]

        logger.info(
            "Deleting vectors %d-%d/%d",
            start + 1,
            end,
            total,
        )

        vectorstore._collection.delete(
            ids=batch_ids
        )


# ============================================================
# PROCESS NEW / CHANGED PAGE
# ============================================================

def update_page(
    vectorstore: Chroma,
    embedding_model: HuggingFaceEmbeddings,
    page_id: str,
    filename: str,
    status: str,
) -> dict:
    """
    Process a NEW or CHANGED page.

    IMPORTANT ORDER:

        Read
          ↓
        Chunk
          ↓
        Embed
          ↓
        Validate
          ↓
        Upsert new vectors
          ↓
        Verify new vectors
          ↓
        Delete old vectors

    This avoids creating a temporary period where the
    page has no vectors if embedding/upsert fails.
    """

    file_path = NEW_STRUCTURED_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"{status} page file not found: {file_path}"
        )

    logger.info("=" * 70)
    logger.info(
        "%s PAGE: %s",
        status,
        page_id,
    )
    logger.info(
        "File: %s",
        filename,
    )
    logger.info("=" * 70)

    # --------------------------------------------------------
    # 1. Hash new content
    # --------------------------------------------------------

    content_hash = calculate_content_hash(
        file_path
    )

    logger.info(
        "Content hash: %s",
        content_hash,
    )

    # --------------------------------------------------------
    # 2. Chunk new content
    # --------------------------------------------------------

    chunks = load_and_chunk_page(
        file_path
    )

    # --------------------------------------------------------
    # 3. Build deterministic IDs
    # --------------------------------------------------------

    new_ids = [
        build_chunk_id(
            page_id,
            index,
        )
        for index in range(
            1,
            len(chunks) + 1,
        )
    ]

    # --------------------------------------------------------
    # 4. Build documents
    # --------------------------------------------------------

    documents = [
        str(chunk["content"])
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # 5. Build metadata
    # --------------------------------------------------------

    metadatas = [
        build_metadata(
            chunk=chunk,
            chunk_id=chunk_id,
            page_id=page_id,
            source=filename,
            content_hash=content_hash,
        )
        for chunk, chunk_id in zip(
            chunks,
            new_ids,
        )
    ]

    # --------------------------------------------------------
    # 6. Generate embeddings
    # --------------------------------------------------------

    embeddings = embed_chunks(
        embedding_model,
        chunks,
    )

    # --------------------------------------------------------
    # 7. Validate everything BEFORE DB modification
    # --------------------------------------------------------

    if not (
        len(new_ids)
        == len(documents)
        == len(metadatas)
        == len(embeddings)
    ):
        raise ValueError(
            f"Prepared vector data mismatch for {page_id}"
        )

    logger.info(
        "Prepared %d vectors for %s",
        len(new_ids),
        page_id,
    )

    # --------------------------------------------------------
    # 8. Find existing vectors
    # --------------------------------------------------------

    old_ids = get_existing_ids_for_source(
        vectorstore,
        filename,
    )

    logger.info(
        "Existing vectors for page: %d",
        len(old_ids),
    )

    # --------------------------------------------------------
    # 9. UPSERT NEW VECTORS FIRST
    # --------------------------------------------------------

    upsert_vectors(
        vectorstore=vectorstore,
        ids=new_ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    # --------------------------------------------------------
    # 10. Verify new vectors exist
    # --------------------------------------------------------

    verification = vectorstore._collection.get(
        ids=new_ids,
        include=[],
    )

    verified_ids = set(
        verification.get("ids", [])
    )

    expected_ids = set(new_ids)

    missing_ids = expected_ids - verified_ids

    if missing_ids:
        raise RuntimeError(
            f"Vector verification failed for {page_id}. "
            f"Missing {len(missing_ids)} vectors."
        )

    logger.info(
        "Verified %d new vectors.",
        len(verified_ids),
    )

    # --------------------------------------------------------
    # 11. Delete OLD vectors
    # --------------------------------------------------------

    new_id_set = set(new_ids)

    obsolete_ids = [
        vector_id
        for vector_id in old_ids
        if vector_id not in new_id_set
    ]

    if obsolete_ids:

        logger.info(
            "Removing %d obsolete vectors.",
            len(obsolete_ids),
        )

        delete_vectors(
            vectorstore,
            obsolete_ids,
        )

    else:

        logger.info(
            "No obsolete vectors to remove."
        )

    # --------------------------------------------------------
    # 12. Final source verification
    # --------------------------------------------------------

    final_ids = get_existing_ids_for_source(
        vectorstore,
        filename,
    )

    final_id_set = set(final_ids)

    missing_after_update = (
        new_id_set - final_id_set
    )

    if missing_after_update:
        raise RuntimeError(
            f"Final verification failed for {page_id}. "
            f"Missing {len(missing_after_update)} vectors."
        )

    # Check that obsolete vectors are actually gone.
    remaining_obsolete = (
        set(obsolete_ids) & final_id_set
    )

    if remaining_obsolete:
        raise RuntimeError(
            f"Old vectors still remain for {page_id}: "
            f"{len(remaining_obsolete)}"
        )

    logger.info(
        "Successfully updated page: %s",
        page_id,
    )

    return {
        "page_id": page_id,
        "status": status,
        "filename": filename,
        "vectors": len(new_ids),
        "old_vectors": len(old_ids),
        "deleted_vectors": len(obsolete_ids),
        "content_hash": content_hash,
    }


# ============================================================
# PROCESS DELETED PAGE
# ============================================================

def delete_page(
    vectorstore: Chroma,
    page_id: str,
    filename: str,
) -> dict:
    """
    Remove every vector belonging to a deleted page.
    """

    logger.info("=" * 70)
    logger.info(
        "DELETED PAGE: %s",
        page_id,
    )
    logger.info(
        "Source: %s",
        filename,
    )
    logger.info("=" * 70)

    old_ids = get_existing_ids_for_source(
        vectorstore,
        filename,
    )

    if not old_ids:

        logger.info(
            "No vectors found for deleted page: %s",
            page_id,
        )

        return {
            "page_id": page_id,
            "status": "DELETED",
            "filename": filename,
            "deleted_vectors": 0,
        }

    logger.info(
        "Found %d vectors to delete.",
        len(old_ids),
    )

    delete_vectors(
        vectorstore,
        old_ids,
    )

    # Verify deletion.
    remaining_ids = get_existing_ids_for_source(
        vectorstore,
        filename,
    )

    if remaining_ids:
        raise RuntimeError(
            f"Deletion verification failed for {page_id}. "
            f"{len(remaining_ids)} vectors remain."
        )

    logger.info(
        "Successfully deleted page: %s",
        page_id,
    )

    return {
        "page_id": page_id,
        "status": "DELETED",
        "filename": filename,
        "deleted_vectors": len(old_ids),
    }


# ============================================================
# MAIN UPDATE PROCESS
# ============================================================

def run_vector_update() -> dict:
    """
    Execute the complete incremental vector update.
    """

    start_time = datetime.now(
        timezone.utc
    )

    logger.info("=" * 70)
    logger.info("STARTING INCREMENTAL VECTOR UPDATE")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Validate directories/files
    # --------------------------------------------------------

    if not NEW_STRUCTURED_DIR.exists():
        raise FileNotFoundError(
            f"New structured directory not found: "
            f"{NEW_STRUCTURED_DIR}"
        )

    # --------------------------------------------------------
    # Load changes
    # --------------------------------------------------------

    changes = load_changes()

    actions = build_actions(
        changes
    )

    summary = changes.get(
        "summary",
        {},
    )

    logger.info(
        "Comparison summary:"
    )

    logger.info(
        "  NEW       : %s",
        summary.get("new", len(changes["pages"]["new"])),
    )

    logger.info(
        "  CHANGED   : %s",
        summary.get(
            "changed",
            len(changes["pages"]["changed"]),
        ),
    )

    logger.info(
        "  UNCHANGED : %s",
        summary.get(
            "unchanged",
            len(changes["pages"]["unchanged"]),
        ),
    )

    logger.info(
        "  DELETED   : %s",
        summary.get(
            "deleted",
            len(changes["pages"]["deleted"]),
        ),
    )

    logger.info(
        "  ACTIONS   : %d",
        len(actions),
    )

    # --------------------------------------------------------
    # Nothing to update
    # --------------------------------------------------------

    if not actions:

        logger.info("=" * 70)
        logger.info(
            "NO VECTOR DATABASE CHANGES REQUIRED"
        )
        logger.info("=" * 70)

        return {
            "success": True,
            "new": 0,
            "changed": 0,
            "deleted": 0,
            "updated_pages": [],
            "duration_seconds": 0,
        }

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    embedding_model = (
        create_embedding_model()
    )

    # --------------------------------------------------------
    # Open vector DB
    # --------------------------------------------------------

    vectorstore = create_vectorstore(
        embedding_model
    )

    before_count = (
        vectorstore._collection.count()
    )

    logger.info(
        "Vector count before update: %d",
        before_count,
    )

    # --------------------------------------------------------
    # Process actions
    # --------------------------------------------------------

    successful_updates = []

    failed_updates = []

    for action in actions:

        page_id = action["page_id"]
        status = action["status"]
        filename = action["filename"]

        try:

            if status in (
                "NEW",
                "CHANGED",
            ):

                result = update_page(
                    vectorstore=vectorstore,
                    embedding_model=embedding_model,
                    page_id=page_id,
                    filename=filename,
                    status=status,
                )

            elif status == "DELETED":

                result = delete_page(
                    vectorstore=vectorstore,
                    page_id=page_id,
                    filename=filename,
                )

            else:

                raise ValueError(
                    f"Unsupported action status: {status}"
                )

            successful_updates.append(
                result
            )

        except Exception as error:

            logger.exception(
                "Failed to process page: %s",
                page_id,
            )

            failed_updates.append(
                {
                    "page_id": page_id,
                    "status": status,
                    "filename": filename,
                    "error": str(error),
                }
            )

    # --------------------------------------------------------
    # Final DB count
    # --------------------------------------------------------

    after_count = (
        vectorstore._collection.count()
    )

    duration = (
        datetime.now(timezone.utc)
        - start_time
    ).total_seconds()

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info(
        "INCREMENTAL VECTOR UPDATE COMPLETED"
    )
    logger.info("=" * 70)

    logger.info(
        "Successful pages : %d",
        len(successful_updates),
    )

    logger.info(
        "Failed pages     : %d",
        len(failed_updates),
    )

    logger.info(
        "Vectors before   : %d",
        before_count,
    )

    logger.info(
        "Vectors after    : %d",
        after_count,
    )

    logger.info(
        "Duration         : %.2f seconds",
        duration,
    )

    logger.info("=" * 70)

    # --------------------------------------------------------
    # Fail the pipeline if ANY page failed
    # --------------------------------------------------------

    if failed_updates:

        logger.error(
            "VECTOR UPDATE FAILED."
        )

        for failure in failed_updates:

            logger.error(
                "  %s [%s] -> %s",
                failure["page_id"],
                failure["status"],
                failure["error"],
            )

        return {
            "success": False,
            "new": len(changes["pages"]["new"]),
            "changed": len(changes["pages"]["changed"]),
            "deleted": len(changes["pages"]["deleted"]),
            "successful_pages": successful_updates,
            "failed_pages": failed_updates,
            "vectors_before": before_count,
            "vectors_after": after_count,
            "duration_seconds": duration,
        }

    return {
        "success": True,
        "new": len(changes["pages"]["new"]),
        "changed": len(changes["pages"]["changed"]),
        "deleted": len(changes["pages"]["deleted"]),
        "successful_pages": successful_updates,
        "failed_pages": [],
        "vectors_before": before_count,
        "vectors_after": after_count,
        "duration_seconds": duration,
    }


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> int:

    try:

        result = run_vector_update()

        if result["success"]:

            logger.info(
                "Vector updater finished successfully."
            )

            return 0

        logger.error(
            "Vector updater finished with errors."
        )

        return 1

    except KeyboardInterrupt:

        logger.warning(
            "Vector update interrupted by user."
        )

        return 130

    except Exception as error:

        logger.exception(
            "Fatal vector updater error: %s",
            error,
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())