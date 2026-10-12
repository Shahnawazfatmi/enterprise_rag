"""
Incremental vector update driven by data/comparison/changes.json.

For every NEW or CHANGED page (and every page whose vectors were built
by an older chunker version):

    Read -> Chunk -> Embed only new chunks -> Upsert -> Verify -> Delete obsolete

Chunk IDs are "<page_id>::<hash of chunk text>", so a chunk whose text
did not change keeps its ID and is not embedded again.

New vectors are written and verified BEFORE old ones are deleted, so a
crash never leaves a page without vectors. On success the updater writes
data/vectorstore/index_version.json; the API reloads when it changes.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import logging
import re
import sys

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import (  # noqa: E402
    CHANGES_FILE,
    CHUNKER_VERSION,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    INDEX_VERSION_FILE,
    NEW_STRUCTURED_DIR,
    VECTORSTORE_DIR,
    embedding_kwargs,
    use_offline_model_cache,
)
from chunker.chunking import process_file  # noqa: E402
from app.index import SNAPSHOT_FILE, export_snapshot  # noqa: E402

import chromadb  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402


# Chroma upsert/delete batch size.
CHROMA_BATCH_SIZE = 100


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


# ============================================================
# CONTENT HASH (same normalization as compare.py)
# ============================================================

def normalize_content(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    normalized_lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line.strip())
        if line:
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def calculate_content_hash(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return hashlib.sha256(normalize_content(text).encode("utf-8")).hexdigest()


# ============================================================
# CHANGES
# ============================================================

def load_changes() -> dict:
    """
    Load and validate changes.json written by compare.py:

    {
        "pages": {"new": [...], "changed": [...], "unchanged": [...], "deleted": [...]},
        "details": {"<page_id>": {"status": ..., "baseline": {...}, "new": {...}}}
    }
    """

    if not CHANGES_FILE.exists():
        raise FileNotFoundError(f"Changes file not found: {CHANGES_FILE}")

    try:
        changes = json.loads(CHANGES_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in changes file: {error}") from error

    if not isinstance(changes, dict):
        raise ValueError("changes.json must contain a JSON object.")

    pages = changes.get("pages")
    details = changes.get("details")

    if not isinstance(pages, dict):
        raise ValueError("Invalid changes.json: 'pages' must be an object.")

    if not isinstance(details, dict):
        raise ValueError("Invalid changes.json: 'details' must be an object.")

    for key in ("new", "changed", "unchanged", "deleted"):
        if not isinstance(pages.get(key), list):
            raise ValueError(f"Invalid changes.json: pages.{key} must be a list.")

    return changes


def page_filename(changes: dict, page_id: str, side: str) -> str:
    """File name of a page on the 'new' or 'baseline' side of the comparison."""

    detail = changes["details"].get(page_id)

    if not isinstance(detail, dict) or not isinstance(detail.get(side), dict):
        raise ValueError(f"Missing '{side}' details for page: {page_id}")

    filename = detail[side].get("file_name")

    if not filename:
        raise ValueError(f"Missing {side}.file_name for page: {page_id}")

    return filename


def build_actions(changes: dict) -> list[dict]:
    """Convert changes.json into update actions."""

    pages = changes["pages"]
    actions = []

    for status, key, side in (
        ("NEW", "new", "new"),
        ("CHANGED", "changed", "new"),
        ("DELETED", "deleted", "baseline"),
    ):
        for page_id in pages[key]:
            actions.append({
                "page_id": page_id,
                "status": status,
                "filename": page_filename(changes, page_id, side),
            })

    return actions


def build_reindex_actions(collection, changes: dict) -> list[dict]:
    """
    Unchanged pages whose vectors are missing or were built with another
    CHUNKER_VERSION. This is how a chunking change reaches every page.
    """

    actions = []

    for page_id in changes["pages"]["unchanged"]:
        filename = page_filename(changes, page_id, "new")
        result = collection.get(where={"source": filename}, include=["metadatas"])
        metadatas = result.get("metadatas") or []

        if not metadatas or any(
            (metadata or {}).get("chunker_version") != CHUNKER_VERSION
            for metadata in metadatas
        ):
            actions.append({"page_id": page_id, "status": "REINDEX", "filename": filename})

    return actions


# ============================================================
# EMBEDDING MODEL / VECTOR STORE
# ============================================================

def create_embedding_model() -> HuggingFaceEmbeddings:
    use_offline_model_cache()
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = HuggingFaceEmbeddings(**embedding_kwargs())
    logger.info("Embedding model loaded successfully.")
    return model


def open_collection():
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


# ============================================================
# CHUNKS
# ============================================================

def load_and_chunk_page(file_path: Path) -> list[dict]:
    chunks = process_file(file_path)

    if not chunks:
        raise ValueError(f"No chunks generated for: {file_path.name}")

    for chunk in chunks:
        if not isinstance(chunk, dict) or not str(chunk.get("content", "")).strip():
            raise ValueError(f"Invalid or empty chunk in {file_path.name}")

    logger.info("Generated %d chunks: %s", len(chunks), file_path.name)

    return chunks


def build_chunk_id(page_id: str, content: str) -> str:
    """
    Stable, content-based ID, e.g.
        consiva.ai_pricing::3f9a1c0b7d2e4a61
    The same text always gets the same ID, so unchanged chunks are
    never re-embedded.
    """

    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
    return f"{page_id}::{digest}"


def build_metadata(chunk: dict, chunk_id: str, page_id: str, source: str, content_hash: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "page_id": page_id,
        "source": source,
        "url": chunk.get("url") or "",
        "title": chunk.get("title") or "",
        "section": chunk.get("section") or "",
        "content_hash": content_hash,
        "chunker_version": CHUNKER_VERSION,
        **{f"H{level}": chunk.get(f"H{level}") or "" for level in range(1, 7)},
    }


def embed_texts(embedding_model: HuggingFaceEmbeddings, texts: list[str]) -> list[list[float]]:
    embeddings = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        logger.info("Embedding chunks %d-%d/%d", start + 1, start + len(batch), len(texts))
        vectors = embedding_model.embed_documents(batch)

        if len(vectors) != len(batch):
            raise ValueError(f"Embedding count mismatch: expected {len(batch)}, got {len(vectors)}")

        embeddings.extend(vectors)

    return embeddings


# ============================================================
# CHROMA HELPERS
# ============================================================

def get_existing_ids_for_source(collection, source: str) -> list[str]:
    result = collection.get(where={"source": source}, include=[])
    return list(result.get("ids") or [])


def in_batches(items: list, size: int = CHROMA_BATCH_SIZE):
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


def delete_vectors(collection, ids: list[str]) -> None:
    for start, batch in in_batches(ids):
        logger.info("Deleting vectors %d-%d/%d", start + 1, start + len(batch), len(ids))
        collection.delete(ids=batch)


# ============================================================
# PROCESS NEW / CHANGED / REINDEX PAGE
# ============================================================

def update_page(collection, get_model, page_id: str, filename: str, status: str) -> dict:
    """
    Order: chunk -> embed new chunks -> upsert -> verify -> delete obsolete.
    The page always has a complete set of vectors in the database.
    """

    file_path = NEW_STRUCTURED_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"{status} page file not found: {file_path}")

    logger.info("=" * 70)
    logger.info("%s PAGE: %s (%s)", status, page_id, filename)

    content_hash = calculate_content_hash(file_path)
    chunks = load_and_chunk_page(file_path)

    ids = [build_chunk_id(page_id, chunk["content"]) for chunk in chunks]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [
        build_metadata(chunk, chunk_id, page_id, filename, content_hash)
        for chunk, chunk_id in zip(chunks, ids)
    ]

    old_ids = set(get_existing_ids_for_source(collection, filename))
    new_positions = [index for index, chunk_id in enumerate(ids) if chunk_id not in old_ids]
    kept_positions = [index for index, chunk_id in enumerate(ids) if chunk_id in old_ids]

    logger.info(
        "Chunks: %d total, %d new, %d unchanged, %d existing vectors",
        len(ids), len(new_positions), len(kept_positions), len(old_ids),
    )

    # 1. Embed and upsert only chunks whose text is new.
    if new_positions:
        vectors = embed_texts(get_model(), [documents[index] for index in new_positions])
        embedding_at = dict(zip(new_positions, vectors))

        for start, batch in in_batches(new_positions):
            logger.info("Upserting vectors %d-%d/%d", start + 1, start + len(batch), len(new_positions))
            collection.upsert(
                ids=[ids[index] for index in batch],
                documents=[documents[index] for index in batch],
                metadatas=[metadatas[index] for index in batch],
                embeddings=[embedding_at[index] for index in batch],
            )

    # 2. Refresh metadata of unchanged chunks (page hash, chunker version).
    for _, batch in in_batches(kept_positions):
        collection.update(
            ids=[ids[index] for index in batch],
            metadatas=[metadatas[index] for index in batch],
        )

    # 3. Verify every chunk of the page is present.
    present = set(collection.get(ids=ids, include=[]).get("ids") or [])
    missing = set(ids) - present

    if missing:
        raise RuntimeError(f"Vector verification failed for {page_id}: {len(missing)} missing")

    # 4. Only now remove vectors that no longer belong to the page.
    obsolete = sorted(old_ids - set(ids))
    delete_vectors(collection, obsolete)

    remaining = set(get_existing_ids_for_source(collection, filename))

    if remaining != set(ids):
        raise RuntimeError(
            f"Final verification failed for {page_id}: "
            f"{len(remaining - set(ids))} extra, {len(set(ids) - remaining)} missing"
        )

    logger.info("Updated %s: %d vectors (%d removed)", page_id, len(ids), len(obsolete))

    return {
        "page_id": page_id,
        "status": status,
        "filename": filename,
        "vectors": len(ids),
        "embedded": len(new_positions),
        "deleted_vectors": len(obsolete),
        "content_hash": content_hash,
    }


def delete_page(collection, page_id: str, filename: str) -> dict:
    """Remove every vector belonging to a deleted page."""

    logger.info("=" * 70)
    logger.info("DELETED PAGE: %s (%s)", page_id, filename)

    old_ids = get_existing_ids_for_source(collection, filename)
    delete_vectors(collection, old_ids)

    if get_existing_ids_for_source(collection, filename):
        raise RuntimeError(f"Deletion verification failed for {page_id}")

    logger.info("Deleted %d vectors for %s", len(old_ids), page_id)

    return {"page_id": page_id, "status": "DELETED", "filename": filename, "deleted_vectors": len(old_ids)}


def write_index_version(vector_count: int) -> None:
    """Tell the running API that the index changed (write after the snapshot)."""

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "vectors": vector_count,
        "chunker_version": CHUNKER_VERSION,
        "embedding_model": EMBEDDING_MODEL,
    }

    temp_file = INDEX_VERSION_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_file.replace(INDEX_VERSION_FILE)


# ============================================================
# MAIN UPDATE PROCESS
# ============================================================

def run_vector_update() -> dict:

    start_time = datetime.now(timezone.utc)

    logger.info("=" * 70)
    logger.info("STARTING INCREMENTAL VECTOR UPDATE (chunker %s)", CHUNKER_VERSION)
    logger.info("=" * 70)

    if not NEW_STRUCTURED_DIR.exists():
        raise FileNotFoundError(f"New structured directory not found: {NEW_STRUCTURED_DIR}")

    changes = load_changes()
    collection = open_collection()

    actions = build_actions(changes) + build_reindex_actions(collection, changes)

    for status in ("NEW", "CHANGED", "DELETED", "REINDEX"):
        logger.info("  %-8s: %d", status, sum(action["status"] == status for action in actions))

    if not actions and INDEX_VERSION_FILE.exists() and SNAPSHOT_FILE.exists():
        logger.info("NO VECTOR DATABASE CHANGES REQUIRED")
        return {"success": True, "actions": 0, "failed_pages": []}

    # The model is loaded only if some chunk really needs embedding.
    model_holder = {}

    def get_model():
        if "model" not in model_holder:
            model_holder["model"] = create_embedding_model()
        return model_holder["model"]

    before_count = collection.count()
    logger.info("Vector count before update: %d", before_count)

    successful = []
    failed = []

    for action in actions:
        try:
            if action["status"] == "DELETED":
                result = delete_page(collection, action["page_id"], action["filename"])
            else:
                result = update_page(
                    collection, get_model, action["page_id"], action["filename"], action["status"]
                )
            successful.append(result)

        except Exception as error:
            logger.exception("Failed to process page: %s", action["page_id"])
            failed.append({**action, "error": str(error)})

    after_count = collection.count()
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()

    logger.info("=" * 70)
    logger.info("VECTOR UPDATE FINISHED")
    logger.info("Successful pages : %d", len(successful))
    logger.info("Failed pages     : %d", len(failed))
    logger.info("Chunks embedded  : %d", sum(item.get("embedded", 0) for item in successful))
    logger.info("Vectors before   : %d", before_count)
    logger.info("Vectors after    : %d", after_count)
    logger.info("Duration         : %.2f seconds", duration)
    logger.info("=" * 70)

    if failed:
        for failure in failed:
            logger.error("  %s [%s] -> %s", failure["page_id"], failure["status"], failure["error"])
    else:
        exported = export_snapshot(collection)
        logger.info("Exported search snapshot: %d vectors", exported)
        write_index_version(after_count)

    return {
        "success": not failed,
        "actions": len(actions),
        "successful_pages": successful,
        "failed_pages": failed,
        "vectors_before": before_count,
        "vectors_after": after_count,
        "duration_seconds": duration,
    }


def main() -> int:
    try:
        result = run_vector_update()
        if result["success"]:
            logger.info("Vector updater finished successfully.")
            return 0
        logger.error("Vector updater finished with errors.")
        return 1

    except KeyboardInterrupt:
        logger.warning("Vector update interrupted by user.")
        return 130

    except Exception as error:
        logger.exception("Fatal vector updater error: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
