"""
Full vector rebuild from data/new_structured.

Use this for disaster recovery or a fresh machine. The nightly pipeline
does NOT need it: a chunker change (CHUNKER_VERSION) already makes the
incremental updater re-index every page.

Safety:
- The new database is built in a temporary folder; the live one is not
  touched until the build is complete and verified.
- The live folder is renamed to a timestamped backup, then the new one
  is moved into place.

Usage:
    python ingestion/full_rebuild.py
"""

from pathlib import Path
from datetime import datetime
import gc
import logging
import shutil
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import chromadb  # noqa: E402

from config import COLLECTION_NAME, NEW_STRUCTURED_DIR, VECTORSTORE_DIR  # noqa: E402
from ingestion.compare import normalize_page_id  # noqa: E402
from app.index import export_snapshot_from_disk  # noqa: E402
from ingestion.vector_updater import (  # noqa: E402
    build_chunk_id,
    build_metadata,
    calculate_content_hash,
    create_embedding_model,
    embed_texts,
    in_batches,
    load_and_chunk_page,
    write_index_version,
)

logger = logging.getLogger(__name__)


def build(target_dir: Path) -> int:
    files = sorted(NEW_STRUCTURED_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(f"No .txt files in {NEW_STRUCTURED_DIR}")

    ids, documents, metadatas = [], [], []

    for file_path in files:
        page_id = normalize_page_id(file_path.name)
        content_hash = calculate_content_hash(file_path)

        for chunk in load_and_chunk_page(file_path):
            chunk_id = build_chunk_id(page_id, chunk["content"])
            ids.append(chunk_id)
            documents.append(chunk["content"])
            metadatas.append(build_metadata(chunk, chunk_id, page_id, file_path.name, content_hash))

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk IDs generated.")

    logger.info("Pages: %d, chunks: %d", len(files), len(ids))

    embeddings = embed_texts(create_embedding_model(), documents)

    client = chromadb.PersistentClient(path=str(target_dir))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    for start, batch in in_batches(list(range(len(ids)))):
        collection.add(
            ids=[ids[index] for index in batch],
            documents=[documents[index] for index in batch],
            metadatas=[metadatas[index] for index in batch],
            embeddings=[embeddings[index] for index in batch],
        )

    count = collection.count()

    if count != len(ids):
        raise RuntimeError(f"Verification failed: expected {len(ids)} vectors, found {count}")

    # Release file handles before the folder is moved (needed on Windows).
    del collection, client
    gc.collect()

    return count


def main() -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = VECTORSTORE_DIR.parent / f"vectorstore_rebuild_{stamp}"
    backup_dir = VECTORSTORE_DIR.parent / f"vectorstore_backup_{stamp}"

    logger.info("Building new vector store in %s", temp_dir)

    try:
        count = build(temp_dir)
    except Exception:
        logger.exception("Rebuild failed; the live vector store was not changed.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 1

    if VECTORSTORE_DIR.exists():
        VECTORSTORE_DIR.rename(backup_dir)
        logger.info("Previous vector store kept at %s", backup_dir)

    temp_dir.rename(VECTORSTORE_DIR)
    export_snapshot_from_disk()
    write_index_version(count)

    logger.info("Rebuild complete: %d vectors", count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
