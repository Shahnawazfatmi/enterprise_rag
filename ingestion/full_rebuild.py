from pathlib import Path
import gc
import hashlib
import logging
import re
import shutil
import sys
from datetime import datetime

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))


# ============================================================
# REUSE EXACT EXISTING CHUNKING LOGIC
# ============================================================

from chunker.chunking import process_file


# ============================================================
# PATHS & CONFIGURATION
# ============================================================

SOURCE_DIR = BASE_DIR / "data" / "new_structured"

VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

COLLECTION_NAME = "consiva_knowledge"

MODEL_NAME = "BAAI/bge-m3"

EMBED_BATCH_SIZE = 32

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
# CONTENT HASH
# ============================================================

def normalize_content(text: str) -> str:
    """
    Same normalization strategy used by vector_updater.py
    and compare.py.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    normalized_lines = []

    for line in text.split("\n"):

        line = line.strip()

        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        )

        if line:
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def calculate_content_hash(
    file_path: Path,
) -> str:

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    normalized = normalize_content(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# PAGE ID
# ============================================================

def get_page_id(file_path: Path) -> str:
    """
    Example:

    consiva.ai_pricing.txt
        ->
    consiva.ai_pricing
    """

    return file_path.stem


# ============================================================
# EMBEDDING MODEL
# ============================================================

def create_embedding_model():

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

    logger.info(
        "Embedding model loaded successfully."
    )

    return model


# ============================================================
# CHUNKING
# ============================================================

def load_and_chunk_page(
    file_path: Path,
) -> list[dict]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Structured page not found: {file_path}"
        )

    logger.info(
        "Chunking: %s",
        file_path.name,
    )

    chunks = process_file(
        file_path
    )

    if not chunks:

        raise ValueError(
            f"No chunks generated for: "
            f"{file_path.name}"
        )

    for chunk in chunks:

        if not isinstance(chunk, dict):

            raise ValueError(
                f"Invalid chunk generated for: "
                f"{file_path.name}"
            )

        content = chunk.get("content")

        if not content or not str(content).strip():

            raise ValueError(
                f"Empty chunk detected in: "
                f"{file_path.name}"
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

    return (
        f"{page_id}::chunk_{index:06d}"
    )


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
    embedding_model,
    chunks: list[dict],
) -> list[list[float]]:

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

        if len(batch_embeddings) != len(
            batch_texts
        ):

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
# SAFE CHROMA UPSERT
# ============================================================

def upsert_vectors(
    vectorstore,
    ids,
    documents,
    metadatas,
    embeddings,
):

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
# PROCESS ONE PAGE
# ============================================================

def process_page(
    vectorstore,
    embedding_model,
    file_path: Path,
):

    page_id = get_page_id(
        file_path
    )

    source = file_path.name

    logger.info("=" * 70)

    logger.info(
        "PROCESSING PAGE: %s",
        page_id,
    )

    logger.info(
        "File: %s",
        source,
    )

    logger.info("=" * 70)

    # --------------------------------------------------------
    # 1. Hash
    # --------------------------------------------------------

    content_hash = calculate_content_hash(
        file_path
    )

    # --------------------------------------------------------
    # 2. Chunk
    # --------------------------------------------------------

    chunks = load_and_chunk_page(
        file_path
    )

    # --------------------------------------------------------
    # 3. IDs
    # --------------------------------------------------------

    ids = [
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
    # 4. Documents
    # --------------------------------------------------------

    documents = [
        str(chunk["content"])
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # 5. Metadata
    # --------------------------------------------------------

    metadatas = [
        build_metadata(
            chunk=chunk,
            chunk_id=chunk_id,
            page_id=page_id,
            source=source,
            content_hash=content_hash,
        )
        for chunk, chunk_id in zip(
            chunks,
            ids,
        )
    ]

    # --------------------------------------------------------
    # 6. Embeddings
    # --------------------------------------------------------

    embeddings = embed_chunks(
        embedding_model,
        chunks,
    )

    # --------------------------------------------------------
    # 7. Validate
    # --------------------------------------------------------

    if not (
        len(ids)
        == len(documents)
        == len(metadatas)
        == len(embeddings)
    ):

        raise ValueError(
            f"Prepared vector mismatch: "
            f"{page_id}"
        )

    # --------------------------------------------------------
    # 8. Upsert
    # --------------------------------------------------------

    upsert_vectors(
        vectorstore=vectorstore,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    # --------------------------------------------------------
    # 9. Verify
    # --------------------------------------------------------

    result = vectorstore._collection.get(
        ids=ids,
        include=[],
    )

    verified_ids = set(
        result.get("ids", [])
    )

    expected_ids = set(ids)

    missing_ids = (
        expected_ids - verified_ids
    )

    if missing_ids:

        raise RuntimeError(
            f"Verification failed for "
            f"{page_id}. "
            f"Missing {len(missing_ids)} vectors."
        )

    logger.info(
        "Verified %d vectors for %s",
        len(verified_ids),
        page_id,
    )

    return len(ids)


# ============================================================
# VALIDATE SOURCE DIRECTORY
# ============================================================

def validate_source_files():

    if not SOURCE_DIR.exists():

        raise FileNotFoundError(
            f"Source directory does not exist: "
            f"{SOURCE_DIR}"
        )

    files = sorted(
        SOURCE_DIR.glob("*.txt")
    )

    if not files:

        raise ValueError(
            f"No .txt files found in: "
            f"{SOURCE_DIR}"
        )

    logger.info(
        "Structured pages found: %d",
        len(files),
    )

    page_ids = [
        get_page_id(file)
        for file in files
    ]

    duplicates = {
        page_id
        for page_id in page_ids
        if page_ids.count(page_id) > 1
    }

    if duplicates:

        raise ValueError(
            "Duplicate page IDs detected: "
            f"{sorted(duplicates)}"
        )

    return files


# ============================================================
# VERIFY FINAL DATABASE
# ============================================================

def verify_database(
    vectorstore,
    expected_page_ids,
):

    collection = vectorstore._collection

    vector_count = collection.count()

    result = collection.get(
        include=["metadatas"]
    )

    metadatas = result.get(
        "metadatas",
        [],
    )

    actual_page_ids = set()

    for metadata in metadatas:

        if not metadata:
            continue

        page_id = metadata.get(
            "page_id"
        )

        if page_id:
            actual_page_ids.add(
                page_id
            )

    expected_page_ids = set(
        expected_page_ids
    )

    missing_pages = (
        expected_page_ids
        - actual_page_ids
    )

    unexpected_pages = (
        actual_page_ids
        - expected_page_ids
    )

    logger.info("=" * 70)
    logger.info("FINAL DATABASE VERIFICATION")
    logger.info("=" * 70)

    logger.info(
        "Vectors       : %d",
        vector_count,
    )

    logger.info(
        "Expected pages: %d",
        len(expected_page_ids),
    )

    logger.info(
        "Actual pages  : %d",
        len(actual_page_ids),
    )

    if missing_pages:

        raise RuntimeError(
            "Missing pages in rebuilt Chroma: "
            f"{sorted(missing_pages)}"
        )

    if unexpected_pages:

        raise RuntimeError(
            "Unexpected pages in rebuilt Chroma: "
            f"{sorted(unexpected_pages)}"
        )

    if len(actual_page_ids) != len(
        expected_page_ids
    ):

        raise RuntimeError(
            "Page coverage verification failed."
        )

    if vector_count == 0:

        raise RuntimeError(
            "Rebuilt Chroma collection is empty."
        )

    logger.info(
        "PAGE COVERAGE VERIFIED: %d/%d",
        len(actual_page_ids),
        len(expected_page_ids),
    )

    logger.info(
        "VECTOR DATABASE VERIFICATION PASSED"
    )


# ============================================================
# SAFE DIRECTORY SWAP
# ============================================================

def swap_vectorstore(
    temporary_dir: Path,
):

    backup_dir = BASE_DIR / (
        "data/vectorstore_backup_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    logger.info("=" * 70)
    logger.info("INSTALLING REBUILT VECTOR DATABASE")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Make sure current Chroma is released
    # --------------------------------------------------------

    gc.collect()

    # --------------------------------------------------------
    # Backup existing vectorstore
    # --------------------------------------------------------

    if VECTORSTORE_DIR.exists():

        logger.info(
            "Backing up existing vectorstore:"
        )

        logger.info(
            "%s",
            backup_dir,
        )

        VECTORSTORE_DIR.rename(
            backup_dir
        )

    try:

        # ----------------------------------------------------
        # Install new database
        # ----------------------------------------------------

        temporary_dir.rename(
            VECTORSTORE_DIR
        )

        logger.info(
            "New vectorstore installed."
        )

        logger.info(
            "Backup preserved at:"
        )

        logger.info(
            "%s",
            backup_dir,
        )

        return backup_dir

    except Exception:

        logger.error(
            "Failed to install rebuilt vectorstore."
        )

        # ----------------------------------------------------
        # Restore old database
        # ----------------------------------------------------

        if VECTORSTORE_DIR.exists():

            shutil.rmtree(
                VECTORSTORE_DIR,
                ignore_errors=True,
            )

        if backup_dir.exists():

            backup_dir.rename(
                VECTORSTORE_DIR
            )

            logger.info(
                "Original vectorstore restored."
            )

        raise


# ============================================================
# MAIN REBUILD
# ============================================================

def main():

    logger.info("=" * 70)
    logger.info("FULL CHROMA VECTOR REBUILD")
    logger.info("=" * 70)

    logger.info(
        "Source      : %s",
        SOURCE_DIR,
    )

    logger.info(
        "Destination : %s",
        VECTORSTORE_DIR,
    )

    logger.info(
        "Collection  : %s",
        COLLECTION_NAME,
    )

    logger.info(
        "Embedding   : %s",
        MODEL_NAME,
    )

    # --------------------------------------------------------
    # 1. Validate all source files
    # --------------------------------------------------------

    files = validate_source_files()

    expected_page_ids = [
        get_page_id(file)
        for file in files
    ]

    # --------------------------------------------------------
    # IMPORTANT:
    # Build into a temporary directory first.
    # Existing Chroma is NOT touched during the build.
    # --------------------------------------------------------

    temporary_dir = BASE_DIR / (
        "data/vectorstore_rebuild_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    if temporary_dir.exists():

        shutil.rmtree(
            temporary_dir
        )

    temporary_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    logger.info(
        "Temporary vectorstore:"
    )

    logger.info(
        "%s",
        temporary_dir,
    )

    try:

        # ----------------------------------------------------
        # 2. Load embedding model
        # ----------------------------------------------------

        embedding_model = (
            create_embedding_model()
        )

        # ----------------------------------------------------
        # 3. Create NEW temporary Chroma
        # ----------------------------------------------------

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(
                temporary_dir
            ),
            embedding_function=embedding_model,
        )

        # ----------------------------------------------------
        # 4. Process all pages
        # ----------------------------------------------------

        total_vectors = 0

        processed_pages = 0

        for index, file_path in enumerate(
            files,
            start=1,
        ):

            logger.info(
                "\n[%d/%d] %s",
                index,
                len(files),
                file_path.name,
            )

            vectors = process_page(
                vectorstore=vectorstore,
                embedding_model=embedding_model,
                file_path=file_path,
            )

            total_vectors += vectors

            processed_pages += 1

        # ----------------------------------------------------
        # 5. Verify temporary database
        # ----------------------------------------------------

        verify_database(
            vectorstore=vectorstore,
            expected_page_ids=expected_page_ids,
        )

        logger.info("=" * 70)
        logger.info("REBUILD SUCCESSFUL")
        logger.info("=" * 70)

        logger.info(
            "Pages processed : %d",
            processed_pages,
        )

        logger.info(
            "Total vectors   : %d",
            total_vectors,
        )

        # ----------------------------------------------------
        # 6. Release Chroma handles
        # ----------------------------------------------------

        del vectorstore
        del embedding_model

        gc.collect()

        # ----------------------------------------------------
        # 7. Safely replace old database
        # ----------------------------------------------------

        backup_dir = swap_vectorstore(
            temporary_dir
        )

        # ----------------------------------------------------
        # 8. Final verification after swap
        # ----------------------------------------------------

        logger.info(
            "Opening installed vectorstore "
            "for final verification..."
        )

        final_embeddings = (
            create_embedding_model()
        )

        final_vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(
                VECTORSTORE_DIR
            ),
            embedding_function=final_embeddings,
        )

        verify_database(
            vectorstore=final_vectorstore,
            expected_page_ids=expected_page_ids,
        )

        final_count = (
            final_vectorstore._collection.count()
        )

        del final_vectorstore
        del final_embeddings

        gc.collect()

        logger.info("=" * 70)
        logger.info("FULL REBUILD COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        logger.info(
            "Pages : %d",
            len(expected_page_ids),
        )

        logger.info(
            "Vectors: %d",
            final_count,
        )

        logger.info(
            "Backup : %s",
            backup_dir,
        )

    except Exception:

        logger.exception(
            "FULL REBUILD FAILED"
        )

        # ----------------------------------------------------
        # Never leave an incomplete temporary DB behind
        # ----------------------------------------------------

        gc.collect()

        if temporary_dir.exists():

            shutil.rmtree(
                temporary_dir,
                ignore_errors=True,
            )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        logger.error(
            "Rebuild interrupted by user."
        )

        sys.exit(130)

    except Exception:

        logger.error(
            "Full rebuild failed."
        )

        sys.exit(1)