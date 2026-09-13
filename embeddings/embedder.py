from pathlib import Path
import json
import logging
import sys

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "chunks" / "chunks.jsonl"

OUTPUT_DIR = BASE_DIR / "data" / "embeddings"

EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "metadata.jsonl"

MODEL_NAME = "BAAI/bge-m3"

BATCH_SIZE = 32


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD CHUNKS SAFELY
# ============================================================

def load_chunks():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {INPUT_FILE}"
        )

    chunks = []
    seen_ids = set()

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:
                chunk = json.loads(line)

            except json.JSONDecodeError:
                logger.warning(
                    f"Skipping invalid JSON at line {line_number}"
                )
                continue

            content = chunk.get("content")

            if not isinstance(content, str):
                logger.warning(
                    f"Skipping chunk without valid content "
                    f"at line {line_number}"
                )
                continue

            content = content.strip()

            if not content:
                logger.warning(
                    f"Skipping empty chunk at line {line_number}"
                )
                continue

            chunk_id = chunk.get("chunk_id")

            if not chunk_id:
                logger.warning(
                    f"Missing chunk_id at line {line_number}"
                )
                continue

            if chunk_id in seen_ids:
                logger.warning(
                    f"Duplicate chunk_id skipped: {chunk_id}"
                )
                continue

            seen_ids.add(chunk_id)

            chunks.append(chunk)

    return chunks


# ============================================================
# CREATE EMBEDDING MODEL
# ============================================================

def create_embedding_model():

    logger.info(
        f"Loading embedding model: {MODEL_NAME}"
    )

    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    metadata_file,
    chunks
):

    with metadata_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        for chunk in chunks:

            metadata = {
                "chunk_id": chunk.get("chunk_id"),
                "source": chunk.get("source"),
                "H1": chunk.get("H1"),
                "H2": chunk.get("H2"),
                "H3": chunk.get("H3"),
                "H4": chunk.get("H4"),
                "H5": chunk.get("H5"),
                "H6": chunk.get("H6"),
                "content": chunk.get("content")
            }

            file.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# MAIN EMBEDDING PIPELINE
# ============================================================

def main():

    logger.info("Starting embedding pipeline")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    logger.info(
        f"Reading chunks from: {INPUT_FILE}"
    )

    chunks = load_chunks()

    if not chunks:
        logger.error(
            "No valid chunks found."
        )
        sys.exit(1)

    logger.info(
        f"Valid chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Create embedding model
    # --------------------------------------------------------

    embeddings_model = create_embedding_model()

    # --------------------------------------------------------
    # Generate embeddings in batches
    # --------------------------------------------------------

    all_embeddings = []

    total = len(chunks)

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = chunks[start:end]

        texts = [
            chunk["content"]
            for chunk in batch
        ]

        logger.info(
            f"Embedding chunks {start + 1}-{end} "
            f"of {total}"
        )

        try:

            vectors = (
                embeddings_model.embed_documents(
                    texts
                )
            )

        except Exception as error:

            logger.exception(
                f"Embedding failed for batch "
                f"{start + 1}-{end}"
            )

            raise error

        all_embeddings.extend(
            vectors
        )

    # --------------------------------------------------------
    # Validate embedding count
    # --------------------------------------------------------

    if len(all_embeddings) != len(chunks):

        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    embeddings_array = np.asarray(
        all_embeddings,
        dtype=np.float32
    )

    if embeddings_array.ndim != 2:

        raise RuntimeError(
            "Invalid embedding array shape."
        )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    np.save(
        EMBEDDINGS_FILE,
        embeddings_array
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_metadata(
        METADATA_FILE,
        chunks
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not EMBEDDINGS_FILE.exists():
        raise RuntimeError(
            "Embeddings file was not created."
        )

    if not METADATA_FILE.exists():
        raise RuntimeError(
            "Metadata file was not created."
        )

    logger.info("=" * 60)
    logger.info("EMBEDDING COMPLETED")
    logger.info("=" * 60)

    logger.info(
        f"Chunks       : {len(chunks)}"
    )

    logger.info(
        f"Vector shape : {embeddings_array.shape}"
    )

    logger.info(
        f"Embedding dim: {embeddings_array.shape[1]}"
    )

    logger.info(
        f"Embeddings   : {EMBEDDINGS_FILE}"
    )

    logger.info(
        f"Metadata     : {METADATA_FILE}"
    )

    logger.info("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()