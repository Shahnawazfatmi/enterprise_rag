from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
COLLECTION_NAME = "consiva_knowledge"

EMBEDDING_MODEL = "BAAI/bge-m3"

TOP_K = 5


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading BGE-M3...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

print("BGE-M3 loaded.")


# ============================================================
# LOAD CHROMA
# ============================================================

print("\nLoading ChromaDB...")

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(VECTORSTORE_DIR),
    embedding_function=embeddings,
)

count = vectorstore._collection.count()

print(f"Chroma vectors: {count}")


# ============================================================
# TEST QUESTION
# ============================================================

question = input(
    "\nEnter your question: "
).strip()

if not question:
    print("Question cannot be empty.")
    exit()


# ============================================================
# RETRIEVE CHUNKS WITH SCORES
# ============================================================

results = vectorstore.similarity_search_with_score(
    question,
    k=TOP_K
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("RETRIEVED CHUNKS")
print("=" * 80)

print(f"\nQuestion: {question}")
print(f"Retrieved: {len(results)} chunks")


for i, (document, score) in enumerate(results, start=1):

    metadata = document.metadata

    print("\n" + "-" * 80)

    print(f"Rank       : {i}")
    print(f"Score      : {score}")

    print(
        f"Chunk ID   : "
        f"{metadata.get('chunk_id', 'N/A')}"
    )

    print(
        f"Page ID    : "
        f"{metadata.get('page_id', 'N/A')}"
    )

    print(
        f"Source     : "
        f"{metadata.get('source', 'N/A')}"
    )

    print(
        f"H1         : "
        f"{metadata.get('H1', 'N/A')}"
    )

    print(
        f"H2         : "
        f"{metadata.get('H2', 'N/A')}"
    )

    print(
        f"H3         : "
        f"{metadata.get('H3', 'N/A')}"
    )

    print("\nCONTENT:")
    print(document.page_content)


print("\n" + "=" * 80)
print("END OF RETRIEVAL TEST")
print("=" * 80)