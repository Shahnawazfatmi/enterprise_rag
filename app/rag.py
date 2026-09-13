from pathlib import Path
import logging

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.llm import load_llm, generate_answer

# ============================================================
# UPDATED: IMPORT QUERY ROUTER
# ============================================================
from app.router import classify_query, get_direct_response


# ============================================================
# PATHS & CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
COLLECTION_NAME = "consiva_knowledge"

EMBEDDING_MODEL = "BAAI/bge-m3"

TOP_K = 5


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD BGE-M3
# ============================================================

def load_embeddings():

    logger.info(
        f"Loading embedding model: {EMBEDDING_MODEL}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    logger.info("BGE-M3 loaded")

    return embeddings


# ============================================================
# LOAD CHROMA
# ============================================================

def load_vectorstore(embeddings):

    if not VECTORSTORE_DIR.exists():

        raise FileNotFoundError(
            f"Vector database not found: "
            f"{VECTORSTORE_DIR}"
        )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )

    count = vectorstore._collection.count()

    if count == 0:

        raise ValueError(
            "Chroma collection is empty."
        )

    logger.info(
        f"Chroma loaded successfully: {count} vectors"
    )

    return vectorstore


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(documents):

    context_parts = []

    for document in documents:

        metadata = document.metadata

        source = metadata.get(
            "source",
            "Unknown"
        )

        h1 = metadata.get("H1")
        h2 = metadata.get("H2")
        h3 = metadata.get("H3")

        headings = " > ".join(
            value
            for value in [h1, h2, h3]
            if value
        )

        context_parts.append(
            f"""
Source: {source}
Section: {headings}

Content:
{document.page_content}
"""
        )

    return "\n\n".join(context_parts)


# ============================================================
# RAG PIPELINE
# ============================================================

class ConsivaRAG:

    def __init__(self):

        logger.info(
            "Initializing Consiva RAG..."
        )

        # Load embedding model
        self.embeddings = load_embeddings()

        # Load Chroma
        self.vectorstore = load_vectorstore(
            self.embeddings
        )

        # Load Gemma
        self.llm = load_llm()

        logger.info(
            "Consiva RAG initialized successfully"
        )

    # --------------------------------------------------------
    # ASK QUESTION
    # --------------------------------------------------------

    def ask_question(
        self,
        question: str,
        top_k: int = TOP_K
    ):

        if not question or not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        question = question.strip()

        # ====================================================
        # UPDATED: QUERY ROUTING
        # ====================================================
        # Check whether the query is a greeting, thanks,
        # goodbye, identity question, or capability question.
        # These do not need vector search or Gemma.
        # ====================================================

        query_type = classify_query(question)

        logger.info(
            f"Query type: {query_type}"
        )

        # ====================================================
        # UPDATED: HANDLE NON-RAG QUERIES DIRECTLY
        # ====================================================
        # If the router identifies the query as something
        # that does not require the knowledge base, return
        # the predefined response immediately.
        # ====================================================

        if query_type != "rag":

            direct_response = get_direct_response(
                query_type
            )

            return {
                "answer": direct_response
            }

        logger.info(
            f"Processing question: {question}"
        )

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        documents = self.vectorstore.similarity_search(
            question,
            k=top_k
        )

        if not documents:

            return {
                "answer": (
                    "I couldn't find that information "
                    "in the knowledge base."
                )
            }

        logger.info(
            f"Retrieved {len(documents)} documents"
        )

        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context = build_context(
            documents
        )

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        answer = generate_answer(
            question=question,
            context=context,
            llm=self.llm
        )

        return {
            "answer": answer
        }


# ============================================================
# TERMINAL TEST
# ============================================================

def main():

    rag = ConsivaRAG()

    print("\n" + "=" * 60)
    print("CONSIVA AI RAG")
    print("=" * 60)

    print("\nType 'exit' to quit.")

    while True:

        question = input(
            "\nAsk a question: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:

            result = rag.ask_question(
                question
            )

            print(
                "\nAnswer:\n"
                + result["answer"]
            )

        except Exception as error:

            logger.error(
                f"Error: {error}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()