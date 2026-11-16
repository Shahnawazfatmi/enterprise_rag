from collections.abc import Iterator
import logging
import re
import threading
import time

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from app.index import SearchIndex
from app.llm import LLMUnavailableError, generate_answer, generate_answer_stream, llm_name, load_llms
from app.next_step import LinkCatalog, choose_next_step
from app.router import classify_query, get_direct_response, wants_detail, wants_steps
from config import (
    INDEX_VERSION_FILE,
    LINK_CATALOG_FILE,
    MAX_CHUNKS_PER_PAGE,
    MAX_HISTORY_MESSAGES,
    MIN_RELEVANCE,
    NEXT_STEP_CANDIDATES,
    NEXT_STEP_ENABLED,
    NEXT_STEP_MIN_SCORE,
    NEXT_STEP_TIMEOUT_SECONDS,
    NOT_FOUND_ANSWER,
    RELATIVE_RELEVANCE_MARGIN,
    RETRIEVAL_CANDIDATES,
    TOP_K,
    embedding_kwargs,
    use_offline_model_cache,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

CITATION = re.compile(r"\[(\d+)\]")

# Follow-ups like "and for Enterprise?" or "does it include SSO?" are
# searched together with the previous question.
FOLLOW_UP = re.compile(
    r"^(and|also|but|so|what about|how about|same|then)\b"
    r"|\b(it|its|that|this|they|them|those|these|there|one)\b",
    re.IGNORECASE,
)
FOLLOW_UP_MAX_WORDS = 12


class KnowledgeBaseUnavailableError(RuntimeError):
    """The search index could not be loaded."""


# ============================================================
# CONTEXT
# ============================================================

def build_context(hits: list[dict]) -> str:
    """Numbered sources for the prompt; the numbers are used for citations."""

    parts = []

    for number, hit in enumerate(hits, start=1):
        url = hit["metadata"].get("url") or hit["metadata"].get("source", "")
        parts.append(f"[{number}] (from {url})\n{hit['text']}")

    return "\n\n".join(parts)


def page_title(metadata: dict) -> str:
    return (
        metadata.get("title")
        or metadata.get("H1")
        or metadata.get("source", "").removesuffix(".txt")
    )


def cited_sources(answer: str, hits: list[dict]) -> list[dict]:
    """Sources the answer cites, in citation-number order."""

    numbers = sorted({int(number) for number in CITATION.findall(answer)})

    return [
        {
            "id": number,
            "title": page_title(hits[number - 1]["metadata"]),
            "section": hits[number - 1]["metadata"].get("section", ""),
            "url": hits[number - 1]["metadata"].get("url", ""),
        }
        for number in numbers
        if 1 <= number <= len(hits)
    ]


def clean_history(history: list[dict] | None) -> list[dict]:
    """
    Keep the last few turns. Old citation numbers are removed: they point
    to the previous turn's sources, not to the sources of this answer.
    """

    messages = []

    for item in history or []:
        content = CITATION.sub("", str(item.get("content", ""))).strip()
        if item.get("role") in ("user", "assistant") and content:
            messages.append({"role": item["role"], "content": content})

    return messages[-MAX_HISTORY_MESSAGES:]


def retrieval_query(question: str, history: list[dict]) -> str:
    """Search short follow-up questions together with the previous user question."""

    previous = [message["content"] for message in history if message["role"] == "user"]
    short = len(question.split()) <= FOLLOW_UP_MAX_WORDS

    if previous and short and FOLLOW_UP.search(question):
        return f"{previous[-1]} {question}"

    return question


# ============================================================
# RAG PIPELINE
# ============================================================

class ConsivaRAG:

    def __init__(self):

        logger.info("Initializing Consiva RAG...")

        use_offline_model_cache()
        self.embeddings = HuggingFaceEmbeddings(**embedding_kwargs())

        self._lock = threading.Lock()
        self._index = None
        self._index_version = None
        self.index_error = None
        self.reload_index_if_changed()

        # Primary LLM first, then the fallback.
        try:
            self.llms = load_llms()
        except LLMUnavailableError as error:
            logger.error(f"LLM not configured: {error}")
            self.llms = []

        # Same models for the next-step pick, but quick: short timeout, no retries.
        self._links = None
        self._links_version = None
        self.next_step_llms = []

        if NEXT_STEP_ENABLED and self.llms:
            self.next_step_llms = load_llms(timeout=NEXT_STEP_TIMEOUT_SECONDS, max_retries=0, max_tokens=150)
            self.reload_links_if_changed()

        logger.info("Consiva RAG initialized")

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    def _current_version(self):
        try:
            return INDEX_VERSION_FILE.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def reload_index_if_changed(self) -> None:
        """Load the snapshot at start-up and again after each pipeline run."""

        version = self._current_version()

        if self._index is not None and version == self._index_version:
            return

        with self._lock:
            if self._index is not None and version == self._index_version:
                return

            try:
                index = SearchIndex.load()
            except Exception as error:
                self.index_error = str(error)
                logger.exception("Could not load the search index")
                return

            self._index = index
            self._index_version = version
            self.index_error = None
            logger.info(f"Search index loaded: {len(index)} chunks")

    @property
    def index(self) -> SearchIndex:
        self.reload_index_if_changed()

        if self._index is None:
            raise KnowledgeBaseUnavailableError(self.index_error or "Search index not loaded.")

        return self._index

    def reload_links_if_changed(self) -> None:
        """Load the next-step link catalog, and again when the pipeline rebuilds it."""

        try:
            version = LINK_CATALOG_FILE.stat().st_mtime_ns
        except FileNotFoundError:
            return

        if version == self._links_version:
            return

        with self._lock:
            if version == self._links_version:
                return

            try:
                links = LinkCatalog.load()
            except Exception:
                logger.exception("Could not load the link catalog; next steps are off until it loads")
                self._links_version = version
                return

            self._links = links
            self._links_version = version
            logger.info(f"Link catalog loaded: {len(links)} links")

    def status(self) -> dict:
        return {
            "index_loaded": self._index is not None,
            "chunks": len(self._index) if self._index is not None else 0,
            "index_error": self.index_error,
            "llm_configured": bool(self.llms),
            "llm": self.llms[0][0] if self.llms else llm_name(),
            "llm_fallback": self.llms[1][0] if len(self.llms) > 1 else None,
            "next_step_links": len(self._links) if self._links is not None else 0,
        }

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    def search(self, query: str) -> tuple[list[dict], np.ndarray]:
        """Hits for the query, and the query vector (reused for the next step)."""

        corrected = self.index.correct_spelling(query)

        if corrected != query:
            logger.info("Spelling corrected for search")
            query = corrected

        query_vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)

        hits = self.index.search(
            query_vector=query_vector,
            query_text=query,
            candidates=RETRIEVAL_CANDIDATES,
            top_k=TOP_K,
            min_relevance=MIN_RELEVANCE,
            relative_margin=RELATIVE_RELEVANCE_MARGIN,
            max_per_page=MAX_CHUNKS_PER_PAGE,
        )

        return hits, query_vector

    def retrieve(self, query: str) -> list[dict]:
        return self.search(query)[0]

    # --------------------------------------------------------
    # ASK QUESTION
    # --------------------------------------------------------

    def prepare(self, question: str, history: list[dict] | None = None) -> tuple[dict | None, dict | None]:
        """
        Everything before the LLM call (routing and retrieval).

        Returns (result, None) when the answer needs no LLM (greeting,
        nothing found), otherwise (None, request) with the question,
        history and hits for ask_question / stream_answer.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        question = question.strip()
        query_type = classify_query(question)

        if query_type != "rag":
            return {"answer": get_direct_response(query_type), "sources": [], "query_type": query_type}, None

        history = clean_history(history)
        hits, query_vector = self.search(retrieval_query(question, history))

        logger.info(
            "Retrieved %d chunks: %s",
            len(hits),
            ", ".join(f"{hit['metadata'].get('source', '?')}({hit['dense']})" for hit in hits),
        )

        if not hits:
            return {"answer": NOT_FOUND_ANSWER, "sources": [], "query_type": "not_found"}, None

        if not self.llms:
            raise LLMUnavailableError(f"The language model ({llm_name()}) is not configured.")

        return None, {
            "question": question,
            "history": history,
            "hits": hits,
            "query_vector": query_vector,
            "detailed": wants_detail(question),
            "steps": wants_steps(question),
        }

    def ask_question(self, question: str, history: list[dict] | None = None) -> dict:

        result, request = self.prepare(question, history)

        if result is not None:
            return result

        answer = generate_answer(
            question=request["question"],
            context=build_context(request["hits"]),
            llms=self.llms,
            history=request["history"],
            detailed=request["detailed"],
            steps=request["steps"],
        )

        return {
            "answer": answer,
            "sources": cited_sources(answer, request["hits"]),
            "query_type": "rag",
        }

    def stream_answer(self, request: dict) -> Iterator[tuple[str, dict]]:
        """
        The answer for a request from prepare(), while the LLM writes it:
        ("delta", {"text"}) pieces, ("reset", {}) when the fallback LLM
        starts over, then ("done", ...) with the same fields as ask_question.
        """

        started = time.perf_counter()
        first_text = None
        pieces = []

        for event, text in generate_answer_stream(
            question=request["question"],
            context=build_context(request["hits"]),
            llms=self.llms,
            history=request["history"],
            detailed=request["detailed"],
            steps=request["steps"],
        ):
            if event == "reset":
                pieces.clear()
                yield "reset", {}
                continue

            if first_text is None:
                first_text = time.perf_counter() - started

            pieces.append(text)
            yield "delta", {"text": text}

        answer = "".join(pieces).strip()

        logger.info(
            "Streamed answer: first text after %.2f s, done after %.2f s",
            first_text or 0.0,
            time.perf_counter() - started,
        )

        yield "done", {
            "answer": answer,
            "sources": cited_sources(answer, request["hits"]),
            "query_type": "rag",
        }

    # --------------------------------------------------------
    # NEXT STEP
    # --------------------------------------------------------

    def answer_vector(self, answer: str, hits: list[dict]) -> np.ndarray | None:
        """Average stored vector of the chunks the answer cites, else of all hits; None without hits."""

        numbers = sorted({int(number) for number in CITATION.findall(answer)})
        cited = [hits[number - 1] for number in numbers if 1 <= number <= len(hits)]

        return self.index.mean_vector([hit["id"] for hit in cited or hits])

    def suggest_next_step(
        self,
        question: str,
        answer: str,
        sources: list[dict],
        query_vector: np.ndarray | None = None,
        hits: list[dict] | None = None,
    ) -> dict | None:
        """
        One relevant link and follow-up question after an answer, or None
        (see app/next_step.py). Never raises: a failed suggestion only
        means no card under the answer.

        The answer side of the match is the average stored vector of the
        chunks the answer cites (all hits if it cites none), not a new
        embedding of the answer text: bge-m3 on a long answer costs about
        1 s on a laptop and far more on a small server CPU, and picks the
        same link in practice.
        """

        if not self.next_step_llms:
            return None

        self.reload_links_if_changed()

        if self._links is None:
            return None

        started = time.perf_counter()

        try:
            if query_vector is None:
                query_vector = self.embeddings.embed_query(question)

            answer_vector = self.answer_vector(answer, hits or [])

            if answer_vector is None:
                answer_vector = query_vector

            candidates = self._links.shortlist(
                question_vector=query_vector,
                answer_vector=answer_vector,
                exclude_urls={source.get("url", "").rstrip("/") for source in sources},
                limit=NEXT_STEP_CANDIDATES,
                min_score=NEXT_STEP_MIN_SCORE,
            )

            step = choose_next_step(question, answer, candidates, self.next_step_llms)

        except Exception:
            logger.exception("Next step failed")
            return None

        logger.info(
            "Next step: %s (%d candidates, %.2f s)",
            step["url"] if step else "none",
            len(candidates),
            time.perf_counter() - started,
        )

        return step


# ============================================================
# TERMINAL TEST
# ============================================================

def main():

    rag = ConsivaRAG()
    history = []

    print("\nCONSIVA AI RAG - type 'exit' to quit.")

    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            result = rag.ask_question(question, history)
            print("\nAnswer:\n" + result["answer"])
            for source in result["sources"]:
                print(f"  [{source['id']}] {source['title']} - {source['url']}")
            history += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": result["answer"]},
            ]

        except Exception as error:
            logger.error(f"Error: {error}")


if __name__ == "__main__":
    main()
