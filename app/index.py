"""
In-memory search index used by the API.

The nightly pipeline is the only process that opens Chroma for writing.
After a successful update it exports a snapshot (snapshot.npz: vectors,
texts, metadata) and writes index_version.json. The API loads the
snapshot into memory and reloads it when index_version.json changes,
so the API never shares the Chroma database with another process.

Search = dense (bge-m3 cosine, exact) plus a small BM25 boost. BM25
catches exact terms such as "8(5)", "CERT-In" or "5,999" that
embeddings match only loosely.
"""

from collections import Counter, defaultdict
from pathlib import Path
import difflib
import json
import logging
import math
import os
import re

import numpy as np

from config import COLLECTION_NAME, VECTORSTORE_DIR

logger = logging.getLogger(__name__)

SNAPSHOT_FILE = VECTORSTORE_DIR / "snapshot.npz"

# Legal references such as "8(5)" or "17(1)(f)" stay one token, so a
# question about section 8(5) matches that exact section.
TOKEN = re.compile(r"\d+(?:\([0-9a-z]{1,3}\))+|[a-z0-9₹§]+(?:[.,][0-9]+)*")

STOPWORDS = set(
    "a about all an and any are as at be by can could do does for from get "
    "has have how i if in is it its many me much my of on or our please should "
    "so tell than that the their them then there these this to under us was "
    "we what when where which who why will with would you your".split()
)


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN.findall(text.lower()) if token not in STOPWORDS]


# ============================================================
# SNAPSHOT EXPORT / LOAD
# ============================================================

def export_snapshot(collection, path: Path = SNAPSHOT_FILE) -> int:
    """Write every vector of a Chroma collection to one .npz file (atomic replace)."""

    data = collection.get(include=["documents", "metadatas", "embeddings"])

    records = [
        {"id": chunk_id, "text": document or "", "metadata": metadata or {}}
        for chunk_id, document, metadata in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    embeddings = np.asarray(data["embeddings"], dtype=np.float32).reshape(len(records), -1)

    temp_file = path.with_name(path.stem + ".tmp.npz")
    np.savez(temp_file, embeddings=embeddings, records=np.array(json.dumps(records, ensure_ascii=False)))
    os.replace(temp_file, path)

    return len(records)


def export_snapshot_from_disk(path: Path = SNAPSHOT_FILE) -> int:
    """Create the snapshot from the Chroma folder (first start after an upgrade)."""

    import chromadb
    from chromadb.api.client import SharedSystemClient

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    try:
        return export_snapshot(client.get_collection(COLLECTION_NAME), path)
    finally:
        # Chroma caches open databases per process; release this one so the
        # API does not keep the files open while the pipeline writes.
        SharedSystemClient.clear_system_cache()


# ============================================================
# SEARCH INDEX
# ============================================================

class SearchIndex:

    def __init__(self, records: list[dict], embeddings: np.ndarray):
        if not records:
            raise ValueError("The knowledge base is empty.")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.clip(norms, 1e-12, None)
        self.records = records
        self._build_bm25()

    @classmethod
    def load(cls, path: Path = SNAPSHOT_FILE) -> "SearchIndex":
        if not path.exists():
            logger.info("No snapshot found; exporting one from Chroma.")
            export_snapshot_from_disk(path)

        with np.load(path) as snapshot:
            embeddings = snapshot["embeddings"].astype(np.float32)
            records = json.loads(str(snapshot["records"]))

        return cls(records, embeddings)

    def __len__(self) -> int:
        return len(self.records)

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    def _build_bm25(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)
        self.lengths = np.zeros(len(self.records), dtype=np.float32)

        for index, record in enumerate(self.records):
            tokens = tokenize(record["text"])
            self.lengths[index] = len(tokens)
            for token, count in Counter(tokens).items():
                self.postings[token].append((index, count))

        self.average_length = float(self.lengths.mean()) or 1.0
        total = len(self.records)
        self.idf = {
            token: math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for token, posting in self.postings.items()
        }

    # --------------------------------------------------------
    # SPELLING
    # --------------------------------------------------------

    def correct_spelling(self, text: str) -> str:
        """
        Replace words that appear nowhere in the knowledge base with the
        closest word that does ("prisng" -> "pricing"). Known words are
        never changed. Misspelled questions otherwise score as low as
        off-topic ones and get refused.
        """

        if not hasattr(self, "_vocabulary"):
            self._vocabulary = sorted(
                token for token, posting in self.postings.items()
                if token.isalpha() and len(token) >= 4 and len(posting) >= 2
            )

        def fix(match: re.Match) -> str:
            word = match.group(0)
            lower = word.lower()

            if lower in self.postings or lower in STOPWORDS:
                return word

            candidates = [
                candidate
                for candidate in difflib.get_close_matches(lower, self._vocabulary, n=5, cutoff=0.75)
                if candidate[0] == lower[0]
            ]

            if not candidates:
                return word

            # Closest spelling first; among equally close words, the one
            # used on more pages ("pricing" over "proving").
            return max(
                candidates,
                key=lambda candidate: (
                    round(difflib.SequenceMatcher(None, lower, candidate).ratio(), 2),
                    len(self.postings[candidate]),
                ),
            )

        return re.sub(r"[A-Za-z]{4,}", fix, text)

    def lexical_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.records), dtype=np.float32)

        for token in set(tokenize(query)):
            for index, count in self.postings.get(token, ()):
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] += self.idf[token] * count * (self.k1 + 1) / (count + norm)

        return scores

    # --------------------------------------------------------
    # HYBRID SEARCH
    # --------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        candidates: int,
        top_k: int,
        min_relevance: float,
        relative_margin: float,
        max_per_page: int,
        lexical_weight: float = 0.1,
    ) -> list[dict]:
        """
        Return up to top_k hits: {"id", "text", "metadata", "score", "dense"}.
        Empty when even the best dense match is below min_relevance.

        score = cosine similarity + lexical_weight * (BM25 / best BM25).
        Meaning decides the order; an exact keyword match adds at most
        `lexical_weight`, enough to lift "8(5)" or "5,999" matches.
        """

        dense = self.embeddings @ (query_vector / max(np.linalg.norm(query_vector), 1e-12))
        best = float(dense.max())

        if best < min_relevance:
            logger.info(f"Best similarity {best:.3f} below {min_relevance}; treating as out of scope.")
            return []

        lexical = self.lexical_scores(query_text)
        lexical_max = float(lexical.max())
        if lexical_max > 0:
            lexical = lexical / lexical_max

        dense_top = np.argsort(-dense)[:candidates]
        lexical_top = [int(index) for index in np.argsort(-lexical)[:candidates] if lexical[index] > 0]

        fused = {
            int(index): float(dense[index] + lexical_weight * lexical[index])
            for index in set(int(index) for index in dense_top) | set(lexical_top)
        }

        strong_lexical = set(lexical_top[:3])
        hits = []
        per_page = Counter()

        for index in sorted(fused, key=fused.get, reverse=True):
            # Weak semantic matches only survive if they are top keyword hits.
            if dense[index] < best - relative_margin and index not in strong_lexical:
                continue

            record = self.records[index]
            page = record["metadata"].get("source", "")

            if per_page[page] >= max_per_page:
                continue

            per_page[page] += 1
            hits.append({
                "id": record["id"],
                "text": record["text"],
                "metadata": record["metadata"],
                "score": round(fused[index], 5),
                "dense": round(float(dense[index]), 4),
            })

            if len(hits) == top_k:
                break

        return hits
