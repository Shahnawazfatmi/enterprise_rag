"""
The next step offered after an answer: one relevant link from the site
and a short follow-up question, for example

    Do you want to run a free check of your website?   [Scan my website]

1. Shortlist (no LLM): rank the link catalog (ingestion/link_catalog.py)
   by similarity to the question and to the answer, drop pages the answer
   already cites, keep the best few above NEXT_STEP_MIN_SCORE.
2. Pick (one short LLM call): the model chooses one candidate or none
   and writes the question and button text. The URL always comes from
   the catalog, so the model can never invent a link.
"""

from pathlib import Path
import json
import logging

import numpy as np
from pydantic import BaseModel, Field

from config import LINK_CATALOG_FILE, NEXT_STEP_UTM

logger = logging.getLogger(__name__)

# Final score = 0.6 * question similarity + 0.4 * (question + answer) similarity.
QUESTION_WEIGHT = 0.6

# Doing something (sign up, contact, scan) is the most useful next step
# when it is relevant at all; policy pages rarely are.
KIND_BONUS = {"action": 0.03, "legal": -0.03}

MAX_QUESTION_CHARS = 140
MAX_LABEL_CHARS = 40


# ============================================================
# CATALOG
# ============================================================

class LinkCatalog:

    def __init__(self, records: list[dict], vectors: np.ndarray):
        if not records:
            raise ValueError("The link catalog is empty.")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.clip(norms, 1e-12, None)
        self.records = records

    @classmethod
    def load(cls, path: Path = LINK_CATALOG_FILE) -> "LinkCatalog":
        with np.load(path) as catalog:
            vectors = catalog["vectors"].astype(np.float32)
            records = json.loads(str(catalog["records"]))
        return cls(records, vectors)

    def __len__(self) -> int:
        return len(self.records)

    def shortlist(
        self,
        question_vector: np.ndarray,
        answer_vector: np.ndarray,
        exclude_urls: set[str],
        limit: int,
        min_score: float,
    ) -> list[dict]:
        """
        Best candidates first: {"url", "kind", "label", "text", "score"}.
        Pages the answer already cites are dropped, but action links are
        kept: "book a demo" cites the contact page and should still offer it.
        Every action link that clears min_score is added after the best
        `limit` candidates (there are only a few), so the LLM can always
        choose between sign-up and contact.
        """

        def unit(vector):
            vector = np.asarray(vector, dtype=np.float32)
            return vector / max(float(np.linalg.norm(vector)), 1e-12)

        scores = (
            QUESTION_WEIGHT * (self.vectors @ unit(question_vector))
            + (1 - QUESTION_WEIGHT) * (self.vectors @ unit(answer_vector))
        )
        scores += np.array([KIND_BONUS.get(record["kind"], 0.0) for record in self.records], dtype=np.float32)

        ranked = [
            index for index in np.argsort(-scores)
            if scores[index] >= min_score
            and (self.records[index]["kind"] == "action" or self.records[index]["url"] not in exclude_urls)
        ]

        chosen = ranked[:limit]
        chosen += [index for index in ranked if self.records[index]["kind"] == "action" and index not in chosen]

        return [{**self.records[index], "score": round(float(scores[index]), 3)} for index in chosen]


# ============================================================
# PICK
# ============================================================

PICK_PROMPT = """You choose the next step offered to a website visitor after the assistant answered their question.

Offer a next step only when it is the obvious thing this visitor would do next. Otherwise choose 0.
Showing nothing is better than showing something loosely related.

Offer one when the visitor:
- asks about DPDP compliance for their business, how Consiva helps them comply, or how a Consiva
  compliance feature works (consent, cookies, rights requests, breach reporting, ROPA, DPIA)
  -> the free account sign-up (register?plan=free); this is also the free DPDP compliance assessment
- wants to try or start Consiva, or asks about the free plan -> the free account sign-up
- asks about the Pro plan or its price -> the Pro sign-up
- wants to talk to a person (demo, call, quote, enterprise deal, custom pricing, account trouble) -> contact
- wants their own website checked (cookies, trackers, banner, leaks) -> the website check
- is learning a general topic and one guide covers exactly that topic in more depth -> that guide

The contact page is only for talking to a person, never for compliance help: compliance goes to the
free account sign-up.

Choose 0 when the question is about a policy or support detail (refunds, terms, support hours, SLAs),
a single plain fact the answer already settles, a greeting, or anything off-topic, and when no
option matches the visitor's actual intent. Never offer the website check just because nothing else fits.

Then write, fitted to this visitor's question and the answer they just read:
- question: a short yes/no question (at most 12 words), in the visitor's language, about the
  specific thing they asked about. Name their topic, do not reuse a generic line. For example:
  cookie consent -> "Do you want to set up DPDP-compliant cookie consent for free?"
  rights requests -> "Do you want to start handling rights requests in Consiva?"
  breach reporting -> "Do you want to set up CERT-In breach tracking now?"
  a general "how do we comply" question -> "Do you want a free DPDP compliance assessment?"
- button: the button text (2 to 4 words) for that same action, e.g. "Set up cookie consent",
  "Start free assessment", "Create free account", "Book a demo".
Do not mention prices or promises that are not in the option."""


class NextStepChoice(BaseModel):
    option: int = Field(description="Number of the chosen option, or 0 for none")
    question: str = Field(description="Short yes/no question to the visitor; empty when option is 0")
    button: str = Field(description="Button text, 2 to 4 words; empty when option is 0")


def pick_message(question: str, answer: str, candidates: list[dict]) -> str:
    options = "\n".join(
        f"{number}. [{candidate['kind']}] {candidate['label']} ({candidate['url']}): {candidate['text'][:300]}"
        for number, candidate in enumerate(candidates, start=1)
    )
    return f"Visitor question:\n{question}\n\nAssistant answer:\n{answer[:2000]}\n\nOptions:\n{options}"


# The website's own sign-up links carry these consent defaults; the app expects them.
APP_LINK_PARAMS = "ga_consent=0&ads_consent=0"


def with_utm(url: str) -> str:
    params = [NEXT_STEP_UTM] if NEXT_STEP_UTM else []

    if url.startswith("https://app.consiva.ai/register"):
        params.append(APP_LINK_PARAMS)

    if not params:
        return url
    return f"{url}{'&' if '?' in url else '?'}{'&'.join(params)}"


def choose_next_step(question: str, answer: str, candidates: list[dict], llms) -> dict | None:
    """
    {"question", "label", "url"} for the option the LLM picks, or None.
    `llms` are (name, model) pairs (see llm.load_llms); the first that
    answers wins. Any failure means no next step, never an error.
    """

    if not candidates:
        return None

    messages = [("system", PICK_PROMPT), ("human", pick_message(question, answer, candidates))]

    for name, llm in llms:
        try:
            choice = llm.with_structured_output(NextStepChoice).invoke(messages)
        except Exception as error:
            logger.warning(f"Next-step pick failed ({name}): {str(error)[:300]}")
            continue

        if not isinstance(choice, NextStepChoice) or not 1 <= choice.option <= len(candidates):
            return None

        candidate = candidates[choice.option - 1]
        text = " ".join(choice.question.split())
        label = " ".join(choice.button.split())

        if not text or len(text) > MAX_QUESTION_CHARS:
            return None

        if not label or len(label) > MAX_LABEL_CHARS:
            label = "Open"

        return {"question": text, "label": label, "url": with_utm(candidate["url"])}

    return None
