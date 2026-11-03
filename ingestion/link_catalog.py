"""
Build the next-step link catalog from the scraped site (data/raw).

After an answer, the API offers one relevant link with a follow-up
question ("Do you want to run a free check of your website?"). The
links come only from this catalog, never from the LLM:

- every page in the sitemap (title + meta description), and
- every link the site itself uses (sign up, contact / demo, ...),
  described by the link texts the site writes for it, plus
- links the site does not contain (the free website check on the
  homepage, and the sign-in page when LOGIN_URL is set).

Each entry is embedded with the same model as the knowledge base. The
result is data/vectorstore/link_catalog.npz (vectors + records); the
API reloads it when the file changes. Entries whose text did not change
keep their vector, so the model is loaded only when something changed.

Usage:
    python ingestion/link_catalog.py
"""

from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import hashlib
import json
import logging
import os
import re
import sys

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import (  # noqa: E402
    EMBED_BATCH_SIZE,
    LINK_CATALOG_FILE,
    LOGIN_URL,
    RAW_DIR,
    embedding_kwargs,
    use_offline_model_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

HOMEPAGE = "https://consiva.ai/"

# Markdown links to consiva.ai or the app; images ("![...](...)") are skipped.
LINK = re.compile(r"(?<!!)\[([^\]]{2,120})\]\((https://(?:app\.)?consiva\.ai[^)\s]*)\)")

# Links the website itself does not contain.
EXTRA_LINKS = [
    {
        "url": HOMEPAGE,
        "kind": "action",
        "label": "Free website DPDPA & GDPR check",
        "text": (
            "Free website DPDPA and GDPR check. Scan my website for cookies, trackers, consent banner "
            "and privacy-policy gaps. Read-only scan, no credit card, set up in 5 minutes."
        ),
    },
]

# Extra description for links whose own text says little. The free account
# is where every DPDP-compliance next step goes (not the contact form).
LINK_DESCRIPTIONS = {
    "https://app.consiva.ai/register?plan=free": (
        "Create a free Consiva account and start DPDP compliance: free DPDP compliance assessment, "
        "consent banner, cookie scanning, consent records and rights requests for your business."
    ),
    "https://app.consiva.ai/register?plan=pro": (
        "Create a Consiva Pro account: the paid plan with higher consent limits, more domains and full automation."
    ),
}

if LOGIN_URL:
    EXTRA_LINKS.append({
        "url": LOGIN_URL,
        "kind": "action",
        "label": "Log in",
        "text": "Log in to the Consiva app. Sign in to my account and dashboard, consent records, forgot password.",
    })


# ============================================================
# URLS
# ============================================================

def canonical_url(url: str) -> str | None:
    """One URL per destination; None for assets and anything off-site."""

    parts = urlsplit(url)

    if parts.netloc == "app.consiva.ai":
        if parts.path.rstrip("/") == "/register":
            # Tracking parameters differ from link to link; only the plan matters.
            # (The API adds utm_source and the site's consent defaults when it offers the link.)
            plan = parse_qs(parts.query).get("plan", ["free"])[0]
            return f"https://app.consiva.ai/register?plan={plan}"
        return f"https://app.consiva.ai{parts.path.rstrip('/')}"

    if parts.netloc != "consiva.ai" or "/assets/" in parts.path:
        return None

    if re.search(r"\.(jpe?g|png|svg|webp|gif|pdf)$", parts.path, re.IGNORECASE):
        return None

    path = parts.path.rstrip("/")
    return f"https://consiva.ai{path}" if path else HOMEPAGE


def link_kind(url: str) -> str:
    """action (sign up, contact) or the kind of page, used to rank candidates."""

    if url.startswith("https://app.consiva.ai") or url.rstrip("/").endswith("/contact"):
        return "action"

    for prefix, kind in (("/legal/", "legal"), ("/blog/", "blog"), ("/dpdp/", "guide"), ("/industry/", "industry")):
        if prefix in url:
            return kind

    return "product"


# ============================================================
# CATALOG
# ============================================================

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_\\#]", "", text)).strip()


def build_entries(raw_pages: list[dict]) -> list[dict]:
    """
    Catalog entries {"url", "kind", "label", "text"} from Firecrawl pages
    ({"url", "markdown", "metadata"}). `text` is what gets embedded.
    """

    pages = {}
    anchors = defaultdict(Counter)

    for page in raw_pages:
        url = canonical_url(page.get("url") or "")
        metadata = page.get("metadata") or {}

        if url:
            pages[url] = {
                "title": clean_text(re.split(r"\s+\|\s+", metadata.get("title") or "")[0]),
                "description": clean_text(metadata.get("description") or ""),
            }

        for match in LINK.finditer(page.get("markdown") or ""):
            target = canonical_url(match.group(2))
            text = clean_text(match.group(1))
            if target and text:
                anchors[target][text] += 1

    extras = {extra["url"]: extra for extra in EXTRA_LINKS}
    entries = []

    for url in sorted(set(pages) | set(anchors)):
        if url in extras:
            continue

        page = pages.get(url, {})
        link_texts = [text for text, _ in anchors[url].most_common(10)]
        label = page.get("title") or (link_texts[0] if link_texts else "")

        if not label:
            continue

        parts = [page.get("title", ""), page.get("description", ""), LINK_DESCRIPTIONS.get(url, "")]
        if link_texts:
            parts.append("Linked as: " + "; ".join(link_texts))

        entries.append({
            "url": url,
            "kind": link_kind(url),
            "label": label[:100],
            "text": ". ".join(part for part in parts if part)[:1500],
        })

    entries.extend(dict(extra) for extra in extras.values())

    return entries


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_previous_vectors(path: Path = LINK_CATALOG_FILE) -> dict[str, np.ndarray]:
    """text hash -> vector from the last catalog, so unchanged entries are not embedded again."""

    if not path.exists():
        return {}

    try:
        with np.load(path) as catalog:
            vectors = catalog["vectors"]
            records = json.loads(str(catalog["records"]))
    except Exception as error:
        logger.warning("Could not read the previous link catalog (%s); embedding everything.", error)
        return {}

    return {record["text_hash"]: vectors[index] for index, record in enumerate(records) if "text_hash" in record}


def embed(entries: list[dict], previous: dict[str, np.ndarray]) -> np.ndarray:
    for entry in entries:
        entry["text_hash"] = text_hash(entry["text"])

    missing = [entry["text"] for entry in entries if entry["text_hash"] not in previous]
    logger.info("Link catalog: %d entries, %d to embed", len(entries), len(missing))

    new_vectors = {}

    if missing:
        from langchain_huggingface import HuggingFaceEmbeddings

        use_offline_model_cache()
        model = HuggingFaceEmbeddings(**embedding_kwargs())

        for start in range(0, len(missing), EMBED_BATCH_SIZE):
            batch = missing[start:start + EMBED_BATCH_SIZE]
            for text, vector in zip(batch, model.embed_documents(batch)):
                new_vectors[text_hash(text)] = np.asarray(vector, dtype=np.float32)

    return np.stack([
        previous[entry["text_hash"]] if entry["text_hash"] in previous else new_vectors[entry["text_hash"]]
        for entry in entries
    ]).astype(np.float32)


def save(entries: list[dict], vectors: np.ndarray, path: Path = LINK_CATALOG_FILE) -> None:
    """Atomic replace, like the search snapshot."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = path.with_name(path.stem + ".tmp.npz")
    np.savez(temp_file, vectors=vectors, records=np.array(json.dumps(entries, ensure_ascii=False)))
    os.replace(temp_file, path)


def main() -> int:
    files = sorted(RAW_DIR.glob("*.json"))

    if not files:
        logger.error("No scraped pages in %s", RAW_DIR)
        return 1

    raw_pages = []
    for file_path in files:
        try:
            raw_pages.append(json.loads(file_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Skipping %s: %s", file_path.name, error)

    entries = build_entries(raw_pages)

    if not entries:
        logger.error("No links found; keeping the previous catalog.")
        return 1

    vectors = embed(entries, load_previous_vectors())
    save(entries, vectors)

    kinds = Counter(entry["kind"] for entry in entries)
    logger.info("Link catalog saved: %d links (%s)", len(entries), ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        logger.exception("Link catalog failed: %s", error)
        sys.exit(1)
