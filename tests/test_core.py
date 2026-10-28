"""
Fast unit tests (no model download, no network, no LLM).

Run from the project root:
    python -m unittest discover -s tests -v
"""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.index import SearchIndex  # noqa: E402
from app.router import classify_query  # noqa: E402
from chunker.chunking import parse_front_matter, process_file  # noqa: E402
from ingestion.master_clean import render_page, to_blocks  # noqa: E402

PRICING_MARKDOWN = """↑

[Home](https://consiva.ai/) › [Pricing](https://consiva.ai/pricing#)

⏳ Limited Time Offer

Basic

₹2,499

₹0

Cookie Consents 1,000 / Mo

Domains 1

✓DPDP Banner (English & Hindi)

✗WordPress Plugin

MOST POPULAR

Pro

₹5,999

Domains 5 + Add-Ons

✓WordPress Plugin

Enterprise Plan

Enterprise

Custom

✓SSO / SAML

### Add-On Packs

Domain Pack

+10 domains

₹80,000/yr

## Common Pricing Questions

Is there a free trial for paid plans?+

The Free plan is permanently free.
"""

HOMEPAGE_MARKDOWN = """# Platform

Compliance Readiness Report

Sample preview

Cookie Consent

82%

Illustrative data. Scan your site to get your own results.

Real paragraph about consent.

[Start Free](https://app.consiva.ai/register) [See the Platform](https://consiva.ai/#platform)

Contents

1. [Summary](https://consiva.ai/legal/x#s1)

![logo](https://consiva.ai/logo.png)

| Plan | Price |
| --- | --- |
| Pro | ₹5,999 |

Email [support@consiva.ai](mailto:support@consiva.ai)
"""


def headings(blocks: list[str]) -> list[str]:
    return [block for block in blocks if block.startswith("#")]


class CleanerTests(unittest.TestCase):

    def test_pricing_plans_become_headings(self):
        found = headings(to_blocks(PRICING_MARKDOWN))
        self.assertIn("## Basic plan (Limited Time Offer)", found)
        self.assertIn("## Pro plan (Most Popular)", found)
        self.assertIn("## Enterprise plan", found)

    def test_add_on_heading_is_not_child_of_last_plan(self):
        found = headings(to_blocks(PRICING_MARKDOWN))
        self.assertIn("## Add-On Packs", found)
        self.assertIn("### Domain Pack", found)

    def test_faq_question_becomes_heading(self):
        found = headings(to_blocks(PRICING_MARKDOWN))
        self.assertIn("### Is there a free trial for paid plans?", found)

    def test_feature_marks_and_prices_are_explicit(self):
        text = "\n".join(to_blocks(PRICING_MARKDOWN))
        self.assertIn("Not included: WordPress Plugin", text)
        self.assertIn("Included: SSO / SAML", text)
        self.assertIn("Regular price: ₹2,499", text)
        self.assertIn("Price: ₹0", text)

    def test_noise_is_removed(self):
        text = "\n".join(to_blocks(HOMEPAGE_MARKDOWN + PRICING_MARKDOWN))
        for noise in ("82%", "Sample preview", "Start Free", "Summary", "logo", "↑", "[Home]"):
            self.assertNotIn(noise, text)

    def test_tables_and_contact_lines_are_kept(self):
        text = "\n".join(to_blocks(HOMEPAGE_MARKDOWN))
        self.assertIn("| --- | --- |", text)
        self.assertIn("support@consiva.ai", text)
        self.assertIn("Real paragraph about consent.", text)


class ChunkerTests(unittest.TestCase):

    def chunks_for(self, blocks: list[str]) -> list[dict]:
        page = render_page("Pricing", "https://consiva.ai/pricing", "Plans and prices for Consiva, from free to enterprise.", blocks)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "consiva.ai_pricing.txt"
            path.write_text(page, encoding="utf-8")
            return process_file(path)

    def test_front_matter(self):
        meta, body = parse_front_matter("---\ntitle: T\nurl: https://x\n---\n\n# H\n")
        self.assertEqual(meta, {"title": "T", "url": "https://x"})
        self.assertTrue(body.strip().startswith("# H"))

    def test_every_chunk_has_page_breadcrumb(self):
        for chunk in self.chunks_for(to_blocks(PRICING_MARKDOWN)):
            self.assertTrue(chunk["content"].startswith("Page: Pricing"))
            self.assertEqual(chunk["url"], "https://consiva.ai/pricing")

    def test_plan_details_stay_with_plan_name(self):
        chunks = self.chunks_for(to_blocks(PRICING_MARKDOWN))
        pro = [chunk for chunk in chunks if "Domains 5" in chunk["content"]]
        self.assertTrue(pro)
        self.assertIn("Section: Pro plan", pro[0]["content"])
        self.assertNotIn("Domains 1", pro[0]["content"])

    def test_summary_chunk_from_description(self):
        chunks = self.chunks_for(to_blocks(PRICING_MARKDOWN))
        self.assertTrue(any("Summary: Plans and prices" in chunk["content"] for chunk in chunks))

    def test_small_sibling_sections_are_merged(self):
        blocks = ["# Guide", "## Part A", "Short text A.", "## Part B", "Short text B."]
        chunks = [chunk for chunk in self.chunks_for(blocks) if "Summary:" not in chunk["content"]]
        self.assertEqual(len(chunks), 1)
        self.assertIn("## Part A", chunks[0]["content"])
        self.assertIn("## Part B", chunks[0]["content"])

    def test_questions_are_not_merged(self):
        blocks = ["## FAQ", "### First question?", "Answer one.", "### Second question?", "Answer two."]
        chunks = [chunk for chunk in self.chunks_for(blocks) if "question?" in chunk["content"]]
        self.assertEqual(len(chunks), 2)


class RouterTests(unittest.TestCase):

    def test_greetings_with_punctuation(self):
        for text in ("hi!", "Hello there", "hey :)", "Good morning."):
            self.assertEqual(classify_query(text), "greeting", text)

    def test_goodbye_and_rag(self):
        self.assertEqual(classify_query("bye"), "goodbye")
        self.assertEqual(classify_query("hi, what does the Pro plan cost?"), "rag")


class FollowUpTests(unittest.TestCase):

    def test_follow_ups_use_previous_question(self):
        from app.rag import clean_history, retrieval_query

        history = clean_history([
            {"role": "user", "content": "How much is the Pro plan?"},
            {"role": "assistant", "content": "It costs 5,999 [1]."},
        ])

        self.assertNotIn("[1]", history[1]["content"])
        self.assertIn("Pro plan", retrieval_query("and for Enterprise?", history))
        self.assertEqual(retrieval_query("What is ROPA?", history), "What is ROPA?")


class SearchIndexTests(unittest.TestCase):

    def setUp(self):
        records = [
            {"id": "a", "text": "Pro plan price 5,999 per month", "metadata": {"source": "pricing.txt"}},
            {"id": "b", "text": "Section 8(5) requires reasonable security safeguards", "metadata": {"source": "act.txt"}},
            {"id": "c", "text": "Cookie banner languages", "metadata": {"source": "cookies.txt"}},
        ]
        embeddings = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.index = SearchIndex(records, embeddings)

    def search(self, vector, text, min_relevance=0.3):
        return self.index.search(
            np.array(vector, dtype=np.float32), text,
            candidates=3, top_k=3, min_relevance=min_relevance,
            relative_margin=0.2, max_per_page=2,
        )

    def test_out_of_scope_returns_nothing(self):
        self.assertEqual(self.search([0.1, 0.1, 0.1], "capital of france", min_relevance=0.9), [])

    def test_best_dense_match_first(self):
        hits = self.search([0.9, 0.1, 0.0], "how much is pro")
        self.assertEqual(hits[0]["id"], "a")

    def test_spelling_correction_uses_known_words_only(self):
        records = [
            {"id": str(index), "text": "pricing plans and consent management", "metadata": {"source": f"{index}.txt"}}
            for index in range(3)
        ]
        index = SearchIndex(records, np.eye(3, dtype=np.float32))

        self.assertEqual(index.correct_spelling("prisng plans"), "pricing plans")
        self.assertEqual(index.correct_spelling("consnt managment"), "consent management")
        self.assertEqual(index.correct_spelling("pricing"), "pricing")

    def test_legal_references_are_single_tokens(self):
        from app.index import tokenize

        self.assertEqual(tokenize("section 8(5) and 17(1)(f)"), ["section", "8(5)", "17(1)(f)"])

    def test_exact_keyword_match_is_kept(self):
        hits = self.search([0.9, 0.1, 0.0], "section 8(5)")
        self.assertIn("b", [hit["id"] for hit in hits])


class ChunkIdTests(unittest.TestCase):

    def test_chunk_ids_depend_on_content_only(self):
        from ingestion.vector_updater import build_chunk_id

        self.assertEqual(build_chunk_id("p", "text"), build_chunk_id("p", "text"))
        self.assertNotEqual(build_chunk_id("p", "text"), build_chunk_id("p", "text 2"))
        self.assertTrue(build_chunk_id("p", "text").startswith("p::"))


class FakeLLM:
    """Streams the given pieces, then raises `error` if set."""

    def __init__(self, pieces, error=None):
        self.pieces, self.error = pieces, error

    def stream(self, messages):
        from types import SimpleNamespace

        for piece in self.pieces:
            yield SimpleNamespace(content=piece)
        if self.error:
            raise self.error


class StreamingTests(unittest.TestCase):

    def stream(self, *llms):
        from app.llm import generate_answer_stream

        return list(generate_answer_stream("q", "[1] source", [(f"llm{i}", llm) for i, llm in enumerate(llms)]))

    def test_pieces_are_streamed_in_order(self):
        events = self.stream(FakeLLM(["Pro costs ", "5,999 [1]."]))
        self.assertEqual(events, [("delta", "Pro costs "), ("delta", "5,999 [1].")])

    def test_fallback_before_any_text_needs_no_reset(self):
        events = self.stream(FakeLLM([], RuntimeError("down")), FakeLLM(["Answer."]))
        self.assertEqual(events, [("delta", "Answer.")])

    def test_fallback_after_partial_text_resets(self):
        events = self.stream(FakeLLM(["Half"], RuntimeError("cut")), FakeLLM(["Full answer."]))
        self.assertEqual(events, [("delta", "Half"), ("reset", ""), ("delta", "Full answer.")])

    def test_empty_stream_counts_as_failure(self):
        events = self.stream(FakeLLM([" "]), FakeLLM(["Answer."]))
        self.assertEqual(events, [("delta", " "), ("reset", ""), ("delta", "Answer.")])

    def test_all_failing_raises_quota_error(self):
        from app.llm import LLMUnavailableError

        with self.assertRaisesRegex(LLMUnavailableError, "quota"):
            self.stream(FakeLLM([], RuntimeError("429 RESOURCE_EXHAUSTED")))


if __name__ == "__main__":
    unittest.main()
