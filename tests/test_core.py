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

    def test_mean_vector_of_stored_chunks(self):
        np.testing.assert_allclose(self.index.mean_vector(["a", "c", "gone"]), [0.5, 0, 0.5])
        self.assertIsNone(self.index.mean_vector(["gone"]))

    def test_next_step_answer_vector_uses_cited_chunks_without_embedding(self):
        from app.rag import ConsivaRAG

        rag = ConsivaRAG.__new__(ConsivaRAG)
        rag._index = self.index
        rag.reload_index_if_changed = lambda: None
        hits = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

        np.testing.assert_allclose(rag.answer_vector("Price [1], cookies [3].", hits), [0.5, 0, 0.5])
        np.testing.assert_allclose(rag.answer_vector("No citations.", hits), [1 / 3] * 3, rtol=1e-6)
        self.assertIsNone(rag.answer_vector("No citations.", []))


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


class AnswerLengthTests(unittest.TestCase):

    def test_detail_only_when_asked(self):
        from app.router import wants_detail

        for text in ("Explain the DPDP penalties in detail", "tell me more", "Give me the full list of rights",
                     "walk me through the breach process step by step"):
            self.assertTrue(wants_detail(text), text)

        for text in ("What does the Pro plan cost?", "What are the penalties under the DPDP Act?",
                     "Is there a free plan and what are its limits?"):
            self.assertFalse(wants_detail(text), text)

    def test_steps_questions_get_numbered_lists(self):
        from app.llm import system_prompt
        from app.router import wants_steps

        for text in ("What should a company do after a personal data breach?", "How do I set up the banner?",
                     "How can we comply with the DPDP Act?", "What are the steps to file a grievance?"):
            self.assertTrue(wants_steps(text), text)

        for text in ("What does the Pro plan cost?", "What is a consent manager?",
                     "What are the penalties under the DPDP Act?"):
            self.assertFalse(wants_steps(text), text)

        self.assertIn("This question asks what to do", system_prompt(steps=True))
        self.assertNotIn("This question asks what to do", system_prompt())

    def test_detailed_prompt_changes_only_the_format_rule(self):
        from app.llm import system_prompt

        normal, detailed = system_prompt(), system_prompt(detailed=True)
        self.assertIn("numbered list", normal)
        self.assertIn("The facts decide how many points", normal)
        self.assertIn("detailed answer", detailed)
        self.assertEqual(normal.split("\n6.")[0], detailed.split("\n6.")[0])
        self.assertEqual(normal.split("\n7.")[1], detailed.split("\n7.")[1])


class LinkCatalogTests(unittest.TestCase):

    def test_canonical_urls(self):
        from ingestion.link_catalog import canonical_url

        self.assertEqual(canonical_url("https://app.consiva.ai/register?ga_consent=0&ads_consent=0"),
                         "https://app.consiva.ai/register?plan=free")
        self.assertEqual(canonical_url("https://app.consiva.ai/register?plan=pro&utm_source=x"),
                         "https://app.consiva.ai/register?plan=pro")
        self.assertEqual(canonical_url("https://consiva.ai/contact#form"), "https://consiva.ai/contact")
        self.assertIsNone(canonical_url("https://consiva.ai/assets/images/a.jpg"))
        self.assertIsNone(canonical_url("https://example.com/page"))

    def test_entries_from_pages_and_links(self):
        from ingestion.link_catalog import HOMEPAGE, build_entries

        pages = [
            {"url": "https://consiva.ai/pricing", "metadata": {"title": "Pricing | Consiva", "description": "Plans"},
             "markdown": "[Get Started Free](https://app.consiva.ai/register?ga_consent=0) "
                         "[Start Free](https://app.consiva.ai/register?plan=free&utm_source=a) "
                         "[Book a Demo](https://consiva.ai/contact) ![x](https://consiva.ai/assets/a.png)"},
            {"url": "https://consiva.ai/", "metadata": {"title": "Home"}, "markdown": ""},
        ]
        entries = {entry["url"]: entry for entry in build_entries(pages)}

        self.assertEqual(entries["https://app.consiva.ai/register?plan=free"]["kind"], "action")
        self.assertIn("Get Started Free", entries["https://app.consiva.ai/register?plan=free"]["text"])
        self.assertEqual(entries["https://consiva.ai/contact"]["kind"], "action")
        self.assertEqual(entries["https://consiva.ai/pricing"]["label"], "Pricing")
        # The homepage is offered as the free website check, once.
        self.assertIn("Scan my website", entries[HOMEPAGE]["text"])
        self.assertFalse(any("/assets/" in url for url in entries))


class NextStepTests(unittest.TestCase):

    def setUp(self):
        from app.next_step import LinkCatalog

        records = [
            {"url": "https://consiva.ai/pricing", "kind": "product", "label": "Pricing", "text": "plans"},
            {"url": "https://app.consiva.ai/register?plan=free", "kind": "action", "label": "Sign up", "text": "start"},
            {"url": "https://consiva.ai/blog/x", "kind": "blog", "label": "Blog", "text": "other"},
        ]
        vectors = np.array([[1, 0, 0], [0.8, 0.6, 0], [0, 0, 1]], dtype=np.float32)
        self.catalog = LinkCatalog(records, vectors)

    def shortlist(self, exclude=(), limit=1, min_score=0.3):
        vector = np.array([1, 0, 0], dtype=np.float32)
        return [c["url"] for c in self.catalog.shortlist(vector, vector, set(exclude), limit, min_score)]

    def test_relevant_actions_are_always_offered(self):
        self.assertEqual(self.shortlist(limit=1), ["https://consiva.ai/pricing", "https://app.consiva.ai/register?plan=free"])

    def test_sign_up_links_are_marked_as_chatbot(self):
        from app.next_step import with_utm

        self.assertEqual(with_utm("https://app.consiva.ai/register?plan=free"),
                         "https://app.consiva.ai/register?plan=free&utm_source=chatbot&ga_consent=0&ads_consent=0")
        self.assertEqual(with_utm("https://consiva.ai/contact"), "https://consiva.ai/contact?utm_source=chatbot")

    def test_cited_pages_and_weak_links_are_dropped(self):
        urls = self.shortlist(exclude={"https://consiva.ai/pricing"}, limit=3)
        self.assertEqual(urls, ["https://app.consiva.ai/register?plan=free"])

    def test_cited_action_links_are_still_offered(self):
        urls = self.shortlist(exclude={"https://app.consiva.ai/register?plan=free"}, limit=3)
        self.assertIn("https://app.consiva.ai/register?plan=free", urls)

    def test_pick_uses_the_catalog_url(self):
        from types import SimpleNamespace
        from app.next_step import NextStepChoice, choose_next_step

        def llm(choice=None, error=None):
            def invoke(messages):
                if error:
                    raise error
                return choice
            return SimpleNamespace(with_structured_output=lambda schema: SimpleNamespace(invoke=invoke))

        candidates = [{"url": "https://consiva.ai/contact", "kind": "action", "label": "Contact", "text": "demo"}]
        pick = NextStepChoice(option=1, question="Do you want to book a demo?", button="Book a demo")

        step = choose_next_step("q", "a", candidates, [("down", llm(error=RuntimeError("x"))), ("ok", llm(pick))])
        self.assertTrue(step["url"].startswith("https://consiva.ai/contact"))
        self.assertIn("utm_source=chatbot", step["url"])
        self.assertEqual(step["label"], "Book a demo")

        none = NextStepChoice(option=0, question="", button="")
        self.assertIsNone(choose_next_step("q", "a", candidates, [("ok", llm(none))]))
        self.assertIsNone(choose_next_step("q", "a", candidates, [("ok", llm(NextStepChoice(option=5, question="x", button="y")))]))


if __name__ == "__main__":
    unittest.main()
