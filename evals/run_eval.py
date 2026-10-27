"""
Evaluate retrieval and answer quality against evals/questions.jsonl.

Usage (from the project root):
    python evals/run_eval.py                 # retrieval only (no LLM calls)
    python evals/run_eval.py --answers       # retrieval + LLM answers
    python evals/run_eval.py --answers --out evals/results/after.json

Metrics:
    retrieval_hit@k  expected page appears in the retrieved chunks
    context_facts    the expected facts appear in the retrieved chunk text
    fact_accuracy    every expected fact appears in the answer
    refusal_ok       off-topic questions get the "not found" answer
"""

from pathlib import Path
import argparse
import json
import re
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUESTIONS_FILE = PROJECT_ROOT / "evals" / "questions.jsonl"

NOT_FOUND_PATTERN = re.compile(
    r"couldn.t find|could not find|not (?:available|mentioned|covered) in|"
    r"don.t have (?:that|this|information)|no information",
    re.IGNORECASE,
)


def load_questions() -> list[dict]:
    with QUESTIONS_FILE.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def retrieve(rag, question: str) -> tuple[list[str], str]:
    """Return (sources, concatenated chunk text) for the chunks sent to the LLM."""

    # New retriever exposes retrieve(); the legacy one only has a vectorstore.
    if hasattr(rag, "retrieve"):
        hits = rag.retrieve(question)
        return (
            [hit["metadata"].get("source", "") for hit in hits],
            "\n".join(hit["text"] for hit in hits),
        )

    documents = rag.vectorstore.similarity_search(question, k=5)
    return (
        [document.metadata.get("source", "") for document in documents],
        "\n".join(document.page_content for document in documents),
    )


def source_hit(expected: list[str], sources: list[str]) -> bool:
    return any(
        pattern.lower() in source.lower()
        for pattern in expected
        for source in sources
    )


def facts_present(facts: list[str], answer: str) -> bool:
    return all(re.search(fact, answer, re.IGNORECASE) for fact in facts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also generate answers with the LLM")
    parser.add_argument("--out", type=Path, help="write per-question results to this JSON file")
    parser.add_argument("--only", help="comma-separated question ids (e.g. to stay within an LLM quota)")
    args = parser.parse_args()

    from app.llm import LLMUnavailableError
    from app.rag import ConsivaRAG

    rag = ConsivaRAG()
    questions = load_questions()

    if args.only:
        wanted = set(args.only.split(","))
        questions = [item for item in questions if item["id"] in wanted]
    results = []
    llm_failures = 0

    for item in questions:
        sources, context = retrieve(rag, item["question"])
        result = {
            "id": item["id"],
            "question": item["question"],
            "retrieved": sources,
            "retrieval_hit": None,
            "context_has_facts": None,
        }

        if item["sources"]:
            result["retrieval_hit"] = source_hit(item["sources"], sources)

        if item["facts"]:
            result["context_has_facts"] = facts_present(item["facts"], context)

        if args.answers:
            result["answer_ok"] = None

            # After two LLM outages in a row, stop calling it (saves quota).
            # Questions with no retrieved chunks never reach the LLM.
            if llm_failures < 2 or not sources:
                started = time.time()
                try:
                    response = rag.ask_question(item["question"])
                except LLMUnavailableError as error:
                    # Quota or outage: recorded, not counted as a wrong answer.
                    llm_failures += 1
                    result["error"] = str(error)
                else:
                    llm_failures = 0
                    answer = response["answer"]
                    result["answer"] = answer
                    result["sources"] = response.get("sources", [])
                    result["seconds"] = round(time.time() - started, 2)

                    if item.get("expect_not_found"):
                        result["answer_ok"] = bool(NOT_FOUND_PATTERN.search(answer))
                    elif item["facts"]:
                        result["answer_ok"] = facts_present(item["facts"], answer)
                    else:
                        # No fixed facts: only require a real, non-refusal answer.
                        result["answer_ok"] = not NOT_FOUND_PATTERN.search(answer)

        results.append(result)
        status = "hit " if result["retrieval_hit"] else ("n/a " if result["retrieval_hit"] is None else "MISS")
        if result["context_has_facts"] is not None:
            status += " facts" if result["context_has_facts"] else " NO-FACTS"
        answer_status = ""
        if args.answers:
            answer_ok = result.get("answer_ok")
            answer_status = " answer=" + {True: "ok", False: "WRONG", None: "SKIPPED"}[answer_ok]
        print(f"[{status}] {item['id']}{answer_status}")

    scored = [r for r in results if r["retrieval_hit"] is not None]
    hit_rate = sum(r["retrieval_hit"] for r in scored) / len(scored)
    with_facts = [r for r in results if r["context_has_facts"] is not None]
    summary = {
        "retrieval_hit_rate": round(hit_rate, 3),
        "retrieval_scored": len(scored),
        "context_fact_rate": round(sum(r["context_has_facts"] for r in with_facts) / len(with_facts), 3),
        "context_fact_scored": len(with_facts),
    }

    if args.answers:
        answered = [r for r in results if r.get("answer_ok") is not None]
        if answered:
            summary["answer_accuracy"] = round(sum(r["answer_ok"] for r in answered) / len(answered), 3)
        summary["answers_scored"] = len(answered)
        summary["answers_skipped_llm_unavailable"] = len(results) - len(answered)

    print(json.dumps(summary, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
