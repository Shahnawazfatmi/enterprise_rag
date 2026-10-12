"""
Heading-aware chunking for the structured pages in data/new_structured.

Each chunk is:

    Page: <page title>
    Section: <H1 > H2 > H3 path>

    <body text>

Why:
- The page title and heading path are part of the embedded text, so
  search can tell "Domains 5" on the Pro plan from "Domains 1" on Basic.
- Small sibling sections under the same parent are merged until a chunk
  reaches about CHUNK_TARGET_CHARS, so one-line sections do not become
  weak vectors of their own.
- FAQ questions and pricing plans / packs always get their own chunk.
- Sections longer than CHUNK_MAX_CHARS are split on paragraph and line
  boundaries, and every piece keeps the same breadcrumb.
- Each page also gets one summary chunk built from its meta description.
"""

from pathlib import Path
import json
import re
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import (  # noqa: E402
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    NEW_STRUCTURED_DIR,
)

INPUT_DIR = NEW_STRUCTURED_DIR
OUTPUT_FILE = BASE_DIR / "data" / "chunks" / "chunks.jsonl"

HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_TARGET_CHARS,
    chunk_overlap=CHUNK_OVERLAP_CHARS,
    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
)


# ============================================================
# PARSING
# ============================================================

def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split the '---' header written by master_clean.py from the body."""

    meta = {}

    if not text.startswith("---\n"):
        return meta, text

    end = text.find("\n---\n", 4)

    if end == -1:
        return meta, text

    for line in text[4:end].splitlines():
        key, separator, value = line.partition(":")
        if separator:
            meta[key.strip()] = value.strip()

    return meta, text[end + 5:]


def parse_sections(body: str) -> list[dict]:
    """
    One section per heading. `path` holds the active headings as
    (level, title) pairs, including the section's own heading.
    """

    sections = []
    path = []
    lines = []

    def close_section():
        content = "\n".join(lines).strip()
        if content:
            sections.append({"path": list(path), "body": content})

    for line in body.splitlines():
        match = HEADING_PATTERN.match(line.rstrip())

        if not match:
            lines.append(line.rstrip())
            continue

        close_section()
        lines = []

        level = len(match.group(1))
        path = [item for item in path if item[0] < level]
        path.append((level, match.group(2).strip()))

    close_section()

    return sections


PLAN_HEADING = re.compile(r"\b(plan|pack)\b(\s*\([^)]*\))?$", re.IGNORECASE)


def is_atomic(section: dict) -> bool:
    """
    FAQ questions and pricing plans / packs always get their own chunk,
    so one plan's limits are never mixed into another plan's chunk.
    """

    if not section["path"]:
        return False

    heading = section["path"][-1][1]
    return heading.endswith("?") or bool(PLAN_HEADING.search(heading))


# ============================================================
# CHUNK TEXT
# ============================================================

def breadcrumb(title: str, path: list[tuple[int, str]]) -> str:
    names = [name for _, name in path]

    # The H1 usually repeats the page title.
    if names and names[0].lower() == title.lower():
        names = names[1:]

    lines = [f"Page: {title}"]

    if names:
        lines.append("Section: " + " > ".join(names))

    return "\n".join(lines)


def headings_metadata(path: list[tuple[int, str]]) -> dict:
    metadata = {f"H{level}": "" for level in range(1, 7)}
    for level, name in path:
        metadata[f"H{level}"] = name
    return metadata


def make_chunk(meta: dict, source: str, path: list, body: str) -> dict:
    title = meta.get("title") or source
    crumb = breadcrumb(title, path)

    return {
        "content": f"{crumb}\n\n{body.strip()}",
        "source": source,
        "url": meta.get("url", ""),
        "title": title,
        "section": " > ".join(name for _, name in path),
        **headings_metadata(path),
    }


# ============================================================
# GROUPING
# ============================================================

def common_prefix(first: list, second: list) -> list:
    size = 0
    while size < min(len(first), len(second)) and first[size] == second[size]:
        size += 1
    return first[:size]


def group_sections(sections: list[dict]) -> list[dict]:
    """
    Merge neighbouring small sections that live under the same parent
    heading as the group's first section.

    A group is {"path": common heading path, "parts": [(path, body)]}.
    """

    groups = []

    for section in sections:
        size = len(section["body"])
        current = groups[-1] if groups else None

        if current and not current["atomic"] and not is_atomic(section):
            shared = common_prefix(current["path"], section["path"])
            first_path = current["parts"][0][0]
            same_parent = len(shared) >= len(first_path) - 1
            fits = (
                (current["size"] < CHUNK_MIN_CHARS or current["size"] + size <= CHUNK_TARGET_CHARS)
                and current["size"] + size <= CHUNK_MAX_CHARS
            )

            if same_parent and fits:
                current["path"] = shared
                current["parts"].append((section["path"], section["body"]))
                current["size"] += size
                continue

        groups.append({
            "path": section["path"],
            "parts": [(section["path"], section["body"])],
            "size": size,
            "atomic": is_atomic(section),
        })

    return groups


def render_group(group: dict) -> str:
    """Body text of a group; headings below the group path stay inline, once each."""

    lines = []
    depth = len(group["path"])
    previous = group["path"]

    for sub_path, body in group["parts"]:
        start = max(depth, len(common_prefix(previous, sub_path)))
        for level, name in sub_path[start:]:
            lines.append(f"{'#' * level} {name}")
        lines.append(body)
        lines.append("")
        previous = sub_path

    return "\n".join(lines).strip()


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(file_path: Path) -> list[dict]:

    text = file_path.read_text(encoding="utf-8", errors="replace")
    meta, body = parse_front_matter(text)
    source = file_path.name

    chunks = []

    description = meta.get("description", "")
    if len(description) >= 40:
        chunks.append(make_chunk(meta, source, [], f"Summary: {description}"))

    for group in group_sections(parse_sections(body)):
        group_body = render_group(group)

        if len(group_body) <= CHUNK_MAX_CHARS:
            chunks.append(make_chunk(meta, source, group["path"], group_body))
            continue

        for piece in SPLITTER.split_text(group_body):
            chunks.append(make_chunk(meta, source, group["path"], piece))

    # Drop exact duplicates within the page.
    unique = []
    seen = set()

    for chunk in chunks:
        key = chunk["content"]
        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    return unique


# ============================================================
# DEBUG EXPORT
# ============================================================

def main():
    """Write every chunk to data/chunks/chunks.jsonl for inspection."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.txt"))
    sizes = []

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for file_path in files:
            for chunk in process_file(file_path):
                sizes.append(len(chunk["content"]))
                output.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    sizes.sort()
    print(f"Documents : {len(files)}")
    print(f"Chunks    : {len(sizes)}")
    if sizes:
        print(f"Median    : {sizes[len(sizes) // 2]} chars")
        print(f"Under 300 : {sum(size < 300 for size in sizes)}")
    print(f"Output    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
