"""
Clean Firecrawl markdown (data/raw/*.json) into structured text files
(data/new_structured/*.txt) for chunking.

Rules:
- Only real markdown headings from the page are kept as headings.
  (Short Title Case lines are NOT promoted to headings any more; that
  turned UI labels like "Cookie Consent / 82%" into sections.)
- Two page patterns are turned into headings on purpose:
    * FAQ questions, rendered by the site as "Question?+"
    * pricing plans: a short name followed by a price line
- UI noise is removed: back-to-top arrows, breadcrumbs, tables of
  contents, images, call-to-action buttons, demo dashboards.
- Paragraphs repeated on many pages (trust bar, newsletter box, CTAs)
  are kept on the homepage only.
- Each file starts with a small front matter block (title, url,
  description) that the chunker uses for breadcrumbs and citations.
- Output files for pages that no longer exist in data/raw are removed,
  so deleted website pages are detected by compare.py.
"""

from collections import Counter
from pathlib import Path
import json
import re
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "new_structured"

HOMEPAGE_FILE = "consiva.ai_.txt"

# A paragraph found on at least this many pages is boilerplate.
BOILERPLATE_MIN_PAGES = 8


# ============================================================
# NOISE PATTERNS
# ============================================================

# Demo widgets with illustrative numbers. Everything from the start
# line to the end line (inclusive) is removed, so the assistant never
# quotes sample scores or fake dashboard counts as facts.
WIDGET_SPANS = [
    ("Compliance Readiness Report", "Illustrative data."),
    ("app.consiva.ai/dashboard", "Access request"),
]

NOISE_LINES = {
    "contents",
    "on this page",
    "enter your website url",
    "sample preview",
    "read more",
    "share",
}

NOISE_LINE_PATTERNS = [
    re.compile(r"^(?!\|)[\W_]*$"),                   # arrows, separators (not table rows)
    re.compile(r"^\[home\]\(.*›", re.IGNORECASE),    # breadcrumb
    re.compile(r"^!\[[^\]]*\]\([^)]*\)$"),           # image only
    re.compile(r"^\d+ questions?$", re.IGNORECASE),  # FAQ group counter
    re.compile(r"^scan my website", re.IGNORECASE),
]

# A list item that is only an in-page anchor link ("1. [Summary](...#s1)").
TOC_ITEM = re.compile(
    r"^(?:\d+[.)]|[-*])?\s*\[[^\]]+\]\([^)]*#[^)]*\)\s*$"
)

# A line made only of short links, e.g. "[Start Free](...) [See the Platform](...)".
LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")

PRICE_LINE = re.compile(
    r"^(₹\s?[\d,]+(?:\.\d+)?(?:\s?/\s?[a-z]+)?|custom|free)$",
    re.IGNORECASE,
)

EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # pictographs, flags
    "←-⇿"           # arrows
    "⌀-⏿"           # technical symbols (⏳ ⏱)
    "☀-➿"           # misc symbols, dingbats (⚖ ✍)
    "⬀-⯿"
    "️‍"
    "]"
)

MARKDOWN_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>])")
HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


# ============================================================
# INLINE CLEANING
# ============================================================

def clean_inline(text: str) -> str:
    """Remove markdown decoration that adds tokens but no meaning."""

    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = text.replace("**", "").replace("__", "")

    # "[billing@consiva.ai](mailto:...)" -> "billing@consiva.ai (email)", so
    # questions asking for an "email address" match contact lines.
    text = LINK.sub(
        lambda match: f"{match.group(1)} (email)" if match.group(2).startswith("mailto:") else match.group(1),
        text,
    )
    text = MARKDOWN_ESCAPE.sub(r"\1", text)

    # Tick / cross marks used in plan feature lists.
    text = re.sub(r"^✓\s*", "Included: ", text)
    text = re.sub(r"^✗\s*", "Not included: ", text)

    text = EMOJI.sub("", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def is_link_only(line: str) -> bool:
    """True for navigation, call-to-action and "related article" lines made only of links."""

    links = LINK.findall(line)

    if not links:
        return False

    if LINK.sub("", line).strip(" -*·|"):
        return False

    # Keep contact lines such as "[sales@consiva.ai](mailto:...)".
    return not any(url.startswith(("mailto:", "tel:")) for _, url in links)


def is_noise(line: str) -> bool:
    lowered = line.lower()

    if lowered in NOISE_LINES:
        return True

    if any(pattern.match(line) for pattern in NOISE_LINE_PATTERNS):
        return True

    return bool(TOC_ITEM.match(line)) or is_link_only(line)


# ============================================================
# STRUCTURE
# ============================================================

def remove_widget_spans(lines: list[str]) -> list[str]:
    for start_text, end_text in WIDGET_SPANS:
        stripped = [line.strip() for line in lines]

        if start_text not in stripped:
            continue

        start = stripped.index(start_text)
        end = next(
            (
                index
                for index in range(start, len(stripped))
                if stripped[index].startswith(end_text)
            ),
            None,
        )

        if end is not None:
            lines = lines[:start] + lines[end + 1:]

    return lines


def is_plan_name(line: str) -> bool:
    words = line.split()
    return (
        1 <= len(words) <= 3
        and not HEADING.match(line)
        and not PRICE_LINE.match(line)
        and not line.startswith(("-", "Included", "Not included"))
        and not re.search(r"\d", line)
    )


def next_content(lines: list[str], index: int, count: int) -> list[str]:
    """The next `count` non-empty lines after `index`."""
    found = []
    for line in lines[index + 1:]:
        if line:
            found.append(line)
            if len(found) == count:
                break
    return found


def mark_structure(lines: list[str]) -> list[str]:
    """
    Turn FAQ questions and pricing plans into headings, and label
    "old price / new price" pairs. Input lines are already cleaned.
    """

    output = []
    current_level = 1
    generated_level = None  # level of plan headings generated since the last real heading
    pending_badge = None    # "MOST POPULAR" / "Limited Time Offer" before a plan name

    for index, line in enumerate(lines):

        heading = HEADING.match(line)

        # Generated headings below do not change current_level, so all
        # plans (or questions) under one real heading become siblings.
        if heading:
            level = len(heading.group(1))

            # A real heading after generated plan headings must not become
            # their child ("Add-On Packs" is not part of "Enterprise plan").
            if generated_level and level > generated_level:
                level = generated_level
                line = f"{'#' * level} {heading.group(2)}"

            current_level = level
            generated_level = None
            pending_badge = None
            output.append(line)
            continue

        if not line:
            output.append(line)
            continue

        # FAQ question rendered as "Question?+"
        if line.endswith("?+") or (line.endswith("+") and "?" in line):
            level = min(current_level + 1, 6)
            output.append(f"{'#' * level} {line.rstrip('+').strip()}")
            continue

        upcoming = next_content(lines, index, 3)

        # Pricing plan or pack: short name, then a price within two lines.
        if is_plan_name(line) and any(PRICE_LINE.match(item) for item in upcoming[:2]):

            # A badge ("MOST POPULAR") or label ("Enterprise Plan") directly
            # before the real name: remember it and use the next line.
            if upcoming and is_plan_name(upcoming[0]):
                pending_badge = line
                continue

            name = line if line.lower().endswith(("plan", "pack")) else f"{line} plan"
            if pending_badge and pending_badge.lower() not in name.lower():
                name = f"{name} ({pending_badge.title()})"
            level = max(2, min(current_level + 1, 6))
            output.append(f"{'#' * level} {name}")
            generated_level = level
            pending_badge = None
            continue

        # Two price lines in a row: the first is the crossed-out price.
        if PRICE_LINE.match(line) and upcoming and PRICE_LINE.match(upcoming[0]):
            output.append(f"Regular price: {line} (the offer price below applies now)")
            continue

        if PRICE_LINE.match(line):
            output.append(f"Price: {line}")
            continue

        output.append(line)

    return output


def remove_repeated_runs(blocks: list[str]) -> list[str]:
    """Drop a run of paragraphs that is immediately repeated (marquees)."""

    index = 0
    output = []

    while index < len(blocks):
        repeated = False

        for size in range(3, 40):
            first = blocks[index:index + size]
            second = blocks[index + size:index + 2 * size]
            if len(second) == size and first == second:
                output.extend(first)
                index += 2 * size
                repeated = True
                break

        if not repeated:
            output.append(blocks[index])
            index += 1

    return output


def to_blocks(markdown: str) -> list[str]:
    """Clean a page and return its paragraphs/headings in order."""

    text = markdown.replace("\r\n", "\n").replace("\r", "\n")

    # Firecrawl writes line breaks inside link text as "\" + newline.
    text = re.sub(r"\\\n", " ", text)

    raw_lines = [line.strip() for line in text.split("\n")]
    raw_lines = remove_widget_spans(raw_lines)

    lines = []

    for raw in raw_lines:

        if not raw:
            lines.append("")
            continue

        if is_noise(raw):
            continue

        heading = HEADING.match(raw)

        if heading:
            title = clean_inline(heading.group(2))
            if title:
                lines.append(f"{heading.group(1)} {title}")
            continue

        bullet = re.match(r"^(?:[-*•])\s+(.*)$", raw)
        cleaned = clean_inline(bullet.group(1) if bullet else raw)

        if not cleaned or is_noise(cleaned):
            continue

        lines.append(f"- {cleaned}" if bullet else cleaned)

    lines = mark_structure(lines)

    # Paragraph blocks: headings are blocks of their own.
    blocks = []
    current = []

    for line in lines:
        if not line or HEADING.match(line):
            if current:
                blocks.append("\n".join(current))
                current = []
            if line:
                blocks.append(line)
            continue
        current.append(line)

    if current:
        blocks.append("\n".join(current))

    return remove_repeated_runs(blocks)


def normalize_block(block: str) -> str:
    return re.sub(r"\s+", " ", block).strip().lower()


# ============================================================
# PAGE METADATA
# ============================================================

def page_title(data: dict, blocks: list[str]) -> str:
    title = (data.get("metadata") or {}).get("title") or ""
    title = re.split(r"\s+\|\s+", title)[0].strip()

    if not title:
        first_h1 = next((block for block in blocks if block.startswith("# ")), "")
        title = first_h1[2:].strip()

    return clean_inline(title) or "Consiva.ai"


def page_description(data: dict) -> str:
    description = (data.get("metadata") or {}).get("description") or ""
    description = " ".join(description.split())

    # The site appends the same tagline to every meta description.
    description = re.sub(r",\s*India's leading DPDP compliance platform.*$", "", description)

    return clean_inline(description)


def create_filename(url: str) -> str:
    return (
        url
        .replace("https://", "")
        .replace("http://", "")
        .replace("/", "_")
        .replace("?", "_")
        .replace("&", "_")
        .replace("=", "_")
        .replace(":", "_")
    )


def render_page(title: str, url: str, description: str, blocks: list[str]) -> str:
    front_matter = [
        "---",
        f"title: {title}",
        f"url: {url}",
    ]

    if description:
        front_matter.append(f"description: {description}")

    front_matter.append("---")

    return "\n".join(front_matter) + "\n\n" + "\n\n".join(blocks).strip() + "\n"


# ============================================================
# MAIN
# ============================================================

def load_page(source_file: Path) -> dict:
    with source_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    markdown = data.get("markdown")

    if not markdown:
        raise ValueError("No markdown content found")

    url = data.get("url") or (data.get("metadata") or {}).get("url") or source_file.stem
    blocks = to_blocks(markdown)

    return {
        "filename": f"{create_filename(url)}.txt",
        "url": url,
        "title": page_title(data, blocks),
        "description": page_description(data),
        "blocks": blocks,
    }


def remove_boilerplate(pages: list[dict]) -> int:
    """Keep paragraphs that repeat across many pages on the homepage only."""

    counts = Counter()

    for page in pages:
        for block in {normalize_block(block) for block in page["blocks"]}:
            if not HEADING.match(block):
                counts[block] += 1

    boilerplate = {block for block, count in counts.items() if count >= BOILERPLATE_MIN_PAGES}
    removed = 0

    for page in pages:
        if page["filename"] == HOMEPAGE_FILE:
            continue

        kept = [block for block in page["blocks"] if normalize_block(block) not in boilerplate]
        removed += len(page["blocks"]) - len(kept)
        page["blocks"] = kept

    return removed


def main() -> int:

    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        return 1

    files = sorted(SOURCE_DIR.glob("*.json"))

    if not files:
        print("ERROR: No JSON files found in data/raw/")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CLEAN + STRUCTURE")
    print("=" * 60)
    print(f"Source : {SOURCE_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Files  : {len(files)}")

    pages = []
    failed = 0

    for source_file in files:
        try:
            pages.append(load_page(source_file))
        except Exception as error:
            failed += 1
            print(f"[FAIL] {source_file.name}: {error}")

    removed = remove_boilerplate(pages)

    written = set()

    for index, page in enumerate(pages, start=1):
        output_file = OUTPUT_DIR / page["filename"]
        output_file.write_text(
            render_page(page["title"], page["url"], page["description"], page["blocks"]),
            encoding="utf-8",
        )
        written.add(output_file.name)
        print(f"[{index}/{len(pages)}] [OK] {page['filename']} ({len(page['blocks'])} blocks)")

    # Pages that disappeared from data/raw must disappear here too,
    # otherwise compare.py can never report them as DELETED.
    stale = [path for path in OUTPUT_DIR.glob("*.txt") if path.name not in written]

    if failed == 0:
        for path in stale:
            path.unlink()
            print(f"[REMOVED] {path.name} (no longer in data/raw)")
    elif stale:
        print(f"Skipped removing {len(stale)} stale file(s) because some pages failed.")

    print("=" * 60)
    print(f"Processed          : {len(pages)}")
    print(f"Failed             : {failed}")
    print(f"Boilerplate blocks : {removed} removed")
    print("=" * 60)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
