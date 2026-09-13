import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "new_structured"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADING DETECTION
# ============================================================

def is_heading(line: str) -> bool:

    line = line.strip()

    if not line:
        return False

    # Existing Markdown heading
    if re.match(r"^#{1,6}\s+", line):
        return True

    # Ignore very long lines
    if len(line) > 120:
        return False

    # Normal sentence endings
    if line.endswith((".", "?", "!", ";", ":")):
        return False

    # Numbered heading
    if re.match(r"^\d+[\.\)]\s+[A-Z]", line):
        return True

    # Roman numeral heading
    if re.match(
        r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)[\.\)]\s+[A-Z]",
        line
    ):
        return True

    # Common heading patterns
    heading_patterns = [
        r"^(what|why|how|who|when|where|which)\s+",
        r"^(about|overview|introduction|services|features|pricing|contact|faq)",
        r"^(benefits|advantages|disadvantages|requirements|eligibility)",
        r"^(key\s+features|key\s+benefits|important|summary|conclusion)",
        r"^(privacy|security|compliance|terms|support)",
    ]

    lower_line = line.lower()

    for pattern in heading_patterns:
        if re.match(pattern, lower_line, re.IGNORECASE):
            return True

    # Mostly title case
    words = line.split()

    if 2 <= len(words) <= 12:

        alpha_words = [
            w for w in words
            if re.search(r"[A-Za-z]", w)
        ]

        if alpha_words:

            capitalized = sum(
                1
                for word in alpha_words
                if word[0].isupper()
            )

            if capitalized / len(alpha_words) >= 0.7:
                return True

    # Uppercase heading
    if 2 <= len(words) <= 10 and line.upper() == line:

        if any(c.isalpha() for c in line):
            return True

    return False


# ============================================================
# HEADING LEVEL
# ============================================================

def heading_level(line: str) -> int:

    clean = line.strip()

    # Existing Markdown heading
    match = re.match(r"^(#{1,6})\s+", clean)

    if match:
        return len(match.group(1))

    # Numbered heading
    if re.match(r"^\d+[\.\)]\s+", clean):
        return 2

    # Roman numeral
    if re.match(
        r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)[\.\)]\s+",
        clean
    ):
        return 2

    return 2


# ============================================================
# MARKDOWN CLEANING
# ============================================================

def clean_markdown(text: str) -> str:

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Remove Firecrawl citation-like artifacts
    text = re.sub(r"\[([^\]]+)\]\(\s*javascript:[^)]+\)", r"\1", text)

    # Normalize bullet styles
    text = re.sub(r"^[*•]\s+", "- ", text, flags=re.MULTILINE)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# STRUCTURE DOCUMENT
# ============================================================

def format_document(text: str) -> str:

    text = clean_markdown(text)

    lines = text.split("\n")

    output = []

    first_meaningful_line = True
    previous_line = ""

    for line in lines:

        stripped = line.strip()

        # ----------------------------------------------------
        # Blank line
        # ----------------------------------------------------

        if not stripped:

            if output and output[-1] != "":
                output.append("")

            previous_line = ""
            continue

        # ----------------------------------------------------
        # Existing Markdown heading
        # ----------------------------------------------------

        markdown_heading = re.match(
            r"^(#{1,6})\s+(.+)",
            stripped
        )

        if markdown_heading:

            output.append(stripped)

            first_meaningful_line = False
            previous_line = stripped

            continue

        # ----------------------------------------------------
        # Bullet list
        # ----------------------------------------------------

        if re.match(r"^[-*•]\s+", stripped):

            output.append(
                re.sub(
                    r"^[*•]\s+",
                    "- ",
                    stripped
                )
            )

            first_meaningful_line = False
            previous_line = stripped

            continue

        # ----------------------------------------------------
        # Numbered list / heading
        # ----------------------------------------------------

        if re.match(r"^\d+[\.\)]\s+", stripped):

            if len(stripped) <= 100:

                level = heading_level(stripped)

                output.append(
                    f"{'#' * level} {stripped}"
                )

            else:

                output.append(stripped)

            first_meaningful_line = False
            previous_line = stripped

            continue

        # ----------------------------------------------------
        # First meaningful line = page title
        # ----------------------------------------------------

        if first_meaningful_line:

            if len(stripped) <= 150:

                output.append(
                    f"# {stripped}"
                )

            else:

                output.append(stripped)

            first_meaningful_line = False
            previous_line = stripped

            continue

        # ----------------------------------------------------
        # Detect heading
        # ----------------------------------------------------

        if is_heading(
            stripped
        ):

            level = heading_level(stripped)

            output.append(
                f"{'#' * level} {stripped}"
            )

        else:

            output.append(stripped)

        previous_line = stripped

    # ========================================================
    # REMOVE EXCESSIVE BLANK LINES
    # ========================================================

    cleaned = []

    blank_count = 0

    for line in output:

        if line == "":

            blank_count += 1

            if blank_count <= 1:
                cleaned.append("")

        else:

            blank_count = 0
            cleaned.append(line)

    return "\n".join(cleaned).strip() + "\n"


# ============================================================
# EXTRACT MARKDOWN FROM FIRECRAWL JSON
# ============================================================

def extract_markdown(data):

    markdown = data.get("markdown")

    if not markdown:
        return None

    return markdown


# ============================================================
# SAFE FILENAME
# ============================================================

def create_filename(url: str) -> str:

    filename = (
        url
        .replace("https://", "")
        .replace("http://", "")
        .replace("/", "_")
        .replace("?", "_")
        .replace("&", "_")
        .replace("=", "_")
        .replace(":", "_")
    )

    return filename


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(source_file: Path):

    with open(
        source_file,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    markdown = extract_markdown(data)

    if not markdown:

        raise ValueError(
            "No markdown content found"
        )

    structured = format_document(
        markdown
    )

    url = data.get(
        "url",
        data.get(
            "metadata",
            {}
        ).get("url", source_file.stem)
    )

    filename = create_filename(url)

    output_file = (
        OUTPUT_DIR /
        f"{filename}.txt"
    )

    output_file.write_text(
        structured,
        encoding="utf-8"
    )

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    if not SOURCE_DIR.exists():

        print(
            f"ERROR: Source directory not found:\n"
            f"{SOURCE_DIR}"
        )

        return

    files = sorted(
        SOURCE_DIR.glob("*.json")
    )

    if not files:

        print(
            "No JSON files found in data/raw/"
        )

        return

    processed = 0
    failed = 0

    print("=" * 60)
    print("NEW RAG CLEAN + STRUCTURE PIPELINE")
    print("=" * 60)

    print(f"Source : {SOURCE_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Files  : {len(files)}")
    print()

    for index, source_file in enumerate(
        files,
        start=1
    ):

        try:

            output_file = process_file(
                source_file
            )

            processed += 1

            print(
                f"[{index}/{len(files)}] "
                f"[OK] {source_file.name} "
                f"-> {output_file.name}"
            )

        except Exception as error:

            failed += 1

            print(
                f"[{index}/{len(files)}] "
                f"[FAIL] {source_file.name}: {error}"
            )

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)

    print(f"Processed : {processed}")
    print(f"Failed    : {failed}")
    print(f"Output    : {OUTPUT_DIR}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()