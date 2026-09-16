from pathlib import Path
import json
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "structured_final"
OUTPUT_DIR = BASE_DIR / "data" / "chunks"

OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"


# ============================================================
# CHUNK CONFIGURATION
# ============================================================

# Approximate limits.
# These are fallback character limits, not token limits.
CHUNK_SIZE = 2400
CHUNK_OVERLAP = 400


# ============================================================
# HEADING DETECTION
# ============================================================

HEADING_PATTERN = re.compile(
    r"^(#{1,6})[ \t]+(.+?)\s*$"
)


# ============================================================
# HEADING HIERARCHY
# ============================================================

def update_heading_context(
    headings: dict,
    level: int,
    title: str
):
    """
    Updates heading hierarchy.

    Example:

    H1
      H2
        H3

    New H2 automatically clears H3-H6.
    New H1 clears H2-H6.
    """

    headings[level] = title

    # Clear all lower-level headings
    for lower_level in range(level + 1, 7):
        headings[lower_level] = None


# ============================================================
# CREATE SECTION CONTEXT
# ============================================================

def get_heading_context(headings: dict):
    """
    Returns the complete active heading hierarchy.
    """

    context = []

    for level in range(1, 7):
        heading = headings[level]

        if heading:
            context.append(
                f"H{level}: {heading}"
            )

    return context


# ============================================================
# PARSE STRUCTURED DOCUMENT
# ============================================================

def parse_structured_document(
    text: str,
    source: str
):
    """
    Parses a structured TXT/Markdown document.

    Important:
    - Headings are NOT discarded.
    - H1-H6 are supported.
    - Heading hierarchy is preserved.
    - Body formatting is preserved as much as possible.
    """

    lines = text.splitlines()

    headings = {
        1: None,
        2: None,
        3: None,
        4: None,
        5: None,
        6: None,
    }

    sections = []

    current_content = []

    def save_section():
        if not current_content:
            return

        content = "\n".join(current_content).strip()

        if not content:
            return

        # Check whether the section contains actual body content
        # beyond Markdown headings.
        body_lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
            and not HEADING_PATTERN.match(line.strip())
        ]

        # Skip heading-only sections.
        if not body_lines:
            return

        active_headings = headings.copy()

        sections.append({
            "content": content,
            "headings": active_headings,
            "source": source
        })

    for line in lines:

        # Preserve the original line as much as possible.
        # Only remove trailing newline/whitespace.
        original_line = line.rstrip()

        match = HEADING_PATTERN.match(
            original_line
        )

        if match:

            # Save content belonging to previous heading
            save_section()

            current_content = []

            level = len(match.group(1))
            title = match.group(2).strip()

            # Update hierarchy BEFORE storing
            # the new heading.
            update_heading_context(
                headings,
                level,
                title
            )

            # IMPORTANT:
            # Keep the heading inside the section text.
            current_content.append(
                f"{'#' * level} {title}"
            )

            continue

        current_content.append(
            original_line
        )

    # Save final section
    save_section()

    return sections


# ============================================================
# BUILD CHUNK TEXT
# ============================================================

def build_chunk_text(
    content: str,
    headings: dict
):
    """
    Adds heading hierarchy directly into chunk text.

    This is important because vector databases may embed
    only page_content and ignore metadata.
    """

    heading_lines = []

    for level in range(1, 7):

        heading = headings[level]

        if heading:
            heading_lines.append(
                f"{'#' * level} {heading}"
            )

    heading_context = "\n".join(
        heading_lines
    )

    if heading_context:

        # Avoid duplicating headings if already present
        content_without_duplicate = content

        first_line = content.splitlines()[0] \
            if content.splitlines() else ""

        if first_line.strip() == heading_lines[-1].strip():

            return content

        return (
            heading_context
            + "\n\n"
            + content_without_duplicate
        )

    return content


# ============================================================
# SPLIT OVERSIZED SECTION
# ============================================================

def split_large_section(section):
    """
    Splits only sections that exceed CHUNK_SIZE.

    Heading context is injected into every resulting chunk.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
        ],
    )

    documents = splitter.create_documents(
        [section["content"]]
    )

    chunks = []

    for document in documents:

        chunk_text = build_chunk_text(
            document.page_content,
            section["headings"]
        )

        chunks.append(
            {
                "content": chunk_text,
                "source": section["source"],
                "H1": section["headings"][1],
                "H2": section["headings"][2],
                "H3": section["headings"][3],
                "H4": section["headings"][4],
                "H5": section["headings"][5],
                "H6": section["headings"][6],
            }
        )

    return chunks


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(file_path: Path):

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    sections = parse_structured_document(
        text=text,
        source=file_path.name
    )

    final_chunks = []

    for section in sections:

        content = section["content"]

        # ----------------------------------------------------
        # Small section
        # ----------------------------------------------------

        if len(content) <= CHUNK_SIZE:

            chunk_text = build_chunk_text(
                content,
                section["headings"]
            )

            final_chunks.append(
                {
                    "content": chunk_text,
                    "source": section["source"],
                    "H1": section["headings"][1],
                    "H2": section["headings"][2],
                    "H3": section["headings"][3],
                    "H4": section["headings"][4],
                    "H5": section["headings"][5],
                    "H6": section["headings"][6],
                }
            )

        # ----------------------------------------------------
        # Oversized section
        # ----------------------------------------------------

        else:

            final_chunks.extend(
                split_large_section(section)
            )

    return final_chunks


# ============================================================
# WRITE JSONL INCREMENTALLY
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    txt_files = sorted(
        INPUT_DIR.glob("*.txt")
    )

    if not txt_files:

        print(
            f"No .txt files found in:\n{INPUT_DIR}"
        )

        return

    total_chunks = 0
    processed_files = 0

    # --------------------------------------------------------
    # Write incrementally.
    # We don't keep the entire dataset in RAM.
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output:

        for file_path in txt_files:

            print(
                f"Processing: {file_path.name}"
            )

            chunks = process_file(
                file_path
            )

            for chunk in chunks:

                total_chunks += 1

                chunk["chunk_id"] = (
                    f"chunk_{total_chunks:06d}"
                )

                output.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False
                    )
                    + "\n"
                )

            processed_files += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 50)
    print("CHUNKING COMPLETED")
    print("=" * 50)

    print(
        f"Documents processed : {processed_files}"
    )

    print(
        f"Total chunks        : {total_chunks}"
    )

    print(
        f"Output file         : {OUTPUT_FILE}"
    )

    print("=" * 50)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()