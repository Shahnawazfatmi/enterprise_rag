"""
Production-ready content comparison for RAG ingestion pipeline.

Compares:

    data/structured_final/  -> last accepted baseline
    data/new_structured/    -> latest structured website content

Detects:

    NEW
    CHANGED
    UNCHANGED
    DELETED

Output:

    data/comparison/changes.json

Important:
- Does NOT modify the baseline.
- Does NOT modify the vector database.
- Does NOT use structured_report.json.
- Uses actual TXT file content for comparison.
- Produces deterministic hashes.
- Detects duplicate page identities.
- Writes changes.json atomically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BASELINE_DIR = BASE_DIR / "data" / "structured_final"
NEW_STRUCTURED_DIR = BASE_DIR / "data" / "new_structured"
COMPARISON_DIR = BASE_DIR / "data" / "comparison"

CHANGES_FILE = COMPARISON_DIR / "changes.json"

SUPPORTED_EXTENSION = ".txt"

SCHEMA_VERSION = "1.0"

# Hash algorithm used for content fingerprints.
HASH_ALGORITHM = "sha256"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("compare")


# ============================================================
# DATA MODEL
# ============================================================

@dataclass(frozen=True)
class PageInfo:
    """
    Information about one structured page.
    """

    page_id: str
    file_name: str
    file_path: str
    content_hash: str
    content_size: int


# ============================================================
# FILENAME / PAGE ID
# ============================================================

def normalize_page_id(filename: str) -> str:
    """
    Converts a TXT filename into a stable page identifier.

    Examples:

        pricing.txt
        pricing_final.txt

    Both become:

        pricing

    Only the trailing '_final' is removed.

    We intentionally do NOT perform aggressive alias matching
    because the current 39-page dataset is the source of truth.
    """

    name = Path(filename).stem.strip().lower()

    if name.endswith("_final"):
        name = name[:-6]

    # Normalize whitespace.
    name = re.sub(r"\s+", "_", name)

    # Prevent accidental duplicate separators.
    name = re.sub(r"_+", "_", name)

    return name.strip("_")


# ============================================================
# CONTENT NORMALIZATION
# ============================================================

def normalize_content(content: str) -> str:
    """
    Normalizes text before hashing.

    This prevents harmless formatting differences from being
    treated as content changes.

    Examples ignored:

        Windows vs Linux line endings
        trailing spaces
        multiple spaces
        blank lines
    """

    # Normalize line endings.
    content = content.replace("\r\n", "\n")
    content = content.replace("\r", "\n")

    normalized_lines: List[str] = []

    for line in content.split("\n"):

        # Remove leading/trailing whitespace.
        line = line.strip()

        # Collapse repeated spaces/tabs.
        line = re.sub(r"[ \t]+", " ", line)

        if line:
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


# ============================================================
# HASHING
# ============================================================

def calculate_content_hash(file_path: Path) -> tuple[str, int]:
    """
    Calculates SHA-256 hash from normalized file content.

    Returns:

        (hash, normalized_content_size)
    """

    try:
        content = file_path.read_text(
            encoding="utf-8",
            errors="strict",
        )
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"File is not valid UTF-8: {file_path}"
        ) from exc

    normalized = normalize_content(content)

    content_bytes = normalized.encode("utf-8")

    digest = hashlib.sha256(content_bytes).hexdigest()

    return digest, len(content_bytes)


# ============================================================
# DIRECTORY VALIDATION
# ============================================================

def validate_directory(directory: Path, name: str) -> None:
    """
    Validates that the required directory exists.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"{name} directory does not exist:\n{directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"{name} path is not a directory:\n{directory}"
        )


# ============================================================
# LOAD DIRECTORY
# ============================================================

def load_pages(directory: Path, label: str) -> Dict[str, PageInfo]:
    """
    Loads all TXT pages from a directory.

    Returns:

        {
            page_id: PageInfo(...)
        }

    Duplicate page IDs are treated as an error instead of
    silently overwriting data.
    """

    validate_directory(directory, label)

    pages: Dict[str, PageInfo] = {}

    files = sorted(
        (
            file
            for file in directory.iterdir()
            if file.is_file()
            and file.suffix.lower() == SUPPORTED_EXTENSION
        ),
        key=lambda path: path.name.lower(),
    )

    logger.info(
        "%s directory: %s",
        label,
        directory,
    )

    logger.info(
        "%s TXT files found: %d",
        label,
        len(files),
    )

    for file_path in files:

        page_id = normalize_page_id(file_path.name)

        if not page_id:
            raise ValueError(
                f"Could not create page ID from file:\n{file_path}"
            )

        # Detect duplicate page IDs.
        if page_id in pages:

            existing = pages[page_id]

            raise ValueError(
                "\nDuplicate page ID detected!\n"
                f"Page ID: {page_id}\n"
                f"File 1: {existing.file_name}\n"
                f"File 2: {file_path.name}\n\n"
                "Rename one of these files before running comparison."
            )

        content_hash, content_size = calculate_content_hash(
            file_path
        )

        pages[page_id] = PageInfo(
            page_id=page_id,
            file_name=file_path.name,
            file_path=str(file_path),
            content_hash=content_hash,
            content_size=content_size,
        )

    return pages


# ============================================================
# COMPARE
# ============================================================

def compare_pages(
    baseline_pages: Dict[str, PageInfo],
    new_pages: Dict[str, PageInfo],
) -> dict:
    """
    Compares baseline against latest content.
    """

    baseline_ids = set(baseline_pages.keys())
    new_ids = set(new_pages.keys())

    # --------------------------------------------------------
    # NEW
    # --------------------------------------------------------

    new_page_ids = sorted(
        new_ids - baseline_ids
    )

    # --------------------------------------------------------
    # DELETED
    # --------------------------------------------------------

    deleted_page_ids = sorted(
        baseline_ids - new_ids
    )

    # --------------------------------------------------------
    # CHANGED / UNCHANGED
    # --------------------------------------------------------

    changed_page_ids: List[str] = []
    unchanged_page_ids: List[str] = []

    common_page_ids = sorted(
        baseline_ids & new_ids
    )

    for page_id in common_page_ids:

        baseline_hash = baseline_pages[
            page_id
        ].content_hash

        new_hash = new_pages[
            page_id
        ].content_hash

        if baseline_hash == new_hash:
            unchanged_page_ids.append(page_id)
        else:
            changed_page_ids.append(page_id)

    # --------------------------------------------------------
    # PAGE DETAILS
    # --------------------------------------------------------

    all_page_ids = sorted(
        baseline_ids | new_ids
    )

    page_details = {}

    for page_id in all_page_ids:

        baseline = baseline_pages.get(page_id)
        new = new_pages.get(page_id)

        page_details[page_id] = {
            "status": get_page_status(
                page_id=page_id,
                new_page_ids=set(new_page_ids),
                changed_page_ids=set(changed_page_ids),
                unchanged_page_ids=set(unchanged_page_ids),
                deleted_page_ids=set(deleted_page_ids),
            ),
            "baseline": (
                {
                    "file_name": baseline.file_name,
                    "content_hash": baseline.content_hash,
                    "content_size": baseline.content_size,
                }
                if baseline
                else None
            ),
            "new": (
                {
                    "file_name": new.file_name,
                    "content_hash": new.content_hash,
                    "content_size": new.content_size,
                }
                if new
                else None
            ),
        }

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    comparison = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "hash_algorithm": HASH_ALGORITHM,

        "directories": {
            "baseline": str(BASELINE_DIR),
            "new": str(NEW_STRUCTURED_DIR),
        },

        "summary": {
            "baseline_pages": len(baseline_pages),
            "new_pages": len(new_pages),

            "new": len(new_page_ids),
            "changed": len(changed_page_ids),
            "unchanged": len(unchanged_page_ids),
            "deleted": len(deleted_page_ids),

            "total_actions_required": (
                len(new_page_ids)
                + len(changed_page_ids)
                + len(deleted_page_ids)
            ),
        },

        "pages": {
            "new": new_page_ids,
            "changed": changed_page_ids,
            "unchanged": unchanged_page_ids,
            "deleted": deleted_page_ids,
        },

        "details": page_details,
    }

    return comparison


# ============================================================
# STATUS
# ============================================================

def get_page_status(
    page_id: str,
    new_page_ids: set[str],
    changed_page_ids: set[str],
    unchanged_page_ids: set[str],
    deleted_page_ids: set[str],
) -> str:

    if page_id in new_page_ids:
        return "NEW"

    if page_id in changed_page_ids:
        return "CHANGED"

    if page_id in unchanged_page_ids:
        return "UNCHANGED"

    if page_id in deleted_page_ids:
        return "DELETED"

    raise RuntimeError(
        f"Unable to determine status for page: {page_id}"
    )


# ============================================================
# ATOMIC JSON WRITE
# ============================================================

def save_json_atomic(data: dict, output_file: Path) -> None:
    """
    Writes JSON atomically.

    The file is first written to a temporary file and then
    replaced.

    This prevents a partially-written changes.json if the
    process is interrupted.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_path = tempfile.mkstemp(
        prefix=".changes_",
        suffix=".tmp",
        dir=output_file.parent,
        text=True,
    )

    try:

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

            file.write("\n")

            file.flush()

            os.fsync(file.fileno())

        os.replace(
            temp_path,
            output_file,
        )

    except Exception:

        try:
            os.unlink(temp_path)
        except OSError:
            pass

        raise


# ============================================================
# DISPLAY RESULT
# ============================================================

def print_result(comparison: dict) -> None:

    summary = comparison["summary"]
    pages = comparison["pages"]

    print()
    print("=" * 70)
    print("RAG CONTENT COMPARISON")
    print("=" * 70)

    print(
        f"Baseline pages : {summary['baseline_pages']}"
    )

    print(
        f"New pages      : {summary['new_pages']}"
    )

    print()

    print(
        f"NEW            : {summary['new']}"
    )

    print(
        f"CHANGED        : {summary['changed']}"
    )

    print(
        f"UNCHANGED      : {summary['unchanged']}"
    )

    print(
        f"DELETED        : {summary['deleted']}"
    )

    print(
        f"DB actions     : {summary['total_actions_required']}"
    )

    print()

    if pages["new"]:
        print("NEW PAGES")
        print("-" * 40)

        for page in pages["new"]:
            print(f"  + {page}")

        print()

    if pages["changed"]:
        print("CHANGED PAGES")
        print("-" * 40)

        for page in pages["changed"]:
            print(f"  ~ {page}")

        print()

    if pages["deleted"]:
        print("DELETED PAGES")
        print("-" * 40)

        for page in pages["deleted"]:
            print(f"  - {page}")

        print()

    if pages["unchanged"]:
        print(
            f"UNCHANGED PAGES: {len(pages['unchanged'])}"
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    try:

        logger.info("Starting content comparison...")

        # ----------------------------------------------------
        # Load baseline
        # ----------------------------------------------------

        baseline_pages = load_pages(
            BASELINE_DIR,
            "BASELINE",
        )

        # ----------------------------------------------------
        # Load latest
        # ----------------------------------------------------

        new_pages = load_pages(
            NEW_STRUCTURED_DIR,
            "NEW",
        )

        # ----------------------------------------------------
        # Compare
        # ----------------------------------------------------

        comparison = compare_pages(
            baseline_pages=baseline_pages,
            new_pages=new_pages,
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        save_json_atomic(
            comparison,
            CHANGES_FILE,
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print_result(comparison)

        print()
        print(
            f"Comparison saved to:\n{CHANGES_FILE}"
        )
        print()

        logger.info(
            "Content comparison completed successfully."
        )

        return 0

    except KeyboardInterrupt:

        logger.warning(
            "Comparison interrupted by user."
        )

        return 130

    except Exception as exc:

        logger.exception(
            "Comparison failed: %s",
            exc,
        )

        return 1


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    sys.exit(main())