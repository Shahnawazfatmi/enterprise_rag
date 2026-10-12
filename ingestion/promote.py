"""
Promote data/new_structured to the baseline (data/baseline).

Runs only after the vector update succeeded. Files are replaced one by
one (copy to a temp file, then os.replace), never by renaming folders:
on Windows a folder rename fails with "Access is denied" whenever another
program (antivirus, search indexer, Explorer) has the folder open.

If promotion stops halfway, the next run sees the not-yet-promoted pages
as CHANGED and processes them again, so no update is lost.
"""

from pathlib import Path
import filecmp
import logging
import os
import shutil
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import BASELINE_DIR, NEW_STRUCTURED_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


def validate_source() -> list[Path]:
    """Validate that the new structured dataset is safe to promote."""

    if not NEW_STRUCTURED_DIR.exists():
        raise FileNotFoundError(f"New structured directory not found: {NEW_STRUCTURED_DIR}")

    files = sorted(NEW_STRUCTURED_DIR.glob("*.txt"))

    if not files:
        raise ValueError("No .txt files found in new_structured. Baseline promotion aborted.")

    logger.info(f"Validated source dataset: {len(files)} pages")
    return files


def promote_baseline() -> None:

    files = validate_source()
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    names = {source.name for source in files}

    for source in files:
        destination = BASELINE_DIR / source.name
        temp_file = destination.with_name(destination.name + ".tmp")
        shutil.copy2(source, temp_file)
        os.replace(temp_file, destination)

    removed = 0
    for old_file in BASELINE_DIR.glob("*.txt"):
        if old_file.name not in names:
            old_file.unlink()
            removed += 1

    mismatched = [
        source.name
        for source in files
        if not filecmp.cmp(source, BASELINE_DIR / source.name, shallow=False)
    ]

    if mismatched or len(list(BASELINE_DIR.glob("*.txt"))) != len(files):
        raise RuntimeError(f"Baseline verification failed: {mismatched[:5]}")

    logger.info(f"Baseline promoted successfully: {len(files)} pages ({removed} removed)")


if __name__ == "__main__":
    try:
        promote_baseline()
        logger.info("Baseline promotion completed successfully.")

    except KeyboardInterrupt:
        logger.error("Operation cancelled.")
        sys.exit(130)

    except Exception as error:
        logger.error(f"Baseline promotion failed: {error}")
        sys.exit(1)
