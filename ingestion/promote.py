from pathlib import Path
import logging
import shutil
import sys
import tempfile

BASE_DIR = Path(__file__).resolve().parent.parent

NEW_STRUCTURED_DIR = BASE_DIR / "data" / "new_structured"
BASELINE_DIR = BASE_DIR / "data" / "structured_final"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


def validate_source() -> None:
    """Validate that the new structured dataset is safe to promote."""

    if not NEW_STRUCTURED_DIR.exists():
        raise FileNotFoundError(
            f"New structured directory not found: {NEW_STRUCTURED_DIR}"
        )

    files = list(NEW_STRUCTURED_DIR.glob("*.txt"))

    if not files:
        raise ValueError(
            "No .txt files found in new_structured. "
            "Baseline promotion aborted."
        )

    logger.info(f"Validated source dataset: {len(files)} pages")


def promote_baseline() -> None:
    """
    Replace structured_final with new_structured safely.

    This function should only be called AFTER the vector DB update
    has completed successfully.
    """

    validate_source()

    BASELINE_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary directory beside the baseline.
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="structured_final_",
            dir=BASELINE_DIR.parent
        )
    )

    try:
        # Copy the complete new dataset into the temporary directory.
        for source_file in NEW_STRUCTURED_DIR.glob("*.txt"):
            destination = temp_dir / source_file.name
            shutil.copy2(source_file, destination)

        copied_files = list(temp_dir.glob("*.txt"))

        if len(copied_files) != len(list(NEW_STRUCTURED_DIR.glob("*.txt"))):
            raise RuntimeError(
                "Baseline copy verification failed."
            )

        # Backup existing baseline before replacement.
        backup_dir = None

        if BASELINE_DIR.exists():
            backup_dir = BASELINE_DIR.parent / "structured_final_backup"

            if backup_dir.exists():
                shutil.rmtree(backup_dir)

            BASELINE_DIR.rename(backup_dir)

        try:
            # Atomic directory replacement.
            temp_dir.rename(BASELINE_DIR)

        except Exception:
            # Restore previous baseline if replacement fails.
            if backup_dir and backup_dir.exists() and not BASELINE_DIR.exists():
                backup_dir.rename(BASELINE_DIR)

            raise

        # Remove backup only after successful promotion.
        if backup_dir and backup_dir.exists():
            shutil.rmtree(backup_dir)

        logger.info(
            f"Baseline promoted successfully: {len(copied_files)} pages"
        )

    except Exception:
        # Clean temporary directory if anything fails.
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        raise


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