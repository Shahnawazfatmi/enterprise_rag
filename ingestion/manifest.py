from pathlib import Path
import hashlib
import json
import logging
from datetime import datetime, timezone


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BASELINE_DIR = BASE_DIR / "data" / "structured_final"
MANIFEST_FILE = BASE_DIR / "data" / "content_manifest.json"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# CONTENT HASHING
# ============================================================

def normalize_content(content: str) -> str:
    """
    Normalize text before hashing so insignificant whitespace
    changes do not appear as content changes.
    """

    if not content:
        return ""

    lines = []

    for line in content.splitlines():
        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines)


def calculate_content_hash(content: str) -> str:
    """
    Generate SHA-256 hash of normalized page content.
    """

    normalized = normalize_content(content)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# PAGE ID
# ============================================================

def get_page_id(file_path: Path) -> str:
    """
    Convert filename into a stable page ID.

    Example:
        consiva.ai_pricing.md
        ->
        consiva.ai_pricing
    """

    return file_path.stem


# ============================================================
# TIMESTAMP
# ============================================================

def utc_timestamp() -> str:
    """
    Return current UTC timestamp in ISO format.
    """

    return datetime.now(timezone.utc).isoformat()


# ============================================================
# LOAD MANIFEST
# ============================================================

def load_manifest() -> dict:
    """
    Load existing content manifest.

    If manifest does not exist or is invalid,
    return an empty manifest.
    """

    if not MANIFEST_FILE.exists():
        return {
            "version": 1,
            "updated_at": utc_timestamp(),
            "pages": {},
        }

    try:
        with MANIFEST_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Manifest root must be a JSON object.")

        data.setdefault("version", 1)
        data.setdefault("pages", {})
        data.setdefault("updated_at", utc_timestamp())

        return data

    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning(
            "Could not load manifest: %s",
            exc,
        )

        return {
            "version": 1,
            "updated_at": utc_timestamp(),
            "pages": {},
        }


# ============================================================
# SAVE MANIFEST
# ============================================================

def save_manifest(manifest: dict) -> None:
    """
    Save manifest atomically.

    The temporary file is written first and then replaced,
    preventing a partially-written manifest.
    """

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest["updated_at"] = utc_timestamp()

    temp_file = MANIFEST_FILE.with_suffix(".tmp")

    try:
        with temp_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                manifest,
                file,
                indent=2,
                ensure_ascii=False,
            )

            file.write("\n")

        temp_file.replace(MANIFEST_FILE)

    except Exception:
        if temp_file.exists():
            temp_file.unlink()

        raise


# ============================================================
# BUILD COMPLETE MANIFEST
# ============================================================

def build_manifest() -> dict:
    """
    Build a complete manifest from the current baseline.

    The baseline is:
        data/structured_final/
    """

    BASELINE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "version": 1,
        "updated_at": utc_timestamp(),
        "pages": {},
    }

    files = sorted(
        BASELINE_DIR.glob("*.md")
    )

    logger.info(
        "Building manifest from %d baseline pages...",
        len(files),
    )

    for file_path in files:

        try:
            content = file_path.read_text(
                encoding="utf-8"
            )

            page_id = get_page_id(file_path)
            content_hash = calculate_content_hash(content)

            manifest["pages"][page_id] = {
                "file": file_path.name,
                "hash": content_hash,
                "updated_at": utc_timestamp(),
            }

        except OSError as exc:

            logger.error(
                "Failed to read %s: %s",
                file_path,
                exc,
            )

    save_manifest(manifest)

    logger.info(
        "Manifest created successfully: %d pages",
        len(manifest["pages"]),
    )

    return manifest


# ============================================================
# UPDATE SINGLE PAGE
# ============================================================

def update_page(
    manifest: dict,
    file_path: Path,
) -> None:
    """
    Add or update a single page inside the manifest.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Page file does not exist: {file_path}"
        )

    content = file_path.read_text(
        encoding="utf-8"
    )

    page_id = get_page_id(file_path)
    content_hash = calculate_content_hash(content)

    manifest.setdefault("pages", {})

    manifest["pages"][page_id] = {
        "file": file_path.name,
        "hash": content_hash,
        "updated_at": utc_timestamp(),
    }


# ============================================================
# REMOVE SINGLE PAGE
# ============================================================

def remove_page(
    manifest: dict,
    page_id: str,
) -> None:
    """
    Remove a deleted page from the manifest.
    """

    pages = manifest.setdefault(
        "pages",
        {},
    )

    if page_id in pages:
        del pages[page_id]

        logger.info(
            "Removed page from manifest: %s",
            page_id,
        )


# ============================================================
# APPLY CHANGES
# ============================================================

def apply_changes(
    successful_pages: list[Path] | None = None,
    deleted_page_ids: list[str] | None = None,
) -> dict:
    """
    Apply successful ingestion changes to the manifest.

    Only pages that were successfully processed by the
    vector update stage should be passed in successful_pages.

    Deleted pages are removed using their page IDs.

    Parameters
    ----------
    successful_pages:
        List of successfully processed structured page files.

    deleted_page_ids:
        List of page IDs that were deleted from the sitemap.
    """

    successful_pages = successful_pages or []
    deleted_page_ids = deleted_page_ids or []

    manifest = load_manifest()

    updated_count = 0
    deleted_count = 0

    # --------------------------------------------------------
    # UPDATE SUCCESSFUL PAGES
    # --------------------------------------------------------

    for file_path in successful_pages:

        file_path = Path(file_path)

        try:

            update_page(
                manifest,
                file_path,
            )

            updated_count += 1

            logger.info(
                "Manifest updated: %s",
                file_path.name,
            )

        except Exception as exc:

            logger.error(
                "Failed to update manifest for %s: %s",
                file_path,
                exc,
            )

            raise

    # --------------------------------------------------------
    # REMOVE DELETED PAGES
    # --------------------------------------------------------

    for page_id in deleted_page_ids:

        if page_id in manifest.get("pages", {}):

            remove_page(
                manifest,
                page_id,
            )

            deleted_count += 1

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_manifest(manifest)

    logger.info(
        "Manifest changes applied | "
        "UPDATED=%d | DELETED=%d | TOTAL=%d",
        updated_count,
        deleted_count,
        len(manifest.get("pages", {})),
    )

    return manifest


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_manifest()