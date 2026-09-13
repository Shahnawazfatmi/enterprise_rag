import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

BASELINE_DIR = BASE_DIR / "data" / "structured_final"
MANIFEST_DIR = BASE_DIR / "data" / "manifest"

MANIFEST_FILE = MANIFEST_DIR / "content_manifest.json"


# ---------------------------------------------------------
# Page ID
# ---------------------------------------------------------

def create_page_id(filename: str) -> str:
    """
    Convert a structured filename into a stable page ID.

    Examples:
        pricing.txt
        pricing_final.txt
        consiva.ai_pricing.txt

    All become:
        pricing
    """

    name = Path(filename).stem.lower()

    # Remove common suffixes
    name = re.sub(r"_final$", "", name)
    name = re.sub(r"_structured$", "", name)

    # Remove website prefix
    name = re.sub(r"^consiva\.ai[_-]?", "", name)

    # Normalize separators
    name = re.sub(r"[_\-\s]+", "-", name)

    # Special cases
    if name in {"homepage", "index", "home-page"}:
        name = "home"

    return name.strip("-")


# ---------------------------------------------------------
# Content hash
# ---------------------------------------------------------

def calculate_content_hash(file_path: Path) -> str:
    """
    Generate SHA256 hash from normalized page content.
    """

    content = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    # Normalize content before hashing
    content = content.replace("\r\n", "\n")
    content = content.replace("\r", "\n")

    # Remove trailing spaces
    lines = [
        line.strip()
        for line in content.splitlines()
    ]

    # Remove excessive blank lines
    normalized = "\n".join(
        line for line in lines if line
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# Build manifest
# ---------------------------------------------------------

def build_manifest():

    if not BASELINE_DIR.exists():
        raise FileNotFoundError(
            f"Baseline directory not found: {BASELINE_DIR}"
        )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pages": {}
    }

    txt_files = sorted(
        BASELINE_DIR.glob("*.txt")
    )

    print("=" * 60)
    print("BUILDING CONTENT MANIFEST")
    print("=" * 60)

    print(f"Baseline directory: {BASELINE_DIR}")
    print(f"Pages found: {len(txt_files)}")
    print()

    for file_path in txt_files:

        page_id = create_page_id(
            file_path.name
        )

        content_hash = calculate_content_hash(
            file_path
        )

        if page_id in manifest["pages"]:
            print(
                f"WARNING: Duplicate page_id detected: "
                f"{page_id}"
            )

            print(
                f"  Existing: "
                f"{manifest['pages'][page_id]['file_name']}"
            )

            print(
                f"  Current : "
                f"{file_path.name}"
            )

            continue

        manifest["pages"][page_id] = {
            "page_id": page_id,
            "file_name": file_path.name,
            "source_url": None,
            "content_hash": content_hash,
            "status": "active",
            "last_updated": None
        }

        print(
            f"REGISTERED: {page_id}"
            f" -> {file_path.name}"
        )

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("MANIFEST CREATED")
    print("=" * 60)

    print(f"File: {MANIFEST_FILE}")
    print(f"Pages registered: {len(manifest['pages'])}")


# ---------------------------------------------------------
# Load manifest
# ---------------------------------------------------------

def load_manifest():

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}"
        )

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ---------------------------------------------------------
# Save manifest
# ---------------------------------------------------------

def save_manifest(manifest):

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest["updated_at"] = (
        datetime.now(timezone.utc).isoformat()
    )

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    build_manifest()