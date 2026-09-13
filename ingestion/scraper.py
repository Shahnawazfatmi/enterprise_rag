
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from firecrawl import FirecrawlApp

from sitemap import get_sitemap_urls


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"

MAX_RETRIES = 5

# Wait after Firecrawl rate limit
RATE_LIMIT_WAIT = 65

# Initial wait for temporary/network errors
RETRY_WAIT = 10

# Maximum wait between retries
MAX_RETRY_WAIT = 120


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# INITIALIZE FIRECRAWL
# ============================================================

def load_firecrawl():

    load_dotenv()

    api_key = os.getenv("FIRECRAWL_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "FIRECRAWL_API_KEY not found in .env"
        )

    logger.info("Initializing Firecrawl...")

    return FirecrawlApp(
        api_key=api_key
    )


# ============================================================
# CREATE SAFE FILENAME
# ============================================================

def get_output_file(url: str):

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

    if not filename:
        filename = "homepage"

    return RAW_DATA_DIR / f"{filename}.json"


# ============================================================
# CHECK RATE LIMIT ERROR
# ============================================================

def is_rate_limit_error(error: Exception) -> bool:

    error_message = str(error).lower()

    return (
        "rate limit" in error_message
        or "rate_limit" in error_message
        or "too many requests" in error_message
        or "429" in error_message
    )


# ============================================================
# CHECK TEMPORARY ERROR
# ============================================================

def is_temporary_error(error: Exception) -> bool:

    error_message = str(error).lower()

    temporary_errors = (
        "timeout",
        "timed out",
        "connection",
        "connect",
        "network",
        "temporarily unavailable",
        "service unavailable",
        "503",
        "502",
        "504",
        "server error",
    )

    return any(
        message in error_message
        for message in temporary_errors
    )


# ============================================================
# SCRAPE SINGLE PAGE
# ============================================================

def scrape_page(
    firecrawl,
    url: str,
    max_retries: int = MAX_RETRIES
):

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            logger.info(
                f"Scraping: {url}"
            )

            result = firecrawl.scrape(
                url,
                formats=["markdown"]
            )

            return result

        except Exception as error:

            # ==================================================
            # RATE LIMIT
            # ==================================================

            if is_rate_limit_error(error):

                if attempt >= max_retries:

                    raise RuntimeError(
                        f"Rate limit persisted after "
                        f"{max_retries} attempts."
                    ) from error

                logger.warning(
                    f"Rate limit reached for {url}"
                )

                logger.warning(
                    f"Waiting {RATE_LIMIT_WAIT} seconds "
                    f"before retry "
                    f"{attempt}/{max_retries}..."
                )

                time.sleep(
                    RATE_LIMIT_WAIT
                )

                continue

            # ==================================================
            # TEMPORARY / NETWORK ERROR
            # ==================================================

            if is_temporary_error(error):

                if attempt >= max_retries:

                    raise RuntimeError(
                        f"Temporary error persisted after "
                        f"{max_retries} attempts."
                    ) from error

                wait_time = min(
                    RETRY_WAIT * (2 ** (attempt - 1)),
                    MAX_RETRY_WAIT
                )

                logger.warning(
                    f"Temporary error while scraping {url}"
                )

                logger.warning(
                    f"Reason: {error}"
                )

                logger.warning(
                    f"Waiting {wait_time} seconds "
                    f"before retry "
                    f"{attempt}/{max_retries}..."
                )

                time.sleep(
                    wait_time
                )

                continue

            # ==================================================
            # UNKNOWN ERROR
            # ==================================================

            raise


# ============================================================
# CONVERT FIRECRAWL RESPONSE
# ============================================================

def convert_result_to_dict(result):

    if hasattr(result, "model_dump"):

        data = result.model_dump()

    elif hasattr(result, "dict"):

        data = result.dict()

    elif isinstance(result, dict):

        data = result

    else:

        raise TypeError(
            "Unsupported Firecrawl response type."
        )

    if not isinstance(data, dict):

        raise TypeError(
            "Firecrawl response could not be converted "
            "to a dictionary."
        )

    return data


# ============================================================
# SAVE PAGE SAFELY
# ============================================================

def save_page(
    url: str,
    result
):

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = get_output_file(url)

    data = convert_result_to_dict(result)

    # Always preserve the source URL
    data["url"] = url

    # ========================================================
    # WRITE TO TEMPORARY FILE FIRST
    # ========================================================
    # This prevents a failed write from corrupting the
    # previously successful raw file.

    temp_file = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            prefix=f"{output_file.stem}_",
            dir=RAW_DATA_DIR,
            delete=False
        ) as file:

            temp_file = Path(file.name)

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
                default=str
            )

            file.flush()

        # Replace old file only after successful write
        temp_file.replace(output_file)

        logger.info(
            f"Saved: {output_file.name}"
        )

    except Exception:

        if temp_file and temp_file.exists():

            try:
                temp_file.unlink()

            except OSError:
                pass

        raise


# ============================================================
# SCRAPE ALL SITEMAP URLS
# ============================================================

def scrape_all_pages():

    firecrawl = load_firecrawl()

    urls = get_sitemap_urls()

    if not urls:

        raise RuntimeError(
            "No URLs found in sitemap."
        )

    logger.info(
        f"Starting scraping for {len(urls)} URLs"
    )

    successful = 0
    failed = 0

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # ALWAYS SCRAPE EVERY URL
    # ========================================================
    # IMPORTANT:
    # There is intentionally NO:
    #
    #     if output_file.exists():
    #         continue
    #
    # Existing files are overwritten with fresh Firecrawl
    # results on every pipeline execution.

    for index, url in enumerate(
        urls,
        start=1
    ):

        logger.info(
            f"[{index}/{len(urls)}] {url}"
        )

        try:

            result = scrape_page(
                firecrawl,
                url
            )

            save_page(
                url,
                result
            )

            successful += 1

        except Exception as error:

            failed += 1

            logger.error(
                f"FAILED: {url}"
            )

            logger.error(
                f"Reason: {error}"
            )

            # Continue with remaining pages.
            # Previous successful raw data remains untouched.

            continue

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    logger.info("=" * 60)

    logger.info(
        f"Scraping completed | "
        f"Scraped: {successful} | "
        f"Failed: {failed}"
    )

    logger.info("=" * 60)

    # ========================================================
    # IMPORTANT FOR PIPELINE
    # ========================================================
    # If any page failed, return failure so pipeline.py
    # does NOT continue to cleaning/vector update/baseline
    # promotion with incomplete scraped data.

    if failed > 0:

        raise RuntimeError(
            f"Scraping completed with "
            f"{failed} failed page(s). "
            f"Pipeline execution should stop."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        scrape_all_pages()

        sys.exit(0)

    except KeyboardInterrupt:

        logger.error(
            "Scraping cancelled by user."
        )

        sys.exit(130)

    except Exception as error:

        logger.error(
            f"Scraping failed: {error}"
        )

        sys.exit(1)

