import requests
import xml.etree.ElementTree as ET
import logging


# ============================================================
# CONFIGURATION
# ============================================================

SITEMAP_URL = "https://consiva.ai/sitemap.xml"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# EXTRACT URLS FROM SITEMAP
# ============================================================

def get_sitemap_urls():

    logger.info(
        f"Reading sitemap: {SITEMAP_URL}"
    )

    response = requests.get(
        SITEMAP_URL,
        timeout=30
    )

    response.raise_for_status()

    root = ET.fromstring(
        response.content
    )

    urls = []

    # Handles standard sitemap namespace
    namespace = {
        "sm": "http://www.sitemaps.org/schemas/sitemap/0.9"
    }

    for url in root.findall("sm:url", namespace):

        loc = url.find(
            "sm:loc",
            namespace
        )

        if loc is not None and loc.text:

            urls.append(
                loc.text.strip()
            )

    logger.info(
        f"Found {len(urls)} URLs"
    )

    return urls


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    urls = get_sitemap_urls()

    print("\n" + "=" * 60)
    print("CONSIVA SITEMAP URLS")
    print("=" * 60)

    for index, url in enumerate(urls, 1):

        print(
            f"{index}. {url}"
        )

    print(
        f"\nTotal URLs: {len(urls)}"
    )