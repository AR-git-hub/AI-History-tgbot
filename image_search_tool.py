"""
image_search_tool.py

Reverse image search via Yandex Images.

Pipeline:
  1. POST raw JPEG bytes to Yandex CDN endpoint
  2. GET search results page
  3. Parse JSON from data-state attribute of ImagesApp div
"""

import re
import json
import logging
import requests
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

YANDEX_UPLOAD_URL = (
    "https://yandex.ru/images-apphost/image-download"
    "?cbird=111&images_avatars_size=preview&images_avatars_namespace=images-cbir"
)
YANDEX_SEARCH_URL = "https://yandex.ru/images/search"

BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/147.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru,en-US;q=0.9,en;q=0.8",
    "Origin": "https://yandex.ru",
    "Referer": "https://yandex.ru/images/",
}


def _upload_to_yandex(image_path: str) -> str:
    """Upload image bytes, return CDN URL."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    headers = {
        **BASE_HEADERS,
        "Content-Type": "image/jpeg",
        "Accept": "*/*",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }

    resp = requests.post(YANDEX_UPLOAD_URL, headers=headers, data=image_bytes, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    logger.debug("Upload response: %s", data)

    url = data.get("url") or data.get("image_url") or data.get("cbir_url")
    if not url:
        raise RuntimeError(f"No URL in Yandex upload response: {data}")
    return url


def _parse_data_state(html: str) -> dict:
    """Extract JSON from data-state attribute of ImagesApp div."""
    match = re.search(
        r'<div[^>]+id="ImagesApp-[^"]*"[^>]+data-state="([^"]*)"',
        html
    )
    if not match:
        return {}
    raw = match.group(1)
    # unescape HTML entities
    raw = raw.replace("&quot;", '"').replace("&amp;", "&").replace("&#39;", "'")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error: %s", e)
        return {}


def _extract_results(state: dict) -> list[dict]:
    """Pull useful info from parsed data-state JSON."""
    results = []
    initial = state.get("initialState", {})

    # 1. Entity recognition (face/object detection)
    for resp in initial.get("cbirObjectResponses", {}).get("objectResponses", []):
        title = resp.get("title", "")
        description = resp.get("description", "")
        source_url = resp.get("sourceUrl", "")
        source_name = resp.get("sourceName", "")
        if title:
            results.append({
                "title": title,
                "snippet": description,
                "link": source_url,
                "domain": source_name,
                "type": "entity",
            })

    # 2. Tags / keywords Yandex detected
    tags = [t.get("text", "") for t in initial.get("cbirTags", {}).get("tags", []) if t.get("text")]
    if tags:
        results.append({
            "title": "Теги: " + ", ".join(tags),
            "snippet": "",
            "link": "",
            "domain": "",
            "type": "tags",
        })

    # 3. Sites with this image
    for site in initial.get("cbirSites", {}).get("sites", [])[:10]:
        title = site.get("title", "")
        url = site.get("url", "")
        domain = site.get("domain", "")
        description = site.get("description", "")
        if title and url:
            results.append({
                "title": title,
                "snippet": description,
                "link": url,
                "domain": domain,
                "type": "site",
            })

    return results


def _format_results(results: list[dict]) -> str:
    if not results:
        return "Yandex reverse image search returned no results."

    lines = ["=== Yandex reverse image search results ===\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        domain = r.get("domain", "")
        link = r.get("link", "")
        snippet = r.get("snippet", "")
        rtype = r.get("type", "")

        lines.append(f"[{i}] {title}")
        if domain:
            lines.append(f"    Source: {domain}")
        if link:
            lines.append(f"    URL:    {link}")
        if snippet:
            lines.append(f"    Desc:   {snippet[:200]}")
        lines.append("")

    return "\n".join(lines)


@tool
def search_similar_images(image_path: str) -> str:
    """
    Yandex reverse image search. Uploads image, parses recognition results
    (entity detection, tags, matching sites) from the response JSON.

    Args:
        image_path: path to local image file.

    Returns:
        Formatted text with recognized entity and matching sites.
    """
    if not Path(image_path).exists():
        return f"Error: file not found - {image_path}"

    try:
        cdn_url = _upload_to_yandex(image_path)
    except Exception as e:
        return f"Upload failed: {e}"

    logger.info("CDN URL: %s", cdn_url)

    search_headers = {
        **BASE_HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(
            YANDEX_SEARCH_URL,
            params={"url": cdn_url, "rpt": "imageview"},
            headers=search_headers,
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Search request failed: {e}"

    state = _parse_data_state(resp.text)
    if not state:
        logger.warning("Could not parse data-state JSON from Yandex response")
        return "Could not parse Yandex search results."

    results = _extract_results(state)
    return _format_results(results)


def run_image_search(image_path: str) -> str:
    """Direct call from agent.py (without LangChain)."""
    return search_similar_images.invoke(image_path)