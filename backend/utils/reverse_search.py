"""
Reverse Image Search
─────────────────────
Mock implementation of reverse image search.
Replace with real API integration (Google Vision, TinEye, etc.)

Real integrations to add:
  - Google Cloud Vision API: vision.googleapis.com
  - TinEye API: api.tineye.com
  - Bing Visual Search: api.cognitive.microsoft.com
"""

import random
import hashlib
from typing import Optional


MOCK_SOURCES = [
    "No prior web presence found.",
    "First seen: Reuters article from March 2021.",
    "Earliest match: AP News, November 2020.",
    "Similar image found in 2019 BBC article — different context.",
    "Exact match: Getty Images stock photo #1184729.",
    "Found on 23 domains. Earliest: CNN, January 2022.",
    "No matches found in reverse search index.",
    "Viral post: First appeared on Twitter in October 2021.",
    "Stock photo: Shutterstock #888234. Licensed image.",
    "Multiple low-credibility site matches. No authoritative source.",
]


def mock_reverse_search(image_url: str) -> str:
    """
    Deterministic mock based on URL.
    Replace this function body with real API call.
    """
    h = int(hashlib.md5(image_url.encode()).hexdigest(), 16)
    return MOCK_SOURCES[h % len(MOCK_SOURCES)]


async def google_reverse_search(image_url: str, api_key: str) -> str:
    """
    Real Google Vision API reverse search.
    Requires GOOGLE_CLOUD_API_KEY environment variable.
    """
    import httpx

    endpoint = "https://vision.googleapis.com/v1/images:annotate"
    payload = {
        "requests": [{
            "image": {"source": {"imageUri": image_url}},
            "features": [{"type": "WEB_DETECTION", "maxResults": 5}]
        }]
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            endpoint,
            json=payload,
            params={"key": api_key},
            timeout=10.0,
        )
        data = resp.json()

    web = data.get("responses", [{}])[0].get("webDetection", {})
    pages = web.get("pagesWithMatchingImages", [])
    full_matches = web.get("fullMatchingImages", [])

    if full_matches:
        best = pages[0].get("url", "") if pages else ""
        return f"Exact match found on {_domain(best)}" if best else "Exact match found"
    elif pages:
        return f"Similar image found on {len(pages)} sites including {_domain(pages[0]['url'])}"
    else:
        return "No matches found in reverse search."


def _domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc
    except Exception:
        return url[:30]
