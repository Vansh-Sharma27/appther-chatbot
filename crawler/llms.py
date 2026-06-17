"""Single source of truth for fetching the site's llms.txt / llms-full.txt.

Both consumers -- the discovery cross-check (crawler.verify.fetch_llms_txt_urls)
and the overview-chunk builder (crawler.chunk.build_overview_doc) -- go through
this module, so:

  * there is exactly ONE llms.txt fetcher (previously verify.py and chunk.py each
    had their own, each building an ad-hoc httpx.Client);
  * the file is fetched with the shared http_client.create_client() (standard
    headers / timeouts / connection pool);
  * the pipeline fetches it only ONCE and derives both the URL list and the
    overview document from the same response (no double fetch).

This module is intentionally dependency-light (httpx + config + http_client only)
so importing it never pulls in the extraction/normalization stack.
"""

from __future__ import annotations

import logging
import re

import httpx

from crawler.config import LLMS_FULL_URL, LLMS_TXT_URL
from crawler.http_client import create_client

logger = logging.getLogger(__name__)

# https:// links referenced inside an llms.txt body (trailing punctuation trimmed
# by extract_llms_urls).
_HTTPS_URL_RE = re.compile(r"https://[^\s\)\]\>\"\']+")


def fetch_llms_txt(
    client: httpx.Client | None = None,
    prefer_full: bool = True,
) -> tuple[str, str]:
    """Fetch llms-full.txt / llms.txt and return ``(text, source_url)``.

    Tries llms-full.txt first (more context) then llms.txt, or the reverse when
    *prefer_full* is False. Returns ``("", "")`` if neither can be fetched.

    When *client* is None a shared create_client() is used and closed afterwards;
    callers (and tests) may inject their own client, which is left open.
    """
    order = [LLMS_FULL_URL, LLMS_TXT_URL] if prefer_full else [LLMS_TXT_URL, LLMS_FULL_URL]
    close_client = client is None
    if client is None:
        client = create_client()
    try:
        for url in order:
            try:
                resp = client.get(url)
            except httpx.HTTPError as exc:
                logger.debug("fetch_llms_txt: could not fetch %s: %s", url, exc)
                continue
            if resp.status_code == 200 and resp.text.strip():
                return resp.text, url
    finally:
        if close_client:
            client.close()
    return "", ""


def extract_llms_urls(text: str) -> list[str]:
    """Return the unique https:// URLs referenced in an llms.txt body.

    Trailing punctuation accidentally captured by the regex (e.g. a sentence-final
    period or closing bracket) is stripped, and order is preserved while de-duping.
    """
    found = _HTTPS_URL_RE.findall(text)
    cleaned = [u.rstrip(".,;:)]}") for u in found]
    return list(dict.fromkeys(cleaned))
