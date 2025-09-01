# fetcher.py — 2025-09-01 (ddgs rename handled, minor polish)
from __future__ import annotations
import asyncio, json, re, os, logging
from typing import List, Dict, Optional, Any

UA = os.getenv(
    "BRIEFGEN_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/117.0 Safari/537.36",
)
HEADERS = {"User-Agent": UA}
PLAYWRIGHT_ENABLED = os.getenv("PY_DISABLE_PLAYWRIGHT", "0") != "1"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------- HTML fetching ----------------
def _sync_requests_html(url: str, timeout: int = 15) -> Optional[str]:
    try:
        import requests
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.ok and "text/html" in r.headers.get("content-type", ""):
            return r.text
    except Exception as exc:
        logger.debug("requests fetch failed: %s", exc, exc_info=False)
    return None

async def _playwright_html(url: str, timeout: int = 15) -> Optional[str]:
    try:
        from playwright.async_api import async_playwright
    except Exception:
        return None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, timeout=timeout * 1000)
            page = await browser.new_page()
            await page.goto(url, timeout=timeout * 1000)
            await page.wait_for_load_state("domcontentloaded", timeout=timeout * 1000)
            html = await page.content()
            await browser.close()
            return html
    except Exception as exc:
        logger.debug("playwright fetch failed: %s", exc, exc_info=False)
        return None

def fetch_html(url: str, timeout: int = 15) -> Optional[str]:
    html: Optional[str] = None
    if PLAYWRIGHT_ENABLED:
        try:
            html = asyncio.run(_playwright_html(url, timeout))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            html = loop.run_until_complete(_playwright_html(url, timeout))
            loop.close()
    return html or _sync_requests_html(url, timeout)

# ---------------- SERP scraper ----------------
def search_serp(query: str, max_results: int = 15) -> List[str]:
    # Google HTML parse (best effort)
    from urllib.parse import quote_plus
    q = quote_plus(query)
    url = f"https://www.google.com/search?q={q}&num={max_results}&hl=en"
    html = _sync_requests_html(url, timeout=10)
    urls: List[str] = []
    if html:
        for m in re.finditer(r'<a href="/url\?q=([^&]+)&', html):
            u = m.group(1)
            if u.startswith("http"):
                urls.append(u)
        if urls:
            return urls[:max_results]

    # Fallback to ddgs (new name of duckduckgo_search)
    try:
        try:
            from ddgs import DDGS  # new package name
        except Exception:
            from duckduckgo_search import DDGS  # backward compatibility
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                href = r.get("href") or r.get("link") or r.get("url")
                if href:
                    urls.append(href)
    except Exception as exc:
        logger.debug("DDGS fallback failed: %s", exc, exc_info=False)
    return urls[:max_results]

# ---------------- structured extractor ----------------
def extract_structured_product(html: str, url: str) -> Dict[str, Any]:
    try:
        import extruct, w3lib.html
    except ImportError:
        return {}
    base_url = w3lib.html.get_base_url(html, url)
    data = extruct.extract(html, base_url=base_url, syntaxes=["json-ld", "microdata"])
    product_nodes = []
    for syntax in ("json-ld", "microdata"):
        for node in data.get(syntax, []):
            if isinstance(node, dict) and node.get("@type") in ("Product", ["Product"]):
                product_nodes.append(node)
    attrs: Dict[str, str] = {}
    for node in product_nodes:
        name = node.get("name") or node.get("title")
        if name:
            attrs.setdefault("name", name)
        brand = node.get("brand")
        if isinstance(brand, dict):
            brand = brand.get("name")
        if brand:
            attrs.setdefault("brand", brand)
        for p in node.get("additionalProperty", []):
            if isinstance(p, dict):
                k = p.get("name") or p.get("propertyID")
                v = p.get("value") or p.get("valueReference")
                if k and v:
                    attrs.setdefault((k or "").lower(), str(v))
        for k in ("weight", "gtin13", "gtin14", "gtin", "mpn", "sku"):
            if node.get(k):
                attrs.setdefault(k, node[k])
    return {"attributes": attrs, "raw": product_nodes}

def download_pdf(url: str, timeout: int = 20) -> Optional[bytes]:
    try:
        import requests
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.ok and "application/pdf" in r.headers.get("content-type", ""):
            return r.content
    except Exception as exc:
        logger.debug("PDF download failed: %s", exc, exc_info=False)
    return None
