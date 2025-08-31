# fetcher.py
from __future__ import annotations
from typing import Optional
import os, hashlib, time

try:
    import requests
except Exception:
    requests = None

CACHE_DIR = os.path.join(os.path.abspath(os.getenv("TMPDIR", "/tmp")), "briefgen_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return os.path.join(CACHE_DIR, f"{h}.html")

def get_html(url: str, *, ttl_seconds: int = 86400) -> Optional[str]:
    """Fetch HTML via requests; optionally use Playwright if USE_PLAYWRIGHT=1."""
    path = _cache_path(url)
    now = time.time()
    if os.path.exists(path) and (now - os.path.getmtime(path)) < ttl_seconds:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            pass

    html = None
    # 1) requests first
    if requests:
        try:
            r = requests.get(url, headers={"User-Agent": "briefgenv2/1.0"}, timeout=15)
            if "text/html" in r.headers.get("Content-Type",""):
                html = r.text
        except Exception:
            html = None

    # 2) optional Playwright render
    if (not html or len(html) < 5000) and os.getenv("USE_PLAYWRIGHT","0") == "1":
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.firefox.launch(headless=True)
                page = browser.new_page()
                page.set_default_timeout(15000)
                page.goto(url, wait_until="load")
                page.wait_for_load_state("networkidle")
                html = page.content()
                browser.close()
        except Exception:
            pass

    if html:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
        except Exception:
            pass
    return html

def link_density(html_text: str) -> float:
    """Crude link density over html text."""
    if not html_text:
        return 0.0
    a_tags = html_text.lower().count("<a ")
    words = max(1, len(html_text.split()))
    return min(1.0, (a_tags * 10) / words)
