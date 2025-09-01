from dataclasses import dataclass
from typing import Optional
import httpx
from bs4 import BeautifulSoup
import trafilatura

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

@dataclass
class Page:
    url: str
    status_code: int
    html: Optional[str]
    text: Optional[str]
    title: Optional[str]

def _extract_title(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
    except Exception:
        pass
    return ""

def _trafilatura_text(html: str, url: str) -> Optional[str]:
    try:
        return trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
        )
    except Exception:
        return None

def _fallback_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script","style","noscript","header","footer","nav","svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

def fetch_and_extract(url: str, timeout: int = 25) -> Page:
    html, status = None, 0
    try:
        with httpx.Client(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=timeout) as client:
            r = client.get(url)
            status = r.status_code
            if r.status_code == 200:
                html = r.text
    except Exception:
        pass

    text, title = None, None
    if html:
        title = _extract_title(html) or None
        text = _trafilatura_text(html, url) or _fallback_visible_text(html)

    return Page(url=url, status_code=status, html=html, text=text, title=title)
