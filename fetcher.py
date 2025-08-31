# fetcher.py
from __future__ import annotations
import os, asyncio
from typing import Optional

# Try Playwright; else return None (caller will fall back to requests)
PLAYWRIGHT_ENABLED = os.getenv("PY_DISABLE_PLAYWRIGHT", "0") != "1"

async def _playwright_html(url: str, timeout: int = 15) -> Optional[str]:
    try:
        from playwright.async_api import async_playwright
    except Exception:
        return None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context()
            page = await ctx.new_page()
            await page.goto(url, timeout=timeout * 1000)
            # wait for network to be idle-ish
            await page.wait_for_load_state("domcontentloaded", timeout=timeout * 1000)
            html = await page.content()
            await browser.close()
            return html
    except Exception:
        return None

def get_html(url: str, timeout: int = 15) -> Optional[str]:
    if not PLAYWRIGHT_ENABLED:
        return None
    try:
        return asyncio.run(_playwright_html(url, timeout))
    except RuntimeError:
        # already inside an event loop (e.g., Streamlit), create a new loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_playwright_html(url, timeout))
        finally:
            loop.close()
    except Exception:
        return None
