# llm.py — Gemini JSON-only helper

import os
import time
from typing import Optional
import google.generativeai as genai

def _configure():
    key = None
    try:
        import streamlit as st
        key = st.secrets.get("GOOGLE_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        pass
    key = key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set (Streamlit Cloud → Settings → Secrets).")
    genai.configure(api_key=key)

def gemini_json(prompt_text: str, model: str = "gemini-1.5-pro", temperature: float = 0.2,
                max_retries: int = 3, retry_base: float = 1.5) -> str:
    _configure()
    last_err: Optional[Exception] = None
    for i in range(max_retries):
        try:
            mdl = genai.GenerativeModel(model)
            resp = mdl.generate_content(
                prompt_text,
                generation_config={"temperature": temperature, "response_mime_type": "application/json"}
            )
            return resp.text or "{}"
        except Exception as e:
            last_err = e
            time.sleep(retry_base * (i + 1))
    raise RuntimeError(f"Gemini call failed after retries: {last_err}")
