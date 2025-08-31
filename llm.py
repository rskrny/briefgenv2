# llm.py
# Thin JSON-only wrappers for Gemini and OpenAI with robust key loading.

from __future__ import annotations
import os, json, re
from typing import Any, Dict, Optional


def _get_secret(key: str) -> Optional[str]:
    """Try env first, then Streamlit secrets if available."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st  # optional
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return None


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON parse for occasionally noisy model outputs."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"raw": text}


def openai_json(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: str = "",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Call OpenAI Chat Completions and force a JSON object response."""
    from openai import OpenAI

    api_key = _get_secret("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)  # uses env var internally if None

    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    return _extract_json(text)


def gemini_json(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: str = "",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Call Gemini and force a JSON object response."""
    import google.generativeai as genai

    api_key = _get_secret("GOOGLE_API_KEY")
    if not api_key:
        # Helpful error for Streamlit Cloud users
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it via Streamlit secrets or environment."
        )
    genai.configure(api_key=api_key)

    model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    full_prompt = (system + "\n\n" + prompt).strip() if system else prompt

    g = genai.GenerativeModel(model)
    resp = g.generate_content(
        full_prompt,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": temperature,
        },
    )

    # google-generativeai returns .text for the aggregated string
    text = getattr(resp, "text", None)
    if not text:
        # Fallback: concatenate text parts if present
        try:
            parts = resp.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") for p in parts)
        except Exception:
            text = ""
    return _extract_json(text or "{}")
