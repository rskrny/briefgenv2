# llm.py
from __future__ import annotations
import os, json, re
from typing import Any, Dict, Optional, List

def _get_secret(key: str) -> Optional[str]:
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
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"raw": text or ""}

# ---------------- OpenAI (text JSON) ----------------
def openai_json(prompt: str, *, model: Optional[str] = None, system: str = "", temperature: float = 0.2, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    from openai import OpenAI
    api_key = _get_secret("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
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

# --------------- OpenAI (images → JSON) ---------------
def openai_json_from_images(prompt: str, image_paths: List[str], *, model: Optional[str] = None, temperature: float = 0.2) -> Dict[str, Any]:
    from openai import OpenAI
    import base64
    api_key = _get_secret("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    model = model or os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
    parts = [{"type": "text", "text": prompt}]
    for p in image_paths:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":parts}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""
    return _extract_json(text)

# ---------------- Gemini (text JSON) ----------------
def gemini_json(prompt: str, *, model: Optional[str] = None, system: str = "", temperature: float = 0.2) -> Dict[str, Any]:
    import google.generativeai as genai
    api_key = _get_secret("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    genai.configure(api_key=api_key)
    model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    full_prompt = (system + "\n\n" + prompt).strip() if system else prompt
    g = genai.GenerativeModel(model)
    resp = g.generate_content(
        full_prompt,
        generation_config={"response_mime_type": "application/json", "temperature": temperature},
    )
    text = getattr(resp, "text", "") or ""
    if not text and getattr(resp, "candidates", None):
        text = "".join(getattr(p, "text", "") for p in resp.candidates[0].content.parts)
    return _extract_json(text)

# --------------- Gemini (images → JSON) ---------------
def gemini_json_from_images(prompt: str, image_paths: List[str], *, model: Optional[str] = None, temperature: float = 0.2) -> Dict[str, Any]:
    import google.generativeai as genai
    from PIL import Image
    api_key = _get_secret("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    genai.configure(api_key=api_key)
    model = model or os.getenv("GEMINI_VISION_MODEL", "gemini-1.5-pro")
    imgs = [Image.open(p) for p in image_paths]
    g = genai.GenerativeModel(model)
    resp = g.generate_content(
        [prompt] + imgs,
        generation_config={"response_mime_type": "application/json", "temperature": temperature},
    )
    text = getattr(resp, "text", "") or ""
    if not text and getattr(resp, "candidates", None):
        text = "".join(getattr(p, "text", "") for p in resp.candidates[0].content.parts)
    return _extract_json(text)
