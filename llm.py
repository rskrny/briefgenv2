import os
import json
from typing import Literal, Dict, Any, List

from prompts import SYSTEM_PROMPT, USER_PROMPT

Provider = Literal["OpenAI", "Gemini"]

def _openai_complete(model: str, system: str, user: str, temperature: float) -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content

def _gemini_complete(model: str, system: str, user: str, temperature: float) -> str:
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    genai.configure(api_key=api_key)
    generation_config = {"temperature": temperature, "response_mime_type": "application/json"}
    prompt = f"{system}\n\n{user}"
    model_obj = genai.GenerativeModel(model_name=model, generation_config=generation_config)
    resp = model_obj.generate_content(prompt)
    return resp.text

def complete_json(provider: Provider, model: str, system: str, user: str, temperature: float) -> Dict[str, Any]:
    raw = _openai_complete(model, system, user, temperature) if provider == "OpenAI" \
        else _gemini_complete(model, system, user, temperature)
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{"); end = raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start:end+1])
        return {}
