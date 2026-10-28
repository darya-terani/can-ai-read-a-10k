"""Free alternatives to a paid API: Google's Gemini (free tier) and local
open-source models through Ollama.

Every provider gets the same instructions, the same excerpt and the same
output schema, so differences in the results come from the models, not from
how they were asked.

Gemini:  get a free key at https://aistudio.google.com, then
         export GEMINI_API_KEY=...
         python -m tenkbench.run gemini:<model name shown in AI Studio>
Ollama:  install from https://ollama.com, then `ollama pull qwen2.5:7b`
         python -m tenkbench.run ollama:qwen2.5:7b
"""

from __future__ import annotations

import json
import os
import time

from .extract import SYSTEM, TOOL, Extraction, clean_values

SCHEMA = TOOL["input_schema"]

# Free tiers limit requests per minute; space calls out and back off on "too many requests".
GEMINI_PAUSE_SECONDS = float(os.environ.get("GEMINI_PAUSE_SECONDS", "6"))
MAX_RETRIES = 5


def extract_gemini(prompt: str, model: str, client=None) -> Extraction:
    from google import genai
    from google.genai import errors, types

    client = client or genai.Client()        # reads GEMINI_API_KEY from the environment
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        temperature=0,
        response_mime_type="application/json",
        response_json_schema=SCHEMA,
    )
    start = time.time()
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            break
        except (errors.ClientError, errors.ServerError) as err:
            # A DAILY quota won't clear by waiting a minute: stop with a clear message.
            # Finished companies are cached, so the next run resumes where this one stopped.
            if getattr(err, "code", None) == 429 and "PerDay" in str(err):
                raise SystemExit(
                    "\nGemini free-tier daily limit reached. Progress is saved; "
                    "run the same command again after the quota resets (usually midnight Pacific)."
                ) from None
            # 429 = per-minute rate limit; 500/503 = Google's servers temporarily overloaded.
            # Both are temporary: wait longer each time and try again.
            temporary = getattr(err, "code", None) in (429, 500, 503)
            if temporary and attempt < MAX_RETRIES - 1:
                wait = 30 * (attempt + 1)
                print(f"  Gemini busy (error {err.code}); retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    time.sleep(GEMINI_PAUSE_SECONDS)
    usage = response.usage_metadata
    return Extraction(
        clean_values(json.loads(response.text)),
        usage.prompt_token_count or 0,
        (usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", 0) or 0),
        time.time() - start,
    )


def _default_post(url: str, payload: dict) -> dict:
    import requests

    r = requests.post(url, json=payload, timeout=900)
    r.raise_for_status()
    return r.json()


def extract_ollama(prompt: str, model: str, post=None, host: str | None = None) -> Extraction:
    """Local model through Ollama's chat API with a JSON-schema output format.

    num_ctx matters: Ollama's default context window is only a few thousand
    tokens and would silently cut off most of the excerpt. 32,768 tokens fits
    the ~44,000-character excerpt with room to spare.
    """
    post = post or _default_post
    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 32768},
    }
    start = time.time()
    data = post(f"{host}/api/chat", payload)
    return Extraction(
        clean_values(json.loads(data["message"]["content"])),
        data.get("prompt_eval_count", 0),
        data.get("eval_count", 0),
        time.time() - start,
    )
