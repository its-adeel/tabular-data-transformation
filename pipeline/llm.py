"""LLM-based structuring of one raw table zone into JSON, via Groq's
OpenAI-compatible chat completions endpoint."""
import os
import re
import json

PROMPT = """You are converting a raw markdown table extracted from a scraped municipal \
zoning ordinance into structured JSON. The scrape is often messy: spanning/merged header \
cells may have been duplicated across every column instead of written once, the real \
column header may sit a row or two below a blank markdown-header row, multiple sub-tables \
may be glued together with no blank line between them, and page headers/footers may be \
stuck onto a cell with no separator.

Return a JSON object of the form {"tables": [...]}. Each element of "tables" is one
logical table:
{
  "zone_group": string or null,   // e.g. a zoning-district group label, if the table has one
  "category": string or null,     // e.g. a sign-type/use-category label, if the table has one
  "table_title": string or null,
  "columns": [string, ...],
  "rows": [ {column_name: value, ...}, ... ],
  "notes": [string, ...]          // legend text, footnotes, anything that isn't a data row
}

Rules:
- Do not paraphrase, round, or "clean up" any value, code, symbol, or number -- copy it \
verbatim from the raw text.
- If a single raw cell contains both a short label and a longer restriction/note (e.g. \
"Wall (No sign may project more than 2 feet...)"), decide whether to keep it as one field \
or split it into a label + a note field -- but if you split it, the note text must still \
appear verbatim somewhere in your output.
- Strip page furniture (running headers/footers like a city/document name repeated verbatim \
with no table meaning) out of cell values, but do not delete regulatory text.
- If a broadcast row repeats the same string across every column, that string becomes ONE \
label, not N columns.

Raw table:
---
<<<TABLE>>>
---

Return only the JSON object, no commentary.
"""


def _load_dotenv(path=".env"):
    """Tiny .env loader -- avoids adding python-dotenv as a dependency."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def call_llm(table_text, model=None):
    """Real call via Groq's OpenAI-compatible chat completions endpoint.
    Requires GROQ_API_KEY in the environment or in a .env file."""
    import requests
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set (checked environment and .env)")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model or GROQ_MODEL,
            "messages": [{"role": "user", "content": PROMPT.replace("<<<TABLE>>>", table_text)}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)["tables"]


def transform_zone_safe(zone_text, llm_fn=call_llm, attempts=4):
    """Never raises. Returns (structured_or_None, error_or_None).
    A malformed generation (bad JSON, transport error, rate limit) is a
    transient failure worth retrying with backoff -- unlike a validation
    rejection, which is a real answer about the table and isn't retried
    here (that's the caller's job, against the *validator*, not this call)."""
    import time
    delay = 2.0
    last_err = "unknown error"
    for i in range(attempts):
        try:
            return llm_fn(zone_text), None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if i < attempts - 1:
                time.sleep(delay)
                delay *= 2
    return None, last_err
