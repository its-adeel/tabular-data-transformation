"""Content-hash cache for LLM table transformations.

Keyed by sha256(raw zone text), stored outside zoning.db so a full db
rebuild (cheap: it's just SQL) never discards already-paid-for LLM calls
(expensive: it's an API call per table). This is what makes a crash mid-run,
or a re-run after tuning the validator, resumable without re-paying for
every table that was already processed.
"""
import hashlib
import json
from pathlib import Path

CACHE_PATH = "table_cache.json"


def table_hash(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def load_cache(path: str = None) -> dict:
    path = Path(path or CACHE_PATH)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: could not read table cache {path} ({e}); starting empty")
        return {}


def save_cache(cache: dict, path: str = None) -> None:
    path = Path(path or CACHE_PATH)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp.replace(path)  # atomic, so an interrupted write can't corrupt the cache


def revalidate_cache(cache: dict, raw_by_hash: dict, validate_fn) -> dict:
    """Re-run `validate_fn(raw, structured)` over every cached entry without
    any API calls -- promotes entries a since-loosened validator now accepts,
    demotes ones a since-tightened validator now rejects. Keeps the cache
    honest whenever the validator changes, at zero cost."""
    promoted = demoted = 0
    for h, entry in cache.items():
        raw = raw_by_hash.get(h)
        if raw is None or entry.get("structured") is None:
            continue
        flagged, reasons = validate_fn(raw, entry["structured"])
        was_ok = entry.get("status") == "ok"
        now_ok = not flagged
        if now_ok and not was_ok:
            promoted += 1
        elif was_ok and not now_ok:
            demoted += 1
        entry["status"] = "ok" if now_ok else "flagged"
        entry["reasons"] = reasons
    return {"promoted": promoted, "demoted": demoted}
