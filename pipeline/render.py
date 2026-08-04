"""Turn structured table JSON back into flat text, and assemble the
per-row `search_text`: clean zones get replaced by their rendered form,
flagged/pending zones are left as raw text until manually reviewed."""
from .zones import find_table_zones
from .llm import call_llm, transform_zone_safe
from .validator import validate
from .cache import table_hash


def render_table(t):
    lines = []
    head = " / ".join(x for x in [t.get("table_title"), t.get("zone_group"), t.get("category")] if x)
    if head:
        lines.append(head)
    cols = t.get("columns", [])
    if cols:
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for row in t.get("rows", []):
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    for note in t.get("notes", []):
        lines.append(f"Note: {note}")
    return "\n".join(lines)


def build_search_text(text, llm_fn=call_llm, cache=None):
    """Returns (search_text, zone_records, status).
    status is one of: 'n/a' (no table in this row), 'pending' (has a table
    but llm_fn is None -- not attempted), 'transformed' (all zones clean),
    'flagged' (at least one zone needs manual review).

    `cache` is a dict keyed by content hash (see pipeline/cache.py) -- a zone
    whose exact text was already processed in a previous run is resolved
    instantly with no API call, so a crash mid-run or a re-run after tuning
    the validator only pays for zones it hasn't seen before."""
    zones = find_table_zones(text)
    if not zones:
        return text, [], "n/a"
    if llm_fn is None:
        return text, [{"zone_index": i, "raw": z, "structured": None,
                        "flagged": None, "flag_reasons": ["not attempted -- no LLM configured"]}
                       for i, z in enumerate(zones)], "pending"

    zone_records = []
    search_text = text
    any_flagged = False
    for i, zone in enumerate(zones):
        h = table_hash(zone)
        cached = cache.get(h) if cache is not None else None
        if cached is not None:
            structured, flagged, reasons = cached["structured"], cached["flagged"], cached["reasons"]
        else:
            structured, err = transform_zone_safe(zone, llm_fn)
            if err is not None:
                flagged, reasons = True, [f"llm error: {err}"]
            else:
                flagged, reasons = validate(zone, structured)
            if cache is not None:
                cache[h] = {"structured": structured, "flagged": flagged, "reasons": reasons}

        any_flagged = any_flagged or flagged
        zone_records.append({"zone_index": i, "raw": zone, "structured": structured,
                              "flagged": flagged, "flag_reasons": reasons})
        if not flagged:
            search_text = search_text.replace(
                zone,
                render_table(structured[0]) if len(structured) == 1
                else "\n".join(render_table(t) for t in structured),
            )
        # if flagged: leave that zone's raw text in place, pending manual review
    return search_text, zone_records, ("flagged" if any_flagged else "transformed")
