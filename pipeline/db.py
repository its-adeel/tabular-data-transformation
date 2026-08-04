"""Driver: reads every data/*.csv row, builds search_text, writes the
whole corpus into a single sqlite table for browsing in a db viewer."""
import csv
import glob
import json
import os
import sqlite3
import sys

from .llm import call_llm
from .render import build_search_text
from .zones import find_table_zones
from .validator import validate
from .cache import load_cache, save_cache, revalidate_cache, table_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT,
    chapter_no TEXT,
    heading_1 TEXT, heading_2 TEXT, heading_3 TEXT, heading_4 TEXT, heading_5 TEXT,
    sub_section_no TEXT,
    source_url TEXT,
    zoneomics_url TEXT,
    is_sub_chunk TEXT,
    sub_chunk_no INTEGER,
    text TEXT,
    search_text TEXT,
    has_table INTEGER,
    transform_status TEXT,   -- n/a | pending | transformed | flagged
    flag_reasons TEXT        -- non-empty only when transform_status = 'flagged'/'pending'
);
"""


def process_file_to_db(path, conn, llm_fn=call_llm, cache=None):
    review = []
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            search_text, zone_records, status = build_search_text(row['text'], llm_fn, cache=cache)
            reasons = [r for zr in zone_records for r in zr["flag_reasons"]]
            conn.execute(
                "INSERT INTO sections (source_file, chapter_no, heading_1, heading_2, heading_3, "
                "heading_4, heading_5, sub_section_no, source_url, zoneomics_url, is_sub_chunk, "
                "sub_chunk_no, text, search_text, has_table, transform_status, flag_reasons) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    os.path.basename(path), row.get("chapter_no"),
                    row.get("heading_1"), row.get("heading_2"), row.get("heading_3"),
                    row.get("heading_4"), row.get("heading_5"),
                    row.get("sub_section_no"), row.get("source_url"), row.get("zoneomics_url"),
                    row.get("is_sub_chunk"), row.get("sub_chunk_no"),
                    row["text"], search_text, int(bool(zone_records)), status,
                    " | ".join(reasons) if reasons else None,
                ),
            )
            if status == "flagged":
                for zr in zone_records:
                    if zr["flagged"]:
                        review.append({
                            "file": os.path.basename(path),
                            "sub_section_no": row["sub_section_no"],
                            "sub_chunk_no": row["sub_chunk_no"],
                            "source_url": row["source_url"],
                            **zr,
                        })
    return review


def main():
    paths = sys.argv[1:] or glob.glob("data/*.csv")
    db_path = "zoning.db"

    llm_fn = call_llm
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set -- rows with tables will be inserted with "
              "transform_status='pending' (search_text = raw text, untransformed).")
        llm_fn = None

    cache = load_cache()
    if cache:
        # Free re-check of every cached result against the current validator --
        # picks up any promotions/demotions from a validator change with zero
        # new API calls.
        raw_by_hash = {}
        for path in paths:
            with open(path, newline='', encoding='utf-8') as fh:
                for row in csv.DictReader(fh):
                    for z in find_table_zones(row['text']):
                        raw_by_hash[table_hash(z)] = z
        promo = revalidate_cache(cache, raw_by_hash, validate)
        if promo["promoted"] or promo["demoted"]:
            print(f"Revalidated cache: {promo['promoted']} promoted, {promo['demoted']} demoted")
        # sync cache's "flagged" flag with the just-revalidated "status"
        for entry in cache.values():
            entry["flagged"] = entry.get("status") == "flagged"

    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS sections")  # rebuild is cheap (SQL only); the
    conn.executescript(SCHEMA)                      # cache is what makes re-runs not re-pay
    with open("review.jsonl", "w") as review_out:
        for path in paths:
            review = process_file_to_db(path, conn, llm_fn, cache=cache)
            conn.commit()
            save_cache(cache)  # checkpoint after every file so a crash loses at most one file's calls
            print(f"{path} -> {db_path}  ({len(review)} flagged for review)")
            for rec in review:
                review_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    conn.close()
    print(f"\nDone. Open {db_path} in any sqlite viewer (e.g. DB Browser for SQLite).")
    print(f"Cache: {len(cache)} unique tables in table_cache.json (re-runs reuse these for free).")
