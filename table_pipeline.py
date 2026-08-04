"""
LLM-based transformer for the markdown tables embedded in the zoning CSVs'
`text` column, plus a validator that flags doubtful output for manual
review. See pipeline/ for the actual implementation:
    pipeline/zones.py      -- find table regions in raw text
    pipeline/llm.py        -- Groq call that structures one table into JSON
    pipeline/validator.py  -- flags output that doesn't verbatim-cover the raw cells
    pipeline/render.py     -- renders structured JSON back to text, builds search_text
    pipeline/db.py         -- driver: whole corpus -> zoning.db

Usage:
    echo 'GROQ_API_KEY=...' > .env
    python table_pipeline.py data/*.csv               # whole corpus -> zoning.db
Output:
    zoning.db      -- sqlite db with all rows + search_text, open with any db viewer
    review.jsonl   -- flagged table zones, for manual spot-checking
"""
from pipeline.db import main

if __name__ == "__main__":
    main()
