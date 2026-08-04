"""Validator -- flags doubtful LLM output, never auto-decides.

Ported from a more mature build-time table normalizer in this project: the
key fix over a naive "every raw cell must survive verbatim" check is that
it only requires VERBATIM survival of the things that are actually the
answer to a matrix-cell question -- permit/zone codes ("P", "CUP", "AG",
"R-1", "P1", "MCUP10") and multi-digit numbers. Everything else (header
words, group labels, footnote markers, section citations) may be legitimately
reworded or folded by the LLM without being flagged. A blanket "every token
survives" check can't tell a defensible reword apart from real data loss;
this can.
"""
import re

# ---------------------------------------------------------------------------
# Cell classification
# ---------------------------------------------------------------------------
_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")
# Shape of a permit/zone code cell: short, no spaces (P, P1, CUP, MCUP, R-1,
# AG, TPZ, C*).
_CODE_SHAPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9*.\-/]{0,7}$")
_DIGIT_RE = re.compile(r"\d")
_LETTER_RE = re.compile(r"[A-Za-z]")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
# A bare footnote/list marker: "1", "1.", "(1)", "(12)" -- matches the code
# shape but is not a code; the LLM may renumber/reword these freely.
_FOOTNOTE_MARKER_RE = re.compile(r"^\(?\d{1,3}\)?\.?$")

MAX_CONTENT_CELL = 40          # cells longer than this are prose, excluded from coverage
MIN_CONTENT_COVERAGE = 0.6     # backstop against wholesale summarization, not a precise count
MIN_COLUMN_BINDING = 0.8       # fraction of rows that must name a declared column code


def _is_code(cell: str) -> bool:
    """True for a permit/zone code -- must survive verbatim. Excludes
    ordinary header words ("Zone", "Front") -- folding a spanning header
    into its sub-columns is desired behavior -- and bare footnote/list
    markers ("1", "(1)"), which are reworded freely by design. A real code
    is letter-containing and either all-uppercase ("P", "AG", "CUP", "R-1")
    or has a digit ("P1", "MCUP10")."""
    if not cell or not _CODE_SHAPE_RE.match(cell) or _FOOTNOTE_MARKER_RE.match(cell):
        return False
    return bool(_LETTER_RE.search(cell)) and (cell.isupper() or bool(_DIGIT_RE.search(cell)))


def _split_cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _pipe_rows(text: str) -> list[str]:
    return [ln for ln in text.split("\n") if ln.count("|") >= 2]


def _is_separator(row: str) -> bool:
    cells = _split_cells(row)
    return bool(cells) and any(cells) and all(c == "" or _SEP_CELL_RE.match(c) for c in cells)


def verbatim_cells(raw_text: str) -> set[str]:
    """Codes that must survive verbatim. A digit-bearing code ("P1",
    "MCUP10") is required on a single sighting. A pure-letter all-caps code
    ("P", "AG", "CUP") is ambiguous by shape alone (could be a column
    abbreviation header) -- required only if it recurs in 2+ rows, or
    appears in a dedicated all-code header row (see column_codes)."""
    rows = [r for r in _pipe_rows(raw_text) if not _is_separator(r)]
    with_digit, letters_only_counts = set(), {}
    for row in rows:
        for cell in _split_cells(row):
            if not _is_code(cell):
                continue
            if _DIGIT_RE.search(cell):
                with_digit.add(cell)
            else:
                letters_only_counts[cell] = letters_only_counts.get(cell, 0) + 1
    recurring = {c for c, n in letters_only_counts.items() if n >= 2}
    return with_digit | recurring | column_codes(raw_text)


def column_codes(raw_text: str) -> set[str]:
    """Codes from a dedicated sub-header row -- a row whose non-empty cells
    are ALL codes and number 2+, e.g. "| AG | AS |" or "| R1 | R2 | R3 |"."""
    for row in _pipe_rows(raw_text):
        if _is_separator(row):
            continue
        filled = [c for c in _split_cells(row) if c]
        if len(filled) >= 2 and all(_is_code(c) for c in filled):
            return set(filled)
    return set()


def content_cells(raw_text: str) -> set[str]:
    """Distinct short-ish cells carrying actual content, for a fractional
    coverage check (not strict) -- a correct transform legitimately folds
    some header/group labels away, but must not silently drop rows."""
    return {
        cell
        for row in _pipe_rows(raw_text) if not _is_separator(row)
        for cell in _split_cells(row)
        if cell and len(cell) <= MAX_CONTENT_CELL and _ALNUM_RE.search(cell)
        and not _FOOTNOTE_MARKER_RE.match(cell)
    }


def _contains_cell(haystack: str, cell: str) -> bool:
    # Word-boundary match so a bare "P" isn't satisfied by the "P" inside "Permitted".
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(cell)}(?![A-Za-z0-9])", haystack) is not None


# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"\$?\d[\d,]*\.?\d*%?")


def _extract_numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


def _significant_numbers(text: str) -> set[str]:
    # Single digits are footnote/list-marker noise ("[1]", "1."), not values --
    # strip a trailing "." before measuring length, or a numbered list marker
    # like "1." (len 2 as extracted) slips past this filter uncaught.
    return {n for n in _extract_numbers(text) if len(n.lstrip("$").rstrip(".")) > 1}


# A municipal-code citation embedded in cell text -- "Section 120.04.020",
# "19.36.100(B)" -- a supplementary cross-reference, not a substantive zoning
# value. Excluded from the *required* numbers so a reformatted citation
# doesn't get flagged as a dropped number; does not weaken the invented-number
# check, which still compares against every number in the raw text.
_CITATION_START_RE = (
    r"(?:\b(?:Sec(?:tions?)?|Ch(?:apters?)?|Art(?:icles?)?)\.?\s*"
    r"(?:[IVXLCDM]{1,6}\.)?\s*\d{1,4}(?:\.\d{1,4}){0,3}(?:\.?[A-Za-z])?(?:\([A-Za-z0-9]+\))?)"
    r"|(?:\b\d{1,4}(?:\.\d{1,4}){2,3}(?:\.?[A-Za-z])?(?:\([A-Za-z0-9]+\))?\b)"
    r"|(?:\b\d{1,4}\.\d{1,4}\([A-Za-z0-9]+\))"
)
_CITATION_CONT_RE = r"\d{1,4}\.\d{1,4}(?:\.\d{1,4})*(?:\.?[A-Za-z])?(?:\([A-Za-z0-9]+\))?"
_CITATION_REF_RE = re.compile(
    rf"(?:{_CITATION_START_RE})(?:\s*(?:,|;|and)\s*{_CITATION_CONT_RE})*", re.I
)


def _data_numbers(raw_text: str) -> set[str]:
    """Numbers in data positions -- excludes spanning title/caption rows
    (a single filled cell) and citation-reference numbers."""
    out = set()
    for row in _pipe_rows(raw_text):
        if _is_separator(row):
            continue
        cells = _split_cells(row)
        if sum(1 for c in cells if c) < 2:  # spanning title/caption row
            continue
        text = _CITATION_REF_RE.sub(" ", " ".join(cells))
        out |= _significant_numbers(text)
    return out


# ---------------------------------------------------------------------------
# Structured-output flattening (our transform emits JSON, not sentence-per-row
# text -- flatten it once so the same string-based checks above apply).
# ---------------------------------------------------------------------------
def flatten(obj) -> str:
    parts = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            parts.append(re.sub(r"\s+", " ", re.sub(r"\*\*", "", o)))

    walk(obj)
    return " \x1f ".join(parts)


def _row_haystacks(structured) -> list[str]:
    """One flattened string per output data row, for the column-binding
    check -- our JSON has real row boundaries, so this reads each row's
    dict directly rather than splitting text on newlines."""
    out = []
    if not isinstance(structured, list):
        return out
    for table in structured:
        for row in table.get("rows", []):
            out.append(flatten(row))
    return out


def validate(raw_text: str, structured) -> tuple[bool, list[str]]:
    """Returns (flagged: bool, reasons: list[str]). Catches gross failures --
    dropped codes, hallucinated numbers, summarizing instead of transcribing,
    inconsistent column binding. Deliberately does not claim to prove every
    cell landed under the right column -- verifying that would require the
    table's original structure, which the scrape destroyed."""
    reasons = []

    if not structured:
        return True, ["LLM returned no tables for a zone that contains a table divider"]

    # A raw block can be markdown-table SYNTAX wrapped around something that
    # isn't tabular data at all -- a figure/diagram caption, an equation laid
    # out for visual alignment, a legal boilerplate block with fill-in blanks.
    # Those have at most one or two short "content" cells (everything else is
    # long prose/captions, already excluded by content_cells' 40-char cap),
    # where a genuine data table has several (codes, numbers, short labels
    # repeated across rows). Zero structured rows for one of these isn't a
    # failure -- it's the correct answer, since there was never any data to
    # extract; only flag zero-rows when the raw block actually looks like data.
    looks_like_data = len(content_cells(raw_text)) >= 2
    if looks_like_data:
        for t in structured if isinstance(structured, list) else []:
            if not t.get("rows"):
                reasons.append(f"table '{t.get('table_title') or t.get('category')}' has zero rows")

    haystack = flatten(structured)

    missing_codes = sorted(c for c in verbatim_cells(raw_text) if not _contains_cell(haystack, c))
    if missing_codes:
        reasons.append(f"dropped {len(missing_codes)} code(s), e.g. {missing_codes[:5]}")

    cand_nums = _significant_numbers(haystack)
    dropped_nums = sorted(_data_numbers(raw_text) - cand_nums)
    if dropped_nums:
        reasons.append(f"dropped number(s): {dropped_nums[:5]}")

    invented_nums = sorted(cand_nums - _significant_numbers(raw_text))
    if invented_nums:
        reasons.append(f"invented number(s): {invented_nums[:5]}")

    # Consistent column binding -- a code appearing SOMEWHERE in the output
    # can pass the check above while most rows are silently left unbound.
    cols = column_codes(raw_text)
    rows = _row_haystacks(structured)
    if cols and len(rows) >= 2:
        bound = sum(1 for r in rows if any(_contains_cell(r, c) for c in cols))
        if 0 < bound < MIN_COLUMN_BINDING * len(rows):
            reasons.append(f"inconsistent column binding: only {bound}/{len(rows)} "
                            f"rows name a column ({sorted(cols)[:4]})")

    # Wholesale summarization / truncation, as a fractional coverage check
    # rather than a strict one (folding headers legitimately drops some cells).
    content = content_cells(raw_text)
    if content:
        kept = sum(1 for c in content if c in haystack)
        coverage = kept / len(content)
        if coverage < MIN_CONTENT_COVERAGE:
            reasons.append(f"only {100 * coverage:.0f}% of cells survived (summarized/truncated?)")

    return (len(reasons) > 0), reasons
