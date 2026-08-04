"""Find markdown table regions inside a row's raw `text` field."""
import re

DIVIDER = re.compile(r'^\s*\|(?:\s*-{2,}\s*\|)+\s*$', re.MULTILINE)


def find_table_zones(text):
    """Contiguous runs of pipe-prefixed lines that contain at least one
    '| --- |' divider. A zone may hold more than one sub-table glued
    together (seen in real files) -- that's fine, the LLM step is told it
    may return several tables for one zone."""
    zones, current = [], []
    for line in text.split('\n'):
        if line.strip().startswith('|'):
            current.append(line)
        else:
            if current:
                zones.append('\n'.join(current))
                current = []
    if current:
        zones.append('\n'.join(current))
    return [z for z in zones if DIVIDER.search(z)]
